import numpy as np
from parameters import interp_method,gyro_E,dpot_fourier_maxm
from scipy.interpolate import griddata,interp1d
from scipy.signal import find_peaks
import math
import setup
import myinterp

def varTwoD(x2d,y2d,f2d,xin,yin):
  Ny,Nx=np.shape(x2d)
  x0=x2d[0,0]
  dx=x2d[0,1]-x0
  y0=y2d[0,0]
  dy=y2d[1,0]-y0
  ix=math.floor((xin-x0)/dx)
  wx=(xin-x0)/dx-ix
  iy=math.floor((yin-y0)/dy)
  wy=(yin-y0)/dy-iy
  if (ix<0) or (ix>Nx-2) or (iy<0) or (iy>Ny-2):
    fout=np.nan
  else:
    fout=f2d[iy,ix]*(1-wy)*(1-wx) + f2d[iy+1,ix]*wy*(1-wx)\
        +f2d[iy,ix+1]*(1-wy)*wx + f2d[iy+1,ix+1]*wy*wx

  return ix,iy,wx,wy,fout

def init(pot0fac,dpotfac,Nr,Nz,comm,summation,use_gpu):
  rank=comm.Get_rank()
  size=comm.Get_size()
  #read mesh
  global rlin,zlin,R,Z,psi2d,psi_surf,psix,Ra,Ba,rsurf,zsurf,theta,dist,zx
  if rank==0:
    rz,psi_rz,rx,zx,psi_surf,psix,Ba,rsurf,zsurf,theta,dist=setup.Grid(Nr,Nz)
  else:
    rz,psi_rz,rx,zx,psi_surf,psix,Ba,rsurf,zsurf,theta,dist=[None]*11
  rz,psi_rz,rx,zx,psi_surf,psix,Ba,rsurf,zsurf,theta,dist\
            =comm.bcast((rz,psi_rz,rx,zx,psi_surf,psix,Ba,rsurf,zsurf,theta,dist),root=0)
  rmesh=rz[:,0]
  zmesh=rz[:,1]
  rlin=np.linspace(min(rmesh),max(rmesh),Nr)
  zlin=np.linspace(min(zmesh),max(zmesh),Nz)
  Ra=(min(rmesh)+max(rmesh))/2 #major radius

  itask1,itask2,itask_list=simple_partition(comm,4,size)
  itask1=itask1[rank]
  itask2=itask2[rank]
  R,Z=np.meshgrid(rlin,zlin)
  if (itask1<=0)and(0<=itask2):
    psi2d=griddata(rz,psi_rz,(R,Z),method=interp_method)
  else:
    psi2d=np.zeros((Nz,Nr),dtype=float)
  #read B field
  global Bmag,Br,Bz,Bphi,br,bz,bphi
  Br,Bz,Bphi=setup.Bfield(rz,rlin,zlin,itask1,itask2)
  #MPI reduction after performing griddata interpoldation on psi and B
  psi2d=comm.allreduce(psi2d,op=summation)
  Br=comm.allreduce(Br,op=summation)
  Bz=comm.allreduce(Bz,op=summation)
  Bphi=comm.allreduce(Bphi,op=summation)
  Bmag=np.sqrt(Br**2+Bz**2+Bphi**2)
  br=Br/Bmag
  bz=Bz/Bmag
  bphi=Bphi/Bmag
  #read potential
  global pot0,dpot
  global Er00,Ez00,Ephi00
  global Er0m,Ez0m,Ephi0m
  if (pot0fac>0)or(dpotfac>0):
    itask1,itask2,ntasks_list=simple_partition(comm,Nr*Nz,size)
    itask1=itask1[rank]
    itask2=itask2[rank]
    setup.Pot_init(rank)
    setup.grid_deriv_init()
    inode1,inode2,_=simple_partition(comm,setup.nnode,size)
    inode1=inode1[rank]
    inode2=inode2[rank]
    setup.get_grid_E(inode1,inode2,comm,summation)
    if use_gpu:
      setup.node_to_2d_init_gpu(itask1,itask2,rlin,zlin)
    else:
      setup.node_to_2d_init(itask1,itask2,rlin,zlin)
    #end if use_gpu
    pot0,dpot,Er00,Ez00,Er0m,Ez0m=setup.efields(rlin,zlin,itask1,itask2,use_gpu)
    pot0=comm.allreduce(pot0fac*pot0,op=summation)
    dpot=comm.allreduce(dpotfac*dpot,op=summation)
    Er00=comm.allreduce(Er00,op=summation)
    Ez00=comm.allreduce(Ez00,op=summation)
    Er0m=comm.allreduce(Er0m,op=summation)
    Ez0m=comm.allreduce(Ez0m,op=summation)
  else:
    pot0=np.zeros((Nz,Nr),dtype=float)
    dpot=np.zeros((Nz,Nr),dtype=float)
  #pin the starting node of the surface at the top or bottom
  if Bphi[math.floor(Nz/2),math.floor(Nr/2)]>0:
    theta0=+np.pi/2
  else:
    theta0=-np.pi/2
  for i in range(np.size(theta)):
    if theta[i]<=theta0: theta[i]=theta[i]+2*np.pi

  idx=np.argsort(theta)
  theta=theta[idx]
  rsurf=rsurf[idx]
  zsurf=zsurf[idx]
  dist=dist[idx]
  #calculate grad B, curl B, and curl b
  global gradBr,gradBz,gradBphi,curlBr,curlBz,curlBphi,curlbr,curlbz,curlbphi
  gradBr,gradBz,gradBphi=setup.Grad(rlin,zlin,Bmag,Nr,Nz)
  curlBr,curlBz,curlBphi=setup.Curl(rlin,zlin,Br,Bz,Bphi,Nr,Nz)
  curlbr,curlbz,curlbphi=setup.Curl(rlin,zlin,br,bz,bphi,Nr,Nz)

  Ephi00=np.zeros((Nz,Nr),dtype=float)
  Ephi0m=np.zeros((Nz,Nr),dtype=float)

  if (not gyro_E):
    global imu1
    imu1=0
  return

def gyropot_gpu(comm,mu_arr,qi,mi,ngyro,pot0fac,dpotfac,iorb1,iorb2):
  import cupy as cp
  gyro_pot_kernel=cp.RawKernel(r'''
  extern "C" __global__
  void gyro_pot(double mu,double* Bmag,double* rlin,double* zlin,double* pot0,double* dpot,\
       double* gyropot0,double* gyrodpot,double mi,double qi,int Nz,int Nr,int ngyro)
  {
    int iz,ir,igyro,idx,ir1,iz1;
    double B, r, z, rho, angle, r1, z1, tmp1, tmp2;
    double r0,z0,dr,dz,wr,wz;
    r0=rlin[0];
    z0=zlin[0];
    dr=rlin[1]-rlin[0];
    dz=zlin[1]-zlin[0];
    iz=blockIdx.x;
    ir=threadIdx.x;
    while (ir<Nr){
      idx=iz*Nr+ir;
      B=Bmag[idx];
      r=rlin[ir];
      z=zlin[iz];
      if (isnan(B)){
        gyropot0[idx]=nan("");
        gyrodpot[idx]=nan("");
        ir=ir+blockDim.x;
        continue;
      }
      gyropot0[idx]=0.0;
      gyrodpot[idx]=0.0;
      rho=sqrt(2*mi*mu/B/qi/qi);
      for (igyro=0;igyro<ngyro;igyro++){
        angle=8.*atan(1.)*double(igyro)/double(ngyro);
        r1=r+rho*cos(angle);
        z1=z+rho*sin(angle);
        ir1=floor((r1-r0)/dr);
        iz1=floor((z1-z0)/dz);
        wr=(r1-r0)/dr-ir1;
        wz=(z1-z0)/dz-iz1;
        if ((ir1<0)||(ir1>Nr-2)||(iz1<0)||(iz1>Nz-2)){
          tmp1=nan("");
          tmp2=nan("");
        }else{
          tmp1=pot0[iz1*Nr+ir1]*(1-wz)*(1-wr)+pot0[(iz1+1)*Nr+ir1]*wz*(1-wr)\
              +pot0[iz1*Nr+ir1+1]*(1-wz)*wr+pot0[(iz1+1)*Nr+ir1+1]*wz*wr;
          tmp2=dpot[iz1*Nr+ir1]*(1-wz)*(1-wr)+dpot[(iz1+1)*Nr+ir1]*wz*(1-wr)\
              +dpot[iz1*Nr+ir1+1]*(1-wz)*wr+dpot[(iz1+1)*Nr+ir1+1]*wz*wr;
        } 
        gyropot0[idx]+=tmp1/double(ngyro);
        gyrodpot[idx]+=tmp2/double(ngyro);
      }
      ir=ir+blockDim.x;
    }
  }
  ''','gyro_pot')

  global gyropot0,gyrodpot
  Nz,Nr=np.shape(Er00)
  nthreads=min(Nr,1024)
  global imu1,imu2
  from parameters import nPphi,nH
  imu1=int(iorb1/(nPphi*nH))
  imu2=int(iorb2/(nPphi*nH))
  Nmu_l=imu2-imu1+1
  gyropot0=np.zeros((Nz,Nr,Nmu_l),dtype=float)
  gyrodpot=np.zeros((Nz,Nr,Nmu_l),dtype=float)
  if (pot0fac==0)and(dpotfac==0): return
  Bmag_gpu=cp.asarray(Bmag,dtype=cp.float64).ravel(order='C')
  rlin_gpu=cp.asarray(rlin,dtype=cp.float64)
  zlin_gpu=cp.asarray(zlin,dtype=cp.float64)
  pot0_gpu=cp.asarray(pot0,dtype=cp.float64).ravel(order='C')
  dpot_gpu=cp.asarray(dpot,dtype=cp.float64).ravel(order='C')
  gyropot0_gpu=cp.zeros((Nz*Nr),dtype=cp.float64)
  gyrodpot_gpu=cp.zeros((Nz*Nr),dtype=cp.float64)
  for imu in range(Nmu_l):
    mu=mu_arr[imu+imu1]
    gyro_pot_kernel((Nz,),(nthreads,),(mu,Bmag_gpu,rlin_gpu,zlin_gpu,pot0_gpu,dpot_gpu,\
                          gyropot0_gpu,gyrodpot_gpu,mi,qi,int(Nz),int(Nr),int(ngyro)))
    gyropot0[:,:,imu]=cp.asnumpy(gyropot0_gpu).reshape((Nz,Nr),order='C')
    gyrodpot[:,:,imu]=cp.asnumpy(gyrodpot_gpu).reshape((Nz,Nr),order='C')
  
  comm.barrier()#just to avoid accidental deletion before completion
  del Bmag_gpu,rlin_gpu,zlin_gpu,pot0_gpu,dpot_gpu,gyropot0_gpu,gyrodpot_gpu
  mempool = cp.get_default_memory_pool()
  pinned_mempool = cp.get_default_pinned_memory_pool()
  mempool.free_all_blocks()
  pinned_mempool.free_all_blocks()
  return 

def gyropot(comm,mu_arr,qi,mi,ngyro,summation,pot0fac,dpotfac):
  global gyropot0,gyrodpot
  Nz,Nr=np.shape(Er00)
  Nmu=np.size(mu_arr)
  rank=comm.Get_rank()
  size=comm.Get_size()
  #parition in (r,z,mu) just like the orbit partitioning
  itask1,itask2,ntasks_list=simple_partition(comm,Nr*Nz*Nmu,size)
  itask1=itask1[rank]
  itask2=itask2[rank]
  #determine local range of mu
  global imu1,imu2
  iz=int(itask1/(Nr*Nmu))
  ir=int((itask1-iz*Nr*Nmu)/Nmu)
  imu1=itask1-iz*Nr*Nmu-ir*Nmu
  iz=int(itask2/(Nr*Nmu))
  ir=int((itask2-iz*Nr*Nmu)/Nmu)
  imu2=itask2-iz*Nr*Nmu-ir*Nmu
  Nmu_l=imu2-imu1+1

  dr=rlin[1]-rlin[0]
  dz=zlin[1]-zlin[0]
  r0=rlin[0]
  z0=zlin[0]
  gyropot0=np.zeros((Nz,Nr,Nmu_l),dtype=float)
  gyrodpot=np.zeros((Nz,Nr,Nmu_l),dtype=float)
  if (pot0fac==0)and(dpotfac==0): return
  for itask in range(itask1,itask2+1):
    iz=int(itask/(Nr*Nmu))
    ir=int((itask-iz*Nr*Nmu)/Nmu)
    imu=itask-iz*Nr*Nmu-ir*Nmu
    mu=mu_arr[imu]
    B=Bmag[iz,ir]
    r=rlin[ir]
    z=zlin[iz]
    if np.isnan(B):
      gyropot0[iz,ir,imu-imu1]=np.nan
      gyrodpot[iz,ir,imu-imu1]=np.nan
      continue
    rho=np.sqrt(2*mi*mu/B/qi**2)
    gyropot0[iz,ir,imu-imu1]=0.0
    gyrodpot[iz,ir,imu-imu1]=0.0
    for igyro in range(ngyro):
      angle=2*np.pi*float(igyro)/float(ngyro)
      r1=r+rho*np.cos(angle)
      z1=z+rho*np.sin(angle)
      ir1=math.floor((r1-r0)/dr)
      wr=(r1-r0)/dr-ir1
      iz1=math.floor((z1-z0)/dz)
      wz=(z1-z0)/dz-iz1
      if (ir1<0) or (ir1>Nr-2) or (iz1<0) or (iz1>Nz-2):
        gyropot0[iz,ir,imu-imu1]=np.nan
        gyrodpot[iz,ir,imu-imu1]=np.nan
        tmp1=np.nan
        tmp2=np.nan
      else:
        tmp1=pot0[iz1,ir1]*(1-wz)*(1-wr) + pot0[iz1+1,ir1]*wz*(1-wr)\
            +pot0[iz1,ir1+1]*(1-wz)*wr + pot0[iz1+1,ir1+1]*wz*wr
        tmp2=dpot[iz1,ir1]*(1-wz)*(1-wr) + dpot[iz1+1,ir1]*wz*(1-wr)\
            +dpot[iz1,ir1+1]*(1-wz)*wr + dpot[iz1+1,ir1+1]*wz*wr
        gyropot0[iz,ir,imu-imu1]=gyropot0[iz,ir,imu-imu1]+tmp1/float(ngyro)
        gyrodpot[iz,ir,imu-imu1]=gyrodpot[iz,ir,imu-imu1]+tmp2/float(ngyro)
  #end for itask
  for imu in range(Nmu):
    if (imu>=imu1)and(imu<=imu2):
      tmp=gyropot0[:,:,imu-imu1]
      tmp=comm.allreduce(tmp,op=summation)
      gyropot0[:,:,imu-imu1]=tmp
    else:
      tmp=np.zeros((Nz,Nr),dtype=float)
      tmp=comm.allreduce(tmp,op=summation)
    comm.barrier()
    if (imu>=imu1)and(imu<=imu2):
      tmp=gyrodpot[:,:,imu-imu1]
      tmp=comm.allreduce(tmp,op=summation)
      gyrodpot[:,:,imu-imu1]=tmp
    else:
      tmp=np.zeros((Nz,Nr),dtype=float)
      tmp=comm.allreduce(tmp,op=summation)
  return

def efield(imu):
  myEr00=Er00
  myEz00=Ez00
  Nz,Nr=np.shape(Er00)
  if gyro_E:
    #To avoid the dot product between b and the ungyroaveraged E00
    myEr0m,myEz0m,myEphi0m=setup.Grad(rlin,zlin,gyropot0[:,:,imu]+gyrodpot[:,:,imu],Nr,Nz)
    myEr0m=-myEr0m-Er00
    myEz0m=-myEz0m-Ez00
  else:
    myEr0m=Er0m
    myEz0m=Ez0m
  
  return myEr00,myEz00,myEr0m,myEz0m

def potfield(imu):
  if gyro_E:
    pot=gyropot0[:,:,imu]+gyrodpot[:,:,imu]
  else:
    pot=pot0+dpot
  return pot

def H_arr(comm,qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr,summation):
  global dH
  nsurf=np.size(rsurf)#number of mesh points along the surface
  Bsurf=np.zeros((nsurf,),dtype=float)#magnitude of B 
  dpotsurf=np.zeros((nsurf,),dtype=float)#m/=0 potential along the surface
  covbphisurf=np.zeros((nsurf,),dtype=float)#covariant component \vec{b}\cdot(dX/d\phi)
  ix=np.zeros((nsurf,),dtype=int)
  iy=np.zeros((nsurf,),dtype=int)
  wx=np.zeros((nsurf,),dtype=float)
  wy=np.zeros((nsurf,),dtype=float)
  for i in range(nsurf):
    r,z=rsurf[i],zsurf[i]
    ix[i],iy[i],wx[i],wy[i],Bsurf[i]=varTwoD(R,Z,Bmag,r,z)
    if np.isnan(Bsurf[i]):
      if comm.Get_rank()==0: print('Setting up H_arr: something wrong at',i,'th point on the surface.')
      exit()
    if not(gyro_E):
      dpotsurf[i]=dpot[iy[i],ix[i]]*(1-wy[i])*(1-wx[i]) + dpot[iy[i]+1,ix[i]]*wy[i]*(1-wx[i])\
                 +dpot[iy[i],ix[i]+1]*(1-wy[i])*wx[i] + dpot[iy[i]+1,ix[i]+1]*wy[i]*wx[i]

    Bphisurf=Bphi[iy[i],ix[i]]*(1-wy[i])*(1-wx[i]) + Bphi[iy[i]+1,ix[i]]*wy[i]*(1-wx[i])\
               +Bphi[iy[i],ix[i]+1]*(1-wy[i])*wx[i] + Bphi[iy[i]+1,ix[i]+1]*wy[i]*wx[i]
    covbphisurf[i]=r*Bphisurf/Bsurf[i]
  
  if not(gyro_E):
    #smooth dpotsurf
    tmp=np.fft.fft(dpotsurf)
    freq=nsurf*np.fft.fftfreq(nsurf)
    tmp[abs(freq)>dpot_fourier_maxm]=0
    dpotsurf=np.real(np.fft.ifft(tmp))
  else:
    dpotsurf_mu=np.zeros((nsurf,nmu),dtype=float)
    for imu in range(nmu):
      if (imu>=imu1)and(imu<=imu2):
        #do the calculation if imu is in my range
        count=1
        tmp=gyrodpot[:,:,imu-imu1]+gyropot0[:,:,imu-imu1]-pot0
        for i in range(nsurf):
          dpotsurf_mu[i,imu]=tmp[iy[i],ix[i]]*(1-wy[i])*(1-wx[i]) + tmp[iy[i]+1,ix[i]]*wy[i]*(1-wx[i])\
                   +tmp[iy[i],ix[i]+1]*(1-wy[i])*wx[i] + tmp[iy[i]+1,ix[i]+1]*wy[i]*wx[i]
      else:
        count=0
      dpotsurf_mu[:,imu]=comm.allreduce(dpotsurf_mu[:,imu],op=summation)
      count=comm.allreduce(count,op=summation)
      dpotsurf_mu[:,imu]=dpotsurf_mu[:,imu]/float(count)
 
  #partition tasks
  rank=comm.Get_rank()
  size=comm.Get_size()
  itask1,itask2,ntasks_list=simple_partition(comm,nmu*nPphi,size)
  itask1=itask1[rank]
  itask2=itask2[rank]

  dH=np.zeros((nmu,nPphi,nH),dtype=float)
  #crossing locations of the orbits with the LCFS
  r_beg=np.zeros((nmu,nPphi,nH),dtype=float)
  z_beg=np.zeros((nmu,nPphi,nH),dtype=float)
  r_end=np.zeros((nmu,nPphi,nH),dtype=float)
  z_end=np.zeros((nmu,nPphi,nH),dtype=float)
  for itask in range(itask1,itask2+1):
    imu=int(itask/(nPphi))
    iPphi=itask-imu*nPphi
    if gyro_E:
      dpotsurf[:]=dpotsurf_mu[:,imu]
      #smooth dpotsurf
      tmp=np.fft.fft(dpotsurf)
      freq=nsurf*np.fft.fftfreq(nsurf)
      tmp[abs(freq)>dpot_fourier_maxm]=0
      dpotsurf=np.real(np.fft.ifft(tmp))
    #end if gyroE
    Hsurf=np.zeros((nsurf,),dtype=float)
    for isurf in range(nsurf): 
      #The zonal potential is not needed for H.
      Hsurf[isurf]=(Pphi_arr[iPphi]-qi*psi_surf)**2/covbphisurf[isurf]**2/2/mi\
                   +mu_arr[imu]*Bsurf[isurf]+qi*dpotsurf[isurf]

    #invert the sign of H if Bphi>0, so the entrance is always at \partial H/partial\theta>0
    Nz,Nr=np.array(np.shape(Bphi))
    if Bphi[math.floor(Nz/2),math.floor(Nr/2)]>0: Hsurf=-Hsurf

    pks1,_=find_peaks(Hsurf)
    pks2,_=find_peaks(-Hsurf)
    if (np.size(pks1)>1)or(np.size(pks2)>1):#rare cases when H has multiple peaks&troughs
      print('H(theta) has multiple peaks at imu=',imu,'iPphi=',iPphi,flush=True)
      theta_total=0.0
      #first, determine number of nodes where \partial H/\partial\theta>0 
      for isurf in range(nsurf):
        inext=(isurf+1)%nsurf
        iprev=(isurf-1)%nsurf
        if(Hsurf[inext]>Hsurf[isurf])and(Hsurf[isurf]>Hsurf[iprev]):
          thetar=theta[inext]
          thetal=theta[iprev]
          if thetar<thetal: thetar=thetar+2*np.pi
          theta_total=theta_total+(thetar-thetal)/2.0
         
      dtheta=theta_total/float(nH+1)
      iH=-1
      thetam=theta[0]+dtheta/2.0#starting from this location so that thetal=theta[0]
      while (thetam<theta[nsurf-1])and(iH<nH-1):
        thetal=thetam-dtheta/2.0
        thetar=thetam+dtheta/2.0
        Hm=myinterp.OneD_NL(theta,Hsurf,thetam)
        Hr=myinterp.OneD_NL(theta,Hsurf,thetar)
        Hl=myinterp.OneD_NL(theta,Hsurf,thetal)
        if(Hr>Hm)and(Hm>Hl):
          iH=iH+1   
          dH[imu,iPphi,iH]=Hr-Hl
          r_beg[imu,iPphi,iH]=myinterp.OneD_NL(theta,rsurf,thetam)
          z_beg[imu,iPphi,iH]=myinterp.OneD_NL(theta,zsurf,thetam)
          #When moving to the left, H first decreases, then reaches minimum, then increases
          while Hl<Hm:
            thetal=thetal-dtheta
            if thetal<theta[0]: thetal=thetal+2*np.pi
            Hl=myinterp.OneD_NL(theta,Hsurf,thetal)
          thetar=thetal+dtheta
          if thetar>=theta[0]+2*np.pi: thetar=thetar-2*np.pi
          Hr=myinterp.OneD_NL(theta,Hsurf,thetar)
          if thetar<thetal: thetar=thetar+2*np.pi
          wH=(Hl-Hm)/(Hl-Hr)
          theta_end=(1-wH)*thetal+wH*thetar
          if theta_end>=theta[0]+2*np.pi: theta_end=theta_end-2*np.pi
          r_end[imu,iPphi,iH]=myinterp.OneD_NL(theta,rsurf,theta_end)
          z_end[imu,iPphi,iH]=myinterp.OneD_NL(theta,zsurf,theta_end)
        #end if
        thetam=thetam+dtheta
      while (iH<nH-1): 
        iH=iH+1
        dH[imu,iPphi,iH]=0 
        r_beg[imu,iPphi,iH]=rsurf[0]
        z_beg[imu,iPphi,iH]=zsurf[0]
        r_end[imu,iPphi,iH]=rsurf[0]
        z_end[imu,iPphi,iH]=zsurf[0]
    else:#most likely H only has one minimum and one maximum as follows
      Hmin,minloc=min(Hsurf),np.argmin(Hsurf)
      Hmax,maxloc=max(Hsurf),np.argmax(Hsurf)
      thetamax=theta[maxloc]
      thetamin=theta[minloc]
      dtheta=(thetamax-thetamin)/float(nH+1)
      for iH in range(nH):
        thetam=thetamin+dtheta*float(iH+1)
        thetal=thetam-dtheta/2.0
        thetar=thetam+dtheta/2.0
        r_beg[imu,iPphi,iH]=myinterp.OneD_NL(theta,rsurf,thetam)
        z_beg[imu,iPphi,iH]=myinterp.OneD_NL(theta,zsurf,thetam)
        Hm=myinterp.OneD_NL(theta,Hsurf,thetam)
        Hr=myinterp.OneD_NL(theta,Hsurf,thetar)
        Hl=myinterp.OneD_NL(theta,Hsurf,thetal)
        dH[imu,iPphi,iH]=Hr-Hl
        if Hm<=Hsurf[0]:
          endloc=np.argmin(abs(Hsurf[0:minloc]-Hm))
        else:
          endloc=maxloc+np.argmin(abs(Hsurf[maxloc:]-Hm))

        if Hm>=Hsurf[endloc]:
          wH=(Hm-Hsurf[endloc])/(Hsurf[endloc-1]-Hsurf[endloc])
          r_end[imu,iPphi,iH]=(1-wH)*rsurf[endloc]+wH*rsurf[endloc-1]
          z_end[imu,iPphi,iH]=(1-wH)*zsurf[endloc]+wH*zsurf[endloc-1]
        else:
          if endloc==nsurf-1:
            nextloc=0
          else:
            nextloc=endloc+1
          wH=(Hm-Hsurf[endloc])/(Hsurf[nextloc]-Hsurf[endloc])
          r_end[imu,iPphi,iH]=(1-wH)*rsurf[endloc]+wH*rsurf[nextloc]
          z_end[imu,iPphi,iH]=(1-wH)*zsurf[endloc]+wH*zsurf[nextloc]
    #end if multipeaks
  #end for itask
  dH=comm.allreduce(dH,op=summation)
  r_beg=comm.allreduce(r_beg,op=summation)
  z_beg=comm.allreduce(z_beg,op=summation)
  r_end=comm.allreduce(r_end,op=summation)
  z_end=comm.allreduce(z_end,op=summation)
  return r_beg,z_beg,r_end,z_end

def partition_orbits2(comm,partition_opt,nmu,nPphi,nH):
  rank = comm.Get_rank()
  size = comm.Get_size()
  if (partition_opt):#partition orbits based on the actual time used
    if rank==0:
      #read from file
      from parameters import input_dir
      fid=open(input_dir+'/time.txt','r')
      size_in=int(fid.readline(8))
      fid.readline(1)
      time_in=np.zeros((size_in),dtype=float)
      for isize in range(size_in):
        time_in[isize]=fid.readline(6)
        fid.readline(1)
      if int(fid.readline(8))!=-1: print('Wrong ending of time.txt!')
      norb=nmu*nPphi*nH
      tau_opt=np.zeros((norb,),dtype=float)
      for iorb in range(norb):
        ind=float(iorb)/float(norb)*float(size_in)
        ind=math.floor(ind)
        tau_opt[iorb]=time_in[ind]

      sum_tau=sum(tau_opt)
      avg_tau=float(sum_tau)/float(size)
      accuml_tau=0
      iorb1_list=np.zeros((size,),dtype=int)
      iorb2_list=np.zeros((size,),dtype=int)
      iorb1_list[:]=1
      iorb2_list[:]=norb
      iorb=0
      for ipe in range(size-1):
        iorb=iorb+1
        accuml_tau=accuml_tau+tau_opt[iorb-1]
        while ((accuml_tau<avg_tau)and(norb-iorb>size-ipe)):
          iorb=iorb+1
          accuml_tau=accuml_tau+tau_opt[iorb-1]
   
        accuml_tau=accuml_tau-avg_tau
        iorb2_list[ipe]=iorb
        iorb1_list[ipe+1]=iorb+1
    else:#for rank!=0
      iorb1_list,iorb2_list=[None]*2
    iorb1_list,iorb2_list=comm.bcast((iorb1_list,iorb2_list),root=0)
    iorb1=iorb1_list[rank]-1#start from 0
    iorb2=iorb2_list[rank]-1
    mynorb=iorb2-iorb1+1
    norb_list=iorb2_list-iorb1_list+1

  else:#parition orbit based on orbit indices, will cause large load imbalance
    iorb1,iorb2,norb_list=simple_partition(comm,nmu*nPphi*nH,size)
    iorb1=iorb1[rank]
    iorb2=iorb2[rank] 
  return iorb1,iorb2,norb_list


def partition_orbits(comm,partition_opt,nmu,nPphi,nH):
  rank = comm.Get_rank()
  size = comm.Get_size()
  if (partition_opt):#partition orbits based on the orbit length
    if rank==0:
      #read from file
      from parameters import input_dir
      fid=open(input_dir+'/tau.txt','r')
      nmu_tau=int(fid.readline(8))
      nPphi_tau=int(fid.readline(8))
      nH_tau=int(fid.readline(8))
      fid.readline(1)
      tau_file=np.zeros((nmu_tau,nPphi_tau,nH_tau),dtype=float)
      norb_tau=nmu_tau*nPphi_tau*nH_tau
      for iorb in range(norb_tau):
        imu=int(iorb/(nPphi_tau*nH_tau))
        iPphi=int((iorb-imu*nPphi_tau*nH_tau)/nH_tau)
        iH=iorb-imu*nPphi_tau*nH_tau-iPphi*nH_tau
        value=fid.readline(20)
        tau_file[imu,iPphi,iH]=value
        if ((iorb+1)%4)==0: fid.readline(1)
      fid.close()
      #interpolate
      wmu=np.zeros((2,),dtype=float)
      wPphi=np.zeros((2,),dtype=float)
      wH=np.zeros((2,),dtype=float)
      norb=nmu*nPphi*nH
      tau_opt=np.zeros((norb,),dtype=float)
      for iorb in range(norb):
        imu=int(iorb/(nPphi*nH))
        iPphi=int((iorb-imu*nPphi*nH)/nH)
        iH=iorb-imu*nPphi*nH-iPphi*nH
        wmu[0]=float(imu*(nmu_tau-1))/float(nmu-1)
        imu_tau=math.floor(wmu[0])
        wmu[1]=wmu[0]-float(imu_tau)
        wmu[0]=1.0-wmu[1]
        if imu==nmu-1:
          wmu=[0,1]
          imu_tau=nmu_tau-2
        wPphi[0]=float(iPphi*(nPphi_tau-1))/float(nPphi-1)
        iPphi_tau=math.floor(wPphi[0])
        wPphi[1]=wPphi[0]-float(iPphi_tau)
        wPphi[0]=1.0-wPphi[1]
        if iPphi==nPphi-1:
          wPphi=[0,1]
          iPphi_tau=nPphi_tau-2
        wH[0]=float(iH*(nH_tau-1))/float(nH-1)
        iH_tau=math.floor(wH[0])
        wH[1]=wH[0]-float(iH_tau)
        wH[0]=1.0-wH[1]
        if iH==nH-1:
          wH=[0,1]
          iH_tau=nH_tau-2
        tmp=tau_file[imu_tau:imu_tau+2,iPphi_tau:iPphi_tau+2,iH_tau:iH_tau+2]
        tau_opt[iorb]=tmp[0,0,0]*wmu[0]*wPphi[0]*wH[0]+tmp[0,0,1]*wmu[0]*wPphi[0]*wH[1]\
                             +tmp[0,1,0]*wmu[0]*wPphi[1]*wH[0]+tmp[0,1,1]*wmu[0]*wPphi[1]*wH[1]\
                             +tmp[1,0,0]*wmu[1]*wPphi[0]*wH[0]+tmp[1,0,1]*wmu[1]*wPphi[0]*wH[1]\
                             +tmp[1,1,0]*wmu[1]*wPphi[1]*wH[0]+tmp[1,1,1]*wmu[1]*wPphi[1]*wH[1]
      #end for iorb
      sum_tau=sum(tau_opt)
      avg_tau=float(sum_tau)/float(size)
      accuml_tau=0
      iorb1_list=np.zeros((size,),dtype=int)
      iorb2_list=np.zeros((size,),dtype=int)
      iorb1_list[:]=1
      iorb2_list[:]=norb
      iorb=0
      for ipe in range(size-1):
        iorb=iorb+1
        accuml_tau=accuml_tau+tau_opt[iorb-1]
        while ((accuml_tau<avg_tau)and(norb-iorb>size-ipe)):
          iorb=iorb+1
          accuml_tau=accuml_tau+tau_opt[iorb-1]
   
        accuml_tau=accuml_tau-avg_tau
        iorb2_list[ipe]=iorb
        iorb1_list[ipe+1]=iorb+1
    else:#for rank!=0
      iorb1_list,iorb2_list=[None]*2
    iorb1_list,iorb2_list=comm.bcast((iorb1_list,iorb2_list),root=0)
    iorb1=iorb1_list[rank]-1#start from 0
    iorb2=iorb2_list[rank]-1
    mynorb=iorb2-iorb1+1
    norb_list=iorb2_list-iorb1_list+1
  else:#parition orbit based on orbit indices, will cause large load imbalance
    iorb1,iorb2,norb_list=simple_partition(comm,nmu*nPphi*nH,size)
    iorb1=iorb1[rank]
    iorb2=iorb2[rank] 
  return iorb1,iorb2,norb_list

def simple_partition(comm,nsteps,nloops):
  if comm.Get_rank()==0:
    nsteps_avg=int(nsteps/nloops)
    nsteps_last=nsteps-nsteps_avg*(nloops-1)
    nsteps_list=np.zeros((nloops,),dtype=int)
    nsteps_list[:]=nsteps_avg
    if nsteps_last>nsteps_avg:
      for iloop in range(nsteps_last-nsteps_avg): nsteps_list[iloop]=nsteps_list[iloop]+1
    istep1=np.zeros((nloops,),dtype=int)
    istep2=np.zeros((nloops,),dtype=int)
    for iloop in range(nloops): istep1[iloop]=int(sum(nsteps_list[0:iloop]))
    istep2=istep1+nsteps_list-1

  else:
    istep1,istep2,nsteps_list=[None]*3
  istep1=comm.bcast(istep1,root=0)
  istep2=comm.bcast(istep2,root=0)
  nsteps_list=comm.bcast(nsteps_list,root=0)
  return istep1,istep2,nsteps_list
