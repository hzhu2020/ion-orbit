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

  return fout

def init(pot0fac,dpotfac,Nr,Nz,comm,rank):
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
  R,Z=np.meshgrid(rlin,zlin)
  psi2d=griddata(rz,psi_rz,(R,Z),method=interp_method)
  Ra=(min(rmesh)+max(rmesh))/2 #major radius
  #read B field
  global Bmag,Br,Bz,Bphi,br,bz,bphi
  if rank==0:
    Br,Bz,Bphi=setup.Bfield(rz,rlin,zlin)
  else:
    Br,Bz,Bphi=[None]*3
  Br,Bz,Bphi=comm.bcast((Br,Bz,Bphi),root=0)
  Bmag=np.sqrt(Br**2+Bz**2+Bphi**2)
  br=Br/Bmag
  bz=Bz/Bmag
  bphi=Bphi/Bmag
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
  global gradBr,gradBz,gradBphi,curlBr,curlBz,curlBphi,curlbr,curlbz,curlbphi,pot0,dpot
  gradBr,gradBz,gradBphi=setup.Grad(rlin,zlin,Bmag,Nr,Nz)
  curlBr,curlBz,curlBphi=setup.Curl(rlin,zlin,Br,Bz,Bphi,Nr,Nz)
  curlbr,curlbz,curlbphi=setup.Curl(rlin,zlin,br,bz,bphi,Nr,Nz)
  if rank==0:
    pot0,dpot=setup.Pot(rz,rlin,zlin)
    pot0=pot0fac*pot0
    dpot=dpotfac*dpot
  else:
    pot0,dpot=[None]*2
  pot0,dpot=comm.bcast((pot0,dpot),root=0)

  global Er00,Ez00,Ephi00
  Er00,Ez00,Ephi00=setup.Grad(rlin,zlin,pot0,Nr,Nz)
  Er00=-Er00
  Ez00=-Ez00
  Ephi00=-Ephi00
  
  global Er0m,Ez0m,Ephi0m
  Er0m,Ez0m,Ephi0m=setup.Grad(rlin,zlin,dpot,Nr,Nz)
  Er0m=-Er0m
  Ez0m=-Ez0m
  Ephi0m=-Ephi0m

def gyropot(comm,mu_arr,qi,mi,ngyro,summation):
  global gyropot0,gyrodpot
  Nz,Nr=np.shape(Er00)
  Nmu=np.size(mu_arr)
  gyropot0=np.zeros((Nz,Nr,Nmu),dtype=float)
  gyrodpot=np.zeros((Nz,Nr,Nmu),dtype=float)
  rank=comm.Get_rank()
  size=comm.Get_size()
  #parition in (r,z,mu) just like the orbit partitioning
  if rank==0:
    ntasks=Nr*Nz*Nmu
    ntasks_avg=int(ntasks/size)
    ntasks_last=ntasks-ntasks_avg*(size-1)
    ntasks_list=np.zeros((size,1),dtype=int)
    ntasks_list[:]=ntasks_avg
    if ntasks_last>ntasks_avg:
      for irank in range(ntasks_last-ntasks_avg): ntasks_list[irank]=ntasks_list[irank]+1
  else:
    ntasks_list=None

  ntasks_list=comm.bcast(ntasks_list,root=0)
  myntasks=int(ntasks_list[rank])
  itask1=int(sum(ntasks_list[0:rank]))
  itask2=itask1+myntasks-1
  for itask in range(itask1,itask2+1):
    iz=int(itask/(Nr*Nmu))
    ir=int((itask-iz*Nr*Nmu)/Nmu)
    imu=itask-iz*Nr*Nmu-ir*Nmu
    mu=mu_arr[imu]
    B=Bmag[iz,ir]
    r=rlin[ir]
    z=zlin[iz]
    if np.isnan(B):
      gyropot0[iz,ir,imu]=np.nan
      gyrodpot[iz,ir,imu]=np.nan
      continue
    rho=np.sqrt(2*mi*mu/B/qi**2)
    gyropot0[iz,ir,imu]=0.0
    gyrodpot[iz,ir,imu]=0.0
    for igyro in range(ngyro):
      angle=2*np.pi*float(igyro)/float(ngyro)
      r1=r+rho*np.cos(angle)
      z1=z+rho*np.sin(angle)
      tmp=varTwoD(R,Z,pot0,r1,z1)
      if np.isnan(tmp):
        gyropot0[iz,ir,imu]=np.nan
      else:
        gyropot0[iz,ir,imu]=gyropot0[iz,ir,imu]+tmp/float(ngyro)
      tmp=varTwoD(R,Z,dpot,r1,z1)
      if np.isnan(tmp):
        gyrodpot[iz,ir,imu]=np.nan
      else:
        gyrodpot[iz,ir,imu]=gyrodpot[iz,ir,imu]+tmp/float(ngyro)
  #end for itask
  gyropot0=comm.allreduce(gyropot0,op=summation)
  gyrodpot=comm.allreduce(gyrodpot,op=summation)
  return

def efield(iorb):
  myEr00=Er00
  myEz00=Ez00
  Nr,Nz=np.shape(Er00)
  if gyro_E:
    from parameters import nPphi,nH
    imu=int(iorb/(nPphi*nH))
    #To avoid the dot product between b and the ungyroaveraged E00
    myEr0m,myEz0m,myEphi0m=setup.Grad(rlin,zlin,gyropot0[:,:,imu]+gyrodpot[:,:,imu],Nr,Nz)
    myEr0m=-myEr0m-Er00
    myEz0m=-myEz0m-Ez00
  else:
    myEr0m=Er0m
    myEz0m=Ez0m
  
  return myEr00,myEz00,myEr0m,myEz0m

def H_arr(comm,qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr,summation):
  global dH
  nsurf=np.size(rsurf)#number of mesh points along the surface
  Bsurf=np.zeros((nsurf,),dtype=float)#magnitude of B 
  dpotsurf=np.zeros((nsurf,),dtype=float)#m/=0 potential along the surface
  covbphisurf=np.zeros((nsurf,),dtype=float)#covariant component \vec{b}\cdot(dX/d\phi)
  for i in range(nsurf):
    r,z=rsurf[i],zsurf[i]
    Bsurf[i]=varTwoD(R,Z,Bmag,r,z)
    if not(gyro_E):
      dpotsurf[i]=varTwoD(R,Z,dpot,r,z)
    covbphisurf[i]=r*varTwoD(R,Z,Bphi,r,z)/Bsurf[i]
  
  if not(gyro_E):
    #smooth dpotsurf
    tmp=np.fft.fft(dpotsurf)
    freq=nsurf*np.fft.fftfreq(nsurf)
    tmp[abs(freq)>dpot_fourier_maxm]=0
    dpotsurf=np.real(np.fft.ifft(tmp))
 
  #partition tasks
  rank=comm.Get_rank()
  size=comm.Get_size()
  if rank==0:
    ntasks=nmu*nPphi
    ntasks_avg=int(ntasks/size)
    ntasks_last=ntasks-ntasks_avg*(size-1)
    ntasks_list=np.zeros((size,1),dtype=int)
    ntasks_list[:]=ntasks_avg
    if ntasks_last>ntasks_avg:
      for irank in range(ntasks_last-ntasks_avg): ntasks_list[irank]=ntasks_list[irank]+1
  else:
    ntasks_list=None
  ntasks_list=comm.bcast(ntasks_list,root=0)
  myntasks=int(ntasks_list[rank])
  itask1=int(sum(ntasks_list[0:rank]))
  itask2=itask1+myntasks-1

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
      dpotsurf[:]=0.0
      for i in range(nsurf):
        r,z=rsurf[i],zsurf[i]
        dpotsurf[i]=varTwoD(R,Z,gyrodpot[:,:,imu]+gyropot0[:,:,imu]-pot0,r,z)
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
      thetam=theta[0]
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
          r_end[imu,iPphi,iH]=-1e9
          z_end[imu,iPphi,iH]=-1e9
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

def H2d(mu,Pphi,mi,qi):
  vp2d=(Pphi-qi*psi2d)/mi/R/Bphi*Bmag
  H=mu*Bmag+0.5*mi*vp2d**2+qi*(pot0+dpot)
  gradHr,gradHz,gradHphi=setup.Grad(rlin,zlin,H,rlin.size,zlin.size)
  return H,gradHr,gradHz 
