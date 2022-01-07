from parameters import xgc,adios_version,input_dir,pot_file,\
                       surf_psin,surf_psitol,surf_rztol,interp_method
if adios_version==1:
  import adios as ad
elif adios_version==2:
  import adios2 as ad
else:
  print('Wrong adios version.',flush=True)
  exit()

import numpy as np
from scipy.interpolate import griddata
import math

def Grid(Nr,Nz):
  fname=input_dir+'/xgc.mesh.bp'
  if adios_version==1:
    f=ad.file(fname)
    rz=f['rz'].read()
    psi_rz=f['psi'].read()
    f.close()
    fname=input_dir+'/xgc.equil.bp'
    f=ad.file(fname)
    rx=f['eq_x_r'].read()
    zx=f['eq_x_z'].read()
    psix=f['eq_x_psi'].read()
    Ba=f['eq_axis_b'].read()
    ra=f['eq_axis_r'].read()
    za=f['eq_axis_z'].read()
    f.close()
  elif adios_version==2:
    f=ad.open(fname,'r')
    rz=f.read('/coordinates/values')
    psi_rz=f.read('psi')
    f.close()
    fname=input_dir+'/xgc.equil.bp'
    f=ad.open(fname,'r')
    rx=f.read('eq_x_r')
    zx=f.read('eq_x_z')
    psix=f.read('eq_x_psi')
    Ba=f.read('eq_axis_b')
    ra=f.read('eq_axis_r')
    za=f.read('eq_axis_z')
    f.close()
  #find the surface closest to given surf_psin
  global surf_psin #need to claim it global since its value is changed below
  dpsi=1e10
  #TODO caution: it's possible that the following will choose non-aligned nodes in the core
  for i in range(np.shape(rz)[0]):
    if (psi_rz[i]<psix+surf_psitol)\
       and (rz[i,1]>zx-surf_rztol)\
       and (abs(psi_rz[i]-psix*surf_psin)<dpsi):
         dpsi=abs(psi_rz[i]-psix*surf_psin)
         minloc=i
  surf_psin=psi_rz[minloc]/psix
  Rsurf=np.array([],dtype=float)
  Zsurf=np.array([],dtype=float)
  for i in range(np.shape(rz)[0]):
    if (abs(psi_rz[i]-psix*surf_psin)<surf_psitol) and (rz[i,1]>zx-surf_rztol):
      Rsurf=np.append(Rsurf,rz[i,0])
      Zsurf=np.append(Zsurf,rz[i,1])
  size=np.size(Rsurf)
  theta=np.zeros((size,),dtype=float)
  dist=np.zeros((size,),dtype=float)
  for i in range(size):
    dist[i]=np.sqrt((Zsurf[i]-za)**2+(Rsurf[i]-ra)**2)
    theta[i]=math.atan2(Zsurf[i]-za,Rsurf[i]-ra)

  return rz,psi_rz,rx,zx,psix*surf_psin,psix,Ba,Rsurf,Zsurf,theta,dist
  

def Grad(r,z,fld,Nr,Nz):
  dr=r[2]-r[1]
  dz=z[2]-z[1]
  gradr=np.nan*np.zeros((Nz,Nr),dtype=float)
  gradz=np.nan*np.zeros((Nz,Nr),dtype=float)
  gradphi=np.zeros((Nz,Nr),dtype=float)#assuming axisymmetry
  for i in range(1,Nz-1):
    gradz[i,:]=(fld[i+1,:]-fld[i-1,:])/2/dz
  for j in range(1,Nr-1):
    gradr[:,j]=(fld[:,j+1]-fld[:,j-1])/2/dr

  return gradr,gradz,gradphi

def Curl(r,z,fldr,fldz,fldphi,Nr,Nz):
  #all calculations below assume axisymmetry
  dr=r[2]-r[1]
  dz=z[2]-z[1]
  curlr=np.nan*np.zeros((Nz,Nr),dtype=float)
  curlz=np.nan*np.zeros((Nz,Nr),dtype=float)
  curlphi=np.nan*np.zeros((Nz,Nr),dtype=float)
  for i in range(1,Nz-1):
    curlr[i,:]=-(fldphi[i+1,:]-fldphi[i-1,:])/2/dz
    curlphi[i,:]=(fldr[i+1,:]-fldr[i-1,:])/2/dz
  for j in range(1,Nr-1):
    curlz[:,j]=(r[j+1]*fldphi[:,j+1]-r[j-1]*fldphi[:,j-1])/2/dr/r[j]
    curlphi[:,j]=curlphi[:,j]-(fldz[:,j+1]-fldz[:,j-1])/2/dr
  return curlr,curlz,curlphi

def Bfield(rz,rlin,zlin,itask1,itask2):
  if adios_version==1:
    fname=input_dir+'/xgc.bfield.bp'
    f=ad.file(fname)
    B=f['/node_data[0]/values'].read()
    f.close()
  elif adios_version==2:
    fname=input_dir+'/xgc.bfield.bp'
    f=ad.open(fname,'r')
    if xgc=='xgca':
      B=f.read('bfield')
    elif xgc=='xgc1':
      B=f.read('/node_data[0]/values')
    else:
      print('Wrong parameter xgc.')
    f.close()

  R,Z=np.meshgrid(rlin,zlin)
  if(itask1<=1)and(1<=itask2):
    Br=griddata(rz,B[:,0],(R,Z),method=interp_method)
  else:
    Br=np.zeros(np.shape(R),dtype=float)
  if(itask1<=2)and(2<=itask2):
    Bz=griddata(rz,B[:,1],(R,Z),method=interp_method)
  else:
    Bz=np.zeros(np.shape(R),dtype=float)
  if(itask1<=3)and(3<=itask2):
    Bphi=griddata(rz,B[:,2],(R,Z),method=interp_method)
  else:
    Bphi=np.zeros(np.shape(R),dtype=float)
  return Br,Bz,Bphi

def Pot_init(rank):
  if adios_version!=2:
    if rank==0: print('Reading potential: ADIOS2 is required!',flush=True)
    exit()
  global guess_min,inv_guess_d,guess_table,guess_xtable,guess_count,guess_list,mapping,nd,psi_rz
  fname=input_dir+'/xgc.mesh.bp'
  f=ad.open(fname,'r')
  guess_min=f.read('guess_min')
  inv_guess_d=f.read('inv_guess_d')
  guess_table=f.read('guess_table')
  guess_xtable=f.read('guess_xtable')
  guess_count=f.read('guess_count')
  guess_list=f.read('guess_list')
  mapping=f.read('mapping')
  nd=f.read('nd')
  psi_rz=f.read('psi')
  f.close()
  guess_table=np.transpose(guess_table)
  mapping=np.transpose(mapping)
  guess_xtable=np.transpose(guess_xtable)
  guess_count=np.transpose(guess_count)
  nd=np.transpose(nd)
  global pot0,dpot
  fname=input_dir+'/'+pot_file
  f=ad.open(fname,'r')
  pot0=f.read('pot0')
  dpot=f.read('dpot')
  f.close()
  if xgc=='xgc1':
    if rank==0: print('xgc=xgc1, apply toroidal average to dpot.',flush=True)
    dpot=np.mean(dpot,axis=0)
  return

def Pot_cpu(rlin,zlin,pot0fac,dpotfac,psi2d,psix,itask1,itask2):
  Nr=np.size(rlin)
  Nz=np.size(zlin)
  pot02d=np.zeros((Nz,Nr),dtype=float)
  dpot2d=np.zeros((Nz,Nr),dtype=float)
  for itask in range(itask1,itask2+1):
    iz=int(itask/Nr)
    ir=int(itask-iz*Nr)
    itr,p=search_tr2([rlin[ir],zlin[iz]])
    if (itr>0):
      if(max(p)<1.0): p=t_coeff_mod([rlin[ir],zlin[iz]],itr,p,psi2d[iz,ir],psix)
      for i in range(3):
        node=nd[i,itr-1]
        dpot2d[iz,ir]=dpot2d[iz,ir]+p[i]*dpot[node-1]
        pot02d[iz,ir]=pot02d[iz,ir]+p[i]*pot0[node-1]
    else:
      pot02d[iz,ir]=np.nan
      dpot2d[iz,ir]=np.nan

  return pot0fac*pot02d,dpotfac*dpot2d

def search_tr2(xy):
   itr=-1
   p=np.zeros((3,),dtype=float)
   eps=1e-10

   ilo,jlo=1,1
   ihi,jhi=np.shape(guess_table)
   ij=np.zeros((2,),dtype=int)
   ij[0]=math.floor((xy[0]-guess_min[0])*inv_guess_d[0])+1
   ij[1]=math.floor((xy[1]-guess_min[1])*inv_guess_d[1])+1
   i=max(ilo,min(ihi,ij[0]))
   j=max(jlo,min(jhi,ij[1]))

   istart=guess_xtable[i-1,j-1]
   iend=istart+guess_count[i-1,j-1]-1
   for k in range(istart,iend+1):
     itrig=guess_list[k-1]
     dx=xy-mapping[:,2,itrig-1]
     p[0:2]=mapping[0:2,0,itrig-1]*dx[0]+mapping[0:2,1,itrig-1]*dx[1]
     p[2]=1.0-p[0]-p[1]
     if min(p)>=-eps:
       itr=itrig
       break

   return itr,p

def t_coeff_mod(xy,itr,p,psi_in,psix):
  eps1=1e-4*psix
  psi=np.zeros((4,),dtype=float)
  psi[3]=psi_in
  for i in range(3): psi[i]=psi_rz[nd[i,itr-1]-1]
  psi_diff=0
  if abs(psi[0]-psi[1])<=eps1:
    psi_diff=3
  elif abs(psi[0]-psi[2])<=eps1:
    psi_diff=2
  elif abs(psi[1]-psi[2])<=eps1:
    psi_diff=1

  if (psi_diff>0):
    a=psi_diff%3+1
    b=(psi_diff+1)%3+1
    psi[:]=abs(psi[:]-psi[psi_diff-1])
    p_temp=psi[3]/psi[a-1]
    p_temp=min(p_temp,1.0)
    p[psi_diff-1]=1.0-p_temp
    t_temp=p[a-1]+p[b-1]
    p[a-1]=p_temp*p[a-1]/t_temp
    p[b-1]=p_temp*p[b-1]/t_temp
  return p

def Pot_gpu(rlin,zlin,pot0fac,dpotfac,psi2d,psix,itask1,itask2):
  import cupy as cp
  interp_pot_kernel = cp.RawKernel(r'''
  extern "C" __device__    
  void search_tr2(double r,double z,int* itr,double* p,int ihi,int jhi,int num_tri,double* guess_min,\
    double* inv_guess_d,int* guess_xtable,int* guess_list,int* guess_count,double* mapping)
  {
    int ilo,jlo,ij[2],ix,iy,istart,iend,itrig;
    double eps=1E-10,dx,dy,tmp;    
    itr[0]=-1;
    p[0]=0.;
    p[1]=0.;
    p[2]=0.;
    ilo=1;
    jlo=1;
    ij[0]=floor((r-guess_min[0])*inv_guess_d[0])+1;
    ij[1]=floor((z-guess_min[1])*inv_guess_d[1])+1;
    ix=min(int(ihi),ij[0]);
    ix=max(ilo,ix);
    iy=min(int(jhi),ij[1]);
    iy=max(jlo,iy);
    istart=guess_xtable[(ix-1)*jhi+iy-1];
    iend=istart+guess_count[(ix-1)*jhi+iy-1]-1;
    for (int k=istart;k<=iend;k++){
      itrig=guess_list[k-1];
      dx=r-mapping[0*3*num_tri+2*num_tri+itrig-1];
      dy=z-mapping[1*3*num_tri+2*num_tri+itrig-1];   
      p[0]=mapping[0*3*num_tri+0*num_tri+itrig-1]*dx\
          +mapping[0*3*num_tri+1*num_tri+itrig-1]*dy;
      p[1]=mapping[1*3*num_tri+0*num_tri+itrig-1]*dx\
          +mapping[1*3*num_tri+1*num_tri+itrig-1]*dy;
      p[2]=1.-p[0]-p[1];
      tmp=min(p[0],p[1]);
      tmp=min(tmp,p[2]);
      if (tmp>=-eps){
        itr[0]=itrig;
        break;
      }
    }
  }
  extern "C" __device__ 
  void t_coeff_mod(double r,double z,double* rlin,double* zlin,double* p,double psix,\
    double psi_in,double* psi_rz,int itr,int num_tri,int* nd,int Nr,int Nz)
  {
    int psi_diff,a,b;
    double psi[4],tmp,eps1,p_temp,t_temp;
    eps1=1E-4*psix;
    psi[0]=psi_rz[nd[0*num_tri+itr-1]-1];
    psi[1]=psi_rz[nd[1*num_tri+itr-1]-1];
    psi[2]=psi_rz[nd[2*num_tri+itr-1]-1];
    psi[3]=psi_in;
    psi_diff=0;
    if (abs(psi[0]-psi[1])<=eps1){
      psi_diff=3;
    }
    else if (abs(psi[0]-psi[2])<=eps1){
      psi_diff=2;
    }
    else if (abs(psi[1]-psi[2])<=eps1){
      psi_diff=1;
    }
    if (psi_diff>0)
    {
      a=psi_diff%3+1;
      b=(psi_diff+1)%3+1;
      tmp=psi[psi_diff-1];
      psi[0]=abs(psi[0]-tmp);
      psi[1]=abs(psi[1]-tmp);
      psi[2]=abs(psi[2]-tmp);
      psi[3]=abs(psi[3]-tmp);
      p_temp=psi[3]/psi[a-1];
      p_temp=min(p_temp,1.0);
      p[psi_diff-1]=1.0-p_temp;
      t_temp=p[a-1]+p[b-1];
      p[a-1]=p_temp*p[a-1]/t_temp;
      p[b-1]=p_temp*p[b-1]/t_temp;
    }
  }   
  extern "C" __global__
  void interp_pot(int ihi,int jhi,int num_tri,double* guess_min,double* inv_guess_d,int* guess_xtable,\
      int* guess_list,int* guess_count,double* mapping,bool tri_psi,double* rlin,double* zlin,double psix,\
      double* psi2d,double* psi_rz,int* nd,int Nr,int Nz,int itask1,int itask2,int nblocks_max,\
      double* pot0,double* dpot,double* pot02d,double* dpot2d) 
  {
      double r,z,p[3],tmp,psi_in; 
      int itask,iz,ir,node,itr[1];
      itask=blockIdx.x+itask1;
      while (itask<=itask2)
      {
        iz=itask/Nr;
        ir=itask-iz*Nr;
        r=rlin[ir];
        z=zlin[iz];
        search_tr2(r,z,itr,p,ihi,jhi,num_tri,guess_min,inv_guess_d,guess_xtable,guess_list,guess_count,mapping);
        tmp=max(p[0],p[1]);
        tmp=max(tmp,p[2]);
        if ((tri_psi)&&(itr[0]>0)&&(tmp<1.0))
        {
          psi_in=psi2d[itask];
          t_coeff_mod(r,z,rlin,zlin,p,psix,psi_in,psi_rz,itr[0],num_tri,nd,Nr,Nz);
        }
        if (itr[0]>0){
          for(int k=0;k<3;k++){
            node=nd[k*num_tri+itr[0]-1];
            pot02d[itask]=pot02d[itask]+p[k]*pot0[node-1];
            dpot2d[itask]=dpot2d[itask]+p[k]*dpot[node-1];
          }
        }else{
          pot02d[itask]=nan("");
          dpot2d[itask]=nan("");
        }
        itask=itask+nblocks_max;
      }

  }
  ''', 'interp_pot')
  Nr=np.size(rlin)
  Nz=np.size(zlin)
  pot0_gpu=cp.array(pot0,dtype=cp.float64)
  dpot_gpu=cp.array(dpot,dtype=cp.float64)
  pot02d_gpu=cp.zeros((Nz*Nr,),dtype=cp.float64)
  dpot2d_gpu=cp.zeros((Nz*Nr,),dtype=cp.float64)
  nblocks_max=4096
  ntasks=itask2-itask1+1
  nblocks=min(nblocks_max,ntasks)
  num_tri=np.shape(mapping)[2]
  ihi,jhi=np.shape(guess_table)
  guess_min_gpu=cp.array(guess_min,dtype=cp.float64)
  inv_guess_d_gpu=cp.array(inv_guess_d,dtype=cp.float64)
  guess_xtable_gpu=cp.array(guess_xtable,dtype=cp.int32).ravel(order='C')
  guess_list_gpu=cp.array(guess_list,dtype=cp.int32)
  guess_count_gpu=cp.array(guess_count,dtype=cp.int32).ravel(order='C')
  mapping_gpu=cp.array(mapping,dtype=cp.float64).ravel(order='C')
  nd_gpu=cp.array(nd,dtype=cp.int32).ravel(order='C')
  rlin_gpu=cp.array(rlin,dtype=cp.float64)
  zlin_gpu=cp.array(zlin,dtype=cp.float64)
  psi2d_gpu=cp.array(psi2d,dtype=cp.float64).ravel(order='C')
  psi_rz_gpu=cp.array(psi_rz,dtype=cp.float64)
  tri_psi=True
  interp_pot_kernel((nblocks,),(1,),(int(ihi),int(jhi),int(num_tri),guess_min_gpu,inv_guess_d_gpu,\
    guess_xtable_gpu,guess_list_gpu,guess_count_gpu,mapping_gpu,tri_psi,rlin_gpu,zlin_gpu,float(psix),\
    psi2d_gpu,psi_rz_gpu,nd_gpu,int(rlin.size),int(zlin.size),int(itask1),int(itask2),int(nblocks_max),\
    pot0_gpu,dpot_gpu,pot02d_gpu,dpot2d_gpu))
  pot02d=cp.asnumpy(pot02d_gpu).reshape((Nz,Nr),order='C')
  dpot2d=cp.asnumpy(dpot2d_gpu).reshape((Nz,Nr),order='C')
  del guess_min_gpu,inv_guess_d_gpu,guess_xtable_gpu,guess_list_gpu,guess_count_gpu,mapping_gpu,\
      nd_gpu,rlin_gpu,zlin_gpu,psi2d_gpu,psi_rz_gpu,pot0_gpu,dpot_gpu,pot02d_gpu,dpot2d_gpu
  mempool = cp.get_default_memory_pool()
  pinned_mempool = cp.get_default_pinned_memory_pool()
  mempool.free_all_blocks()
  pinned_mempool.free_all_blocks()
  return pot0fac*pot02d,dpotfac*dpot2d
