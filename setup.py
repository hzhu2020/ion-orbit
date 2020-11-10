import adios as ad
import numpy as np
from scipy.interpolate import griddata
import math
from parameters import input_dir,LCFS_psitol,LCFS_rztol

def Grid(Nr,Nz):
  fname=input_dir+'/xgc.mesh.bp'
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
  #find the LCFS
  RLCFS=np.array([],dtype=float)
  ZLCFS=np.array([],dtype=float)
  for i in range(np.shape(rz)[0]):
    if (abs(psi_rz[i]-psix)<LCFS_psitol) and (rz[i,1]>zx-LCFS_rztol):
      RLCFS=np.append(RLCFS,rz[i,0])
      ZLCFS=np.append(ZLCFS,rz[i,1])
  size=np.size(RLCFS)
  theta=np.zeros((size,),dtype=float)
  for i in range(size):
    theta[i]=math.atan2(ZLCFS[i]-za,RLCFS[i]-ra)
    if (abs(ZLCFS[i]-zx)<LCFS_rztol)and(abs(RLCFS[i]-rx)<LCFS_rztol): thetax=theta[i]
  for i in range(size):
    if theta[i]<=thetax: theta[i]=theta[i]+2*np.pi
  idx=np.argsort(theta)
  RLCFS=RLCFS[idx]
  ZLCFS=ZLCFS[idx]
  return rz,psi_rz,rx,zx,psix,Ba,RLCFS,ZLCFS
  
def Tempix(step):
  #get ion equilibrium temperature at the LCFS
  fname=input_dir+'/xgc.oneddiag.bp'
  f=ad.file(fname)
  psi=f['psi'].read() #psi is normalized
  Tperp=f['i_perp_temperature_df_1d'].read()
  f.close()
  psi=np.array(psi[0,:])
  Tperp=np.array(Tperp[step,:])
  Ta=np.nan
  for i in range(np.size(psi)):
    if abs(psi[i]-1.0)<1e-3: Ta=Tperp[i]
  return Ta

def Grad(r,z,fld,Nr,Nz):
  dr=r[2]-r[1]
  dz=z[2]-z[1]
  gradr=np.nan*np.zeros((Nz,Nr),dtype=float)
  gradz=np.nan*np.zeros((Nz,Nr),dtype=float)
  gradphi=np.nan*np.zeros((Nz,Nr),dtype=float)
  for i in range(1,Nz-1):
    for j in range(1,Nr-1):
      gradr[i,j]=(fld[i,j+1]-fld[i,j-1])/2/dr
      gradz[i,j]=(fld[i+1,j]-fld[i-1,j])/2/dz
      gradphi[i,j]=0.0 #assuming axisymmetry
  return gradr,gradz,gradphi

def Curl(r,z,fldr,fldz,fldphi,Nr,Nz):
  #all calculations below assume axisymmetry
  dr=r[2]-r[1]
  dz=z[2]-z[1]
  curlr=np.nan*np.zeros((Nz,Nr),dtype=float)
  curlz=np.nan*np.zeros((Nz,Nr),dtype=float)
  curlphi=np.nan*np.zeros((Nz,Nr),dtype=float)
  for i in range(1,Nz-1):
    for j in range(1,Nr-1):
      curlr[i,j]=-(fldphi[i+1,j]-fldphi[i-1,j])/2/dz
      curlphi[i,j]=(fldr[i+1,j]-fldr[i-1,j])/2/dz-(fldz[i,j+1]-fldz[i,j-1])/2/dr
      curlz[i,j]=(r[j+1]*fldphi[i,j+1]-r[j-1]*fldphi[i,j-1])/2/dr/r[j]
  return curlr,curlz,curlphi

def Bfield(rz,rlin,zlin):
  fname=input_dir+'/xgc.bfield.bp'
  f=ad.file(fname)
  B=f['/node_data[0]/values'].read()
  f.close()
  R,Z=np.meshgrid(rlin,zlin)
  Br=griddata(rz,B[:,0],(R,Z),method='linear')
  Bz=griddata(rz,B[:,1],(R,Z),method='linear')
  Bphi=griddata(rz,B[:,2],(R,Z),method='linear')
  return Br,Bz,Bphi

def Pot00(step):
  fname=input_dir+'/xgc.oneddiag.bp'
  f=ad.file(fname)
  psi00=f['psi00'].read()
  pot00=f['pot00_1d'].read()
  f.close()
  psi00=np.array(psi00[0,:])
  pot00_1d=np.array(pot00[step,:])#choose the last step
  return psi00,pot00_1d
