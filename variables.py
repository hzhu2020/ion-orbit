import numpy as np
from scipy.interpolate import griddata
import math
import setup
import myinterp

def init(potfac,Nr,Nz,temp_step,pot00_step):
  #read mesh
  global R,Z,psi2d,psix,Ra,Ba,rLCFS,zLCFS
  rz,psi_rz,rx,zx,psix,Ba,rLCFS,zLCFS=setup.Grid(Nr,Nz)
  rmesh=rz[:,0]
  zmesh=rz[:,1]
  rlin=np.linspace(min(rmesh),max(rmesh),Nr)
  zlin=np.linspace(min(zmesh),max(zmesh),Nz)
  R,Z=np.meshgrid(rlin,zlin)
  psi2d=griddata(rz,psi_rz,(R,Z),method='linear')
  Ra=(min(rmesh)+max(rmesh))/2 #major radius
  #read B field
  global Bmag,Br,Bz,Bphi,br,bz,bphi
  Br,Bz,Bphi=setup.Bfield(rz,rlin,zlin)
  Bmag=np.sqrt(Br**2+Bz**2+Bphi**2)
  br=Br/Bmag
  bz=Bz/Bmag
  bphi=Bphi/Bmag
  #calculate grad B, curl B, and curl b
  global gradBr,gradBz,gradBphi,curlBr,curlBz,curlBphi,curlbr,curlbz,curlbphi
  gradBr,gradBz,gradBphi=setup.Grad(rlin,zlin,Bmag,Nr,Nz)
  curlBr,curlBz,curlBphi=setup.Curl(rlin,zlin,Br,Bz,Bphi,Nr,Nz)
  curlbr,curlbz,curlbphi=setup.Curl(rlin,zlin,br,bz,bphi,Nr,Nz)
  #read 1d pot00 and interpolate it to 2D
  psi00,pot00_1d=setup.Pot00(pot00_step)
  pot00_2d=np.nan*np.zeros((Nz,Nr),dtype=float)
  for i in range(1,Nz-1):
    for j in range(1,Nr-1):
      psi=myinterp.TwoD(R,Z,psi2d,rlin[j],zlin[i])
      if not(np.isnan(psi)):
        if zlin[i]>=zx:
          pot00_2d[i,j]=potfac*myinterp.OneD(psi00,pot00_1d,psi)
        else:
          pot00_2d[i,j]=0
  #calculate E=-grad phi TODO: including gyroaverage of E
  global Er,Ez,Ephi
  Er,Ez,Ephi=setup.Grad(rlin,zlin,pot00_2d,Nr,Nz)
  Er=-Er
  Ez=-Ez
  Ephi=-Ephi

def H_arr(qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr):
  global Hmin,Hmax
  nLCFS=np.size(rLCFS)#number of mesh points along the LCFS
  BLCFS=np.zeros((nLCFS,),dtype=float)#magnitude of B 
  covbphiLCFS=np.zeros((nLCFS,),dtype=float)#covariant component \vec{b}\cdot(dX/d\phi)
  for i in range(nLCFS):
    r,z=rLCFS[i],zLCFS[i]
    BLCFS[i]=myinterp.TwoD(R,Z,Bmag,r,z)
    covbphiLCFS[i]=r*myinterp.TwoD(R,Z,Bphi,r,z)/BLCFS[i]
  Hmin=np.zeros((nmu,nPphi),dtype=float)
  Hmax=np.zeros((nmu,nPphi),dtype=float)
  #crossing locations of the orbits with the LCFS
  r_beg=np.zeros((nmu,nPphi,nH),dtype=float)
  z_beg=np.zeros((nmu,nPphi,nH),dtype=float)
  r_end=np.zeros((nmu,nPphi,nH),dtype=float)
  z_end=np.zeros((nmu,nPphi,nH),dtype=float)
  for imu in range(nmu):
    for iPphi in range(nPphi):
      HLCFS=np.zeros((nLCFS,),dtype=float)
      for iLCFS in range(nLCFS): 
        #The zonal potential does not show up in H. TODO: including m/=0 potential.
        HLCFS[iLCFS]=(Pphi_arr[iPphi]-qi*psix)**2/covbphiLCFS[iLCFS]**2/2/mi+mu_arr[imu]*BLCFS[iLCFS]
      Hmin[imu,iPphi],minloc=min(HLCFS),np.argmin(HLCFS)
      Hmax[imu,iPphi],maxloc=max(HLCFS),np.argmax(HLCFS)
      diH=math.floor((maxloc-minloc)/(nH+2))
      for iH in range(nH):
        iLCFS=minloc+diH*(iH+1)
        r_beg[imu,iPphi,iH]=rLCFS[iLCFS]
        z_beg[imu,iPphi,iH]=zLCFS[iLCFS]
        if HLCFS[iLCFS]<=HLCFS[0]:
          endloc=np.argmin(abs(HLCFS[0:minloc]-HLCFS[iLCFS]))
        else:
          endloc=maxloc+np.argmin(abs(HLCFS[maxloc:]-HLCFS[iLCFS]))
        if HLCFS[iLCFS]>=HLCFS[endloc]:
          wH=(HLCFS[iLCFS]-HLCFS[endloc])/(HLCFS[endloc-1]-HLCFS[endloc])
          r_end[imu,iPphi,iH]=(1-wH)*rLCFS[endloc]+wH*rLCFS[endloc-1]
          z_end[imu,iPphi,iH]=(1-wH)*zLCFS[endloc]+wH*zLCFS[endloc-1]
        else:
          wH=(HLCFS[iLCFS]-HLCFS[endloc])/(HLCFS[endloc+1]-HLCFS[endloc])
          r_end[imu,iPphi,iH]=(1-wH)*rLCFS[endloc]+wH*rLCFS[endloc+1]
          z_end[imu,iPphi,iH]=(1-wH)*zLCFS[endloc]+wH*zLCFS[endloc+1]
  return r_beg,z_beg,r_end,z_end
