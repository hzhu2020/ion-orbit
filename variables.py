import numpy as np
from parameters import interp_method
from scipy.interpolate import griddata,interp1d
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

def init(potfac,Nr,Nz,pot00_step,comm,rank):
  #read mesh
  global rlin,zlin,R,Z,psi2d,psix,Ra,Ba,rLCFS,zLCFS,theta,dist
  if rank==0:
    rz,psi_rz,rx,zx,psix,Ba,rLCFS,zLCFS,theta,dist=setup.Grid(Nr,Nz)
  else:
    rz,psi_rz,rx,zx,psix,Ba,rLCFS,zLCFS,theta,dist=[None]*10
  rz,psi_rz,rx,zx,psix,Ba,rLCFS,zLCFS,theta,dist\
            =comm.bcast((rz,psi_rz,rx,zx,psix,Ba,rLCFS,zLCFS,theta,dist),root=0)
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
  #calculate grad B, curl B, and curl b
  global gradBr,gradBz,gradBphi,curlBr,curlBz,curlBphi,curlbr,curlbz,curlbphi,pot00_2d
  gradBr,gradBz,gradBphi=setup.Grad(rlin,zlin,Bmag,Nr,Nz)
  curlBr,curlBz,curlBphi=setup.Curl(rlin,zlin,Br,Bz,Bphi,Nr,Nz)
  curlbr,curlbz,curlbphi=setup.Curl(rlin,zlin,br,bz,bphi,Nr,Nz)
  #read 1d pot00 and interpolate it to 2D
  if rank==0:
    psi00,pot00_1d=setup.Pot00(pot00_step)
  else:
    psi00,pot00_1d=None,None
  psi00,pot00_1d=comm.bcast((psi00,pot00_1d),root=0)
  pot_interp=interp1d(psi00,pot00_1d,kind=interp_method)
  pot00_2d=np.nan*np.zeros((Nz,Nr),dtype=float)
  for i in range(1,Nz-1):
    for j in range(1,Nr-1):
      psi=varTwoD(R,Z,psi2d,rlin[j],zlin[i])
      if not(np.isnan(psi)) and (psi<max(psi00)) and (psi>min(psi00)):
        pot00_2d[i,j]=potfac*pot_interp(psi)
      else:
        pot00_2d[i,j]=np.nan
 
  #calculate E=-grad phi TODO: including gyroaverage of E
  global Er,Ez,Ephi
  Er,Ez,Ephi=setup.Grad(rlin,zlin,pot00_2d,Nr,Nz)
  Er=-Er
  Ez=-Ez
  Ephi=-Ephi

def H_arr(qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr):
  global Hmin,Hmax,dH
  nLCFS=np.size(rLCFS)#number of mesh points along the LCFS
  BLCFS=np.zeros((nLCFS,),dtype=float)#magnitude of B 
  covbphiLCFS=np.zeros((nLCFS,),dtype=float)#covariant component \vec{b}\cdot(dX/d\phi)
  for i in range(nLCFS):
    r,z=rLCFS[i],zLCFS[i]
    BLCFS[i]=varTwoD(R,Z,Bmag,r,z)
    covbphiLCFS[i]=r*varTwoD(R,Z,Bphi,r,z)/BLCFS[i]
  Hmin=np.zeros((nmu,nPphi),dtype=float)
  Hmax=np.zeros((nmu,nPphi),dtype=float)
  dH=np.zeros((nmu,nPphi,nH),dtype=float)
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
        dH[imu,iPphi,iH]=0.5*(HLCFS[iLCFS+diH]-HLCFS[iLCFS-diH])
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
          if endloc==nLCFS-1:
            nextloc=0
          else:
            nextloc=endloc+1
          wH=(HLCFS[iLCFS]-HLCFS[endloc])/(HLCFS[nextloc]-HLCFS[endloc])
          r_end[imu,iPphi,iH]=(1-wH)*rLCFS[endloc]+wH*rLCFS[nextloc]
          z_end[imu,iPphi,iH]=(1-wH)*zLCFS[endloc]+wH*zLCFS[nextloc]
  return r_beg,z_beg,r_end,z_end

def H2d(mu,Pphi,mi,qi):
  vp2d=(Pphi-qi*psi2d)/mi/R/Bphi*Bmag
  H=mu*Bmag+0.5*mi*vp2d**2+qi*pot00_2d
  gradHr,gradHz,gradHphi=setup.Grad(rlin,zlin,H,rlin.size,zlin.size)
  return H,gradHr,gradHz 
