import numpy as np
from parameters import interp_method
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
  global rlin,zlin,R,Z,psi2d,psi_surf,Ra,Ba,rsurf,zsurf,theta,dist
  if rank==0:
    rz,psi_rz,rx,zx,psi_surf,Ba,rsurf,zsurf,theta,dist=setup.Grid(Nr,Nz)
  else:
    rz,psi_rz,rx,zx,psi_surf,Ba,rsurf,zsurf,theta,dist=[None]*10
  rz,psi_rz,rx,zx,psi_surf,Ba,rsurf,zsurf,theta,dist\
            =comm.bcast((rz,psi_rz,rx,zx,psi_surf,Ba,rsurf,zsurf,theta,dist),root=0)
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

  #calculate E=-grad phi TODO: including gyroaverage of E
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

def H_arr(qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr):
  global Hmin,Hmax,dH
  nsurf=np.size(rsurf)#number of mesh points along the surface
  Bsurf=np.zeros((nsurf,),dtype=float)#magnitude of B 
  dpotsurf=np.zeros((nsurf,),dtype=float)#m/=0 potential along the surface
  covbphisurf=np.zeros((nsurf,),dtype=float)#covariant component \vec{b}\cdot(dX/d\phi)
  for i in range(nsurf):
    r,z=rsurf[i],zsurf[i]
    Bsurf[i]=varTwoD(R,Z,Bmag,r,z)
    dpotsurf[i]=varTwoD(R,Z,dpot,r,z)
    covbphisurf[i]=r*varTwoD(R,Z,Bphi,r,z)/Bsurf[i]
  
  #smooth dpotsurf
  tmp=np.fft.fft(dpotsurf)
  freq=nsurf*np.fft.fftfreq(nsurf)
  tmp[abs(freq)>10]=0
  dpotsurf=np.real(np.fft.ifft(tmp))
 
  Hmin=np.zeros((nmu,nPphi),dtype=float)
  Hmax=np.zeros((nmu,nPphi),dtype=float)
  dH=np.zeros((nmu,nPphi,nH),dtype=float)
  #crossing locations of the orbits with the LCFS
  r_beg=np.zeros((nmu,nPphi,nH),dtype=float)
  z_beg=np.zeros((nmu,nPphi,nH),dtype=float)
  r_end=np.zeros((nmu,nPphi,nH),dtype=float)
  z_end=np.zeros((nmu,nPphi,nH),dtype=float)
  multipeak=np.zeros((nmu,nPphi),dtype=int)
  for imu in range(nmu):
    for iPphi in range(nPphi):
      Hsurf=np.zeros((nsurf,),dtype=float)
      for isurf in range(nsurf): 
        #The zonal potential is not needed for H.
        Hsurf[isurf]=(Pphi_arr[iPphi]-qi*psi_surf)**2/covbphisurf[isurf]**2/2/mi\
                     +mu_arr[imu]*Bsurf[isurf]+qi*dpotsurf[isurf]

      pks1,_=find_peaks(Hsurf)
      pks2,_=find_peaks(-Hsurf)
      #TODO: the following assumes b_\varphi<0. Should include the opposite case later
      if (np.size(pks1)>1)or(np.size(pks2)>1):#rare cases when H has multiple peaks&troughs 
        multipeak[imu,iPphi]=1
        nsurf_tmp=0
        #first, determine number of nodes where \partial H/\partial\theta>0 
        for isurf in range(nsurf):
          inext=(isurf+1)%nsurf
          iprev=(isurf-1)%nsurf
          if(Hsurf[inext]>Hsurf[isurf])and(Hsurf[isurf]>Hsurf[iprev]): nsurf_tmp=nsurf_tmp+1
         
        diH=math.floor(nsurf_tmp/(nH+1))
        iH=-1
        isurf=0
        while (isurf<nsurf)and(iH<nH-1):
          inext=(isurf+diH)%nsurf
          iprev=(isurf-diH)%nsurf
          if(Hsurf[inext]>Hsurf[isurf])and(Hsurf[isurf]>Hsurf[iprev]):
            iH=iH+1   
            dH[imu,iPphi,iH]=0.5*(Hsurf[inext]-Hsurf[iprev])
            r_beg[imu,iPphi,iH]=rsurf[isurf]
            z_beg[imu,iPphi,iH]=zsurf[isurf]
            r_end[imu,iPphi,iH]=-1e9
            z_end[imu,iPphi,iH]=-1e9
          isurf=isurf+diH
        while (iH<nH-1): 
          iH=iH+1
          dH[imu,iPphi,iH]=0 
          r_beg[imu,iPphi,iH]=rsurf[0]
          z_beg[imu,iPphi,iH]=zsurf[0]
          r_end[imu,iPphi,iH]=rsurf[0]
          z_end[imu,iPphi,iH]=zsurf[0]
      else:#most likely H only has one minimum and one maximum as follows
        Hmin[imu,iPphi],minloc=min(Hsurf),np.argmin(Hsurf)
        Hmax[imu,iPphi],maxloc=max(Hsurf),np.argmax(Hsurf)
        #diH=math.floor((maxloc-minloc)/(nH+2))
        diH=math.floor((maxloc-minloc)/(nH+1))
        for iH in range(nH):
          isurf=minloc+diH*(iH+1)
          dH[imu,iPphi,iH]=0.5*(Hsurf[isurf+diH]-Hsurf[isurf-diH])
          r_beg[imu,iPphi,iH]=rsurf[isurf]
          z_beg[imu,iPphi,iH]=zsurf[isurf]
          if Hsurf[isurf]<=Hsurf[0]:
            endloc=np.argmin(abs(Hsurf[0:minloc]-Hsurf[isurf]))
          else:
            endloc=maxloc+np.argmin(abs(Hsurf[maxloc:]-Hsurf[isurf]))

          if Hsurf[isurf]>=Hsurf[endloc]:
            wH=(Hsurf[isurf]-Hsurf[endloc])/(Hsurf[endloc-1]-Hsurf[endloc])
            r_end[imu,iPphi,iH]=(1-wH)*rsurf[endloc]+wH*rsurf[endloc-1]
            z_end[imu,iPphi,iH]=(1-wH)*zsurf[endloc]+wH*zsurf[endloc-1]
          else:
            if endloc==nsurf-1:
              nextloc=0
            else:
              nextloc=endloc+1
            wH=(Hsurf[isurf]-Hsurf[endloc])/(Hsurf[nextloc]-Hsurf[endloc])
            r_end[imu,iPphi,iH]=(1-wH)*rsurf[endloc]+wH*rsurf[nextloc]
            z_end[imu,iPphi,iH]=(1-wH)*zsurf[endloc]+wH*zsurf[nextloc]

  return r_beg,z_beg,r_end,z_end

def H2d(mu,Pphi,mi,qi):
  vp2d=(Pphi-qi*psi2d)/mi/R/Bphi*Bmag
  H=mu*Bmag+0.5*mi*vp2d**2+qi*(pot0+dpot)
  gradHr,gradHz,gradHphi=setup.Grad(rlin,zlin,H,rlin.size,zlin.size)
  return H,gradHr,gradHz 
