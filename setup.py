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

def Pot(rz,rlin,zlin,pot0fac,dpotfac,itask1,itask2):
  if adios_version==1:
    print('Not added yet.',flush=True)
  elif adios_version==2:
    fname=input_dir+'/'+pot_file
    f=ad.open(fname,'r')
    pot0=f.read('pot0')
    dpot=f.read('dpot')
    f.close()
    if xgc=='xgc1':
      print('xgc=xgc1, apply toroidal average to dpot.')
      dpot=np.mean(dpot,axis=0)
    R,Z=np.meshgrid(rlin,zlin)
    if (itask1<=4)and(4<=itask2)and(pot0fac>0):
      pot02d=griddata(rz,pot0,(R,Z),method=interp_method)
    else:
      pot02d=np.zeros(np.shape(R),dtype=float)
    if (itask1<=5)and(5<=itask2)and(dpotfac>0):
      dpot2d=griddata(rz,dpot,(R,Z),method=interp_method)
    else:
      dpot2d=np.zeros(np.shape(R),dtype=float)
  return pot0fac*pot02d,dpotfac*dpot2d
