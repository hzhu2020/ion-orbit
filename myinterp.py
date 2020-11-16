import math
import numpy as np

def init(x2d,y2d):
  global Nx,Ny,dx,dy,x0,y0
  Ny,Nx=np.shape(x2d)
  x0=x2d[0,0]
  dx=x2d[0,1]-x0
  y0=y2d[0,0]
  dy=y2d[1,0]-y0

def OneD(x1d,f1d,xin):
  Nx=np.size(x1d)
  x0=x1d[0]
  dx=x1d[1]-x0  
  ix=math.floor((xin-x0)/dx)
  wx=(xin-x0)/dx-ix
  if (ix<0) or (ix>Nx-2):
    fout=np.nan
  else:
    fout=f1d[ix]*(1-wx)+f1d[ix+1]*wx

  return fout

def TwoD(f2d,xin,yin):
  if xin==np.nan:
    print(xin,yin,dx,dy)
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
