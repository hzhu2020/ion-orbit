from matplotlib import use
use("Agg")
import matplotlib.pyplot as plt
import variables as var
import numpy as np
from parameters import debug_dir
import os

def plotEB():
  if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
  h=plt.figure()
  ax=plt.subplot(111)
  ax.imshow(var.Er00,origin='lower')
  plt.savefig(debug_dir+'/Er00.pdf')
  ax.imshow(var.Er0m,origin='lower',vmin=-10,vmax=10)
  plt.savefig(debug_dir+'/Er0m.pdf')
  ax.imshow(var.Ez00,origin='lower')
  plt.savefig(debug_dir+'/Ez00.pdf')
  ax.imshow(var.Ez0m,origin='lower',vmin=-10,vmax=10)
  plt.savefig(debug_dir+'/Ez0m.pdf')
  ax.imshow(var.dpot,origin='lower',vmin=-1,vmax=1)
  plt.savefig(debug_dir+'/dpot.pdf')
  ax.imshow(var.Bmag,origin='lower')
  plt.savefig(debug_dir+'/Bmag.pdf')
  plt.close(h)

def plot_gyroE():
  if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
  h=plt.figure()
  ax=plt.subplot(111)
  nmu=np.shape(var.gyroEr00)[2]
  for imu in range(nmu): 
    ax.imshow(var.gyroEr00[:,:,imu],origin='lower')
    plt.savefig(debug_dir+'/gyroEr00_'+str(imu)+'.pdf')
    ax.imshow(var.gyroEz00[:,:,imu],origin='lower')
    plt.savefig(debug_dir+'/gyroEz00_'+str(imu)+'.pdf')
    ax.imshow(var.gyroEr0m[:,:,imu],origin='lower')
    plt.savefig(debug_dir+'/gyroEr0m_'+str(imu)+'.pdf')
    ax.imshow(var.gyroEz0m[:,:,imu],origin='lower')
    plt.savefig(debug_dir+'/gyroEz0m_'+str(imu)+'.pdf')

def write_surf():  
  if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
  output=open(debug_dir+'/surface.txt','w')
  nsurf=np.size(var.rsurf)
  output.write('%8d\n'%nsurf) 
  for i in range(nsurf):
    output.write('%19.10E %19.10E\n'%(var.rsurf[i],var.zsurf[i]))
  output.write('%8d\n'%-1)
  output.close()
  
def plotOrbits(r_orb,z_orb,steps_orb,mynorb,rank): 
  if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
  h=plt.figure()
  ax=plt.subplot(111)
  ax.plot(var.rsurf,var.zsurf,'b',linewidth=0.5)
  for iorb in range(mynorb):
    step=steps_orb[iorb]
    ax.plot(r_orb[iorb,0:step],z_orb[iorb,0:step],linewidth=0.5)
  plt.savefig(debug_dir+'/orbit'+str(rank)+'.pdf')
  plt.close(h)
