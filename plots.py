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
  ax.imshow(var.Ez,origin='lower')
  plt.savefig(debug_dir+'/Ez.pdf')
  ax.imshow(var.Bmag,origin='lower')
  plt.savefig(debug_dir+'/Bmag.pdf')
  plt.close(h)

def writeLCFS():  
  if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
  output=open(debug_dir+'/LCFS.txt','w')
  nLCFS=np.size(var.rLCFS)
  output.write('%8d\n'%nLCFS) 
  for i in range(nLCFS):
    output.write('%19.10E %19.10E\n'%(var.rLCFS[i],var.zLCFS[i]))
  output.write('%8d\n'%-1)
  output.close()
  
def plotOrbits(r_orb,z_orb,steps_orb,mynorb,rank): 
  if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
  h=plt.figure()
  ax=plt.subplot(111)
  ax.plot(var.rLCFS,var.zLCFS,'b',linewidth=0.5)
  for iorb in range(mynorb):
    step=steps_orb[iorb]
    ax.plot(r_orb[iorb,0:step],z_orb[iorb,0:step],linewidth=0.5)
  plt.savefig(debug_dir+'/orbit'+str(rank)+'.pdf')
  plt.close(h)
