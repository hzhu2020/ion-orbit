import numpy as np
import math
import setup
from parameters import qi,mi,f0_vp_max,f0_smu_max,potfac,temp_step,pot00_step,\
                       Nr,Nz,nmu,nPphi,nH,nt,dt_orb,dt_xgc,debug
import variables as var
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
Ta=setup.Tempix(temp_step)
vt=np.sqrt(qi*Ta/mi) #thermal speed
var.init(potfac,Nr,Nz,pot00_step,comm,rank)
#setup orbit arrays
mu_max=(f0_smu_max**2)*mi*(vt**2)/2/var.Ba
vp_max=f0_vp_max*vt
mu_arr=np.linspace(0,mu_max,nmu)
mu_arr[0]=mu_arr[1]/2 #to avoid zero mu
vphi_arr=np.linspace(-vp_max,vp_max,nPphi)
#vphi_arr is the estimated value for v_\para; they also differ by a sign if Bphi<0
if var.Bphi[math.floor(Nz/2),math.floor(Nr/2)]>0:
  Pphi_arr=mi*var.Ra*vphi_arr+qi*var.psix
else:
  Pphi_arr=-mi*var.Ra*vphi_arr+qi*var.psix
#H array
r_beg,z_beg,r_end,z_end=var.H_arr(qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr)

dmu=mu_arr[0]*2
dPphi=Pphi_arr[1]-Pphi_arr[0]
output=open('integral.txt','w')
output.write('%19.10E %19.10E\n'%(dmu,dPphi))
count=0
for imu in range(nmu):
  for iPphi in range(nPphi):
    for iH in range(nH):
      count=count+1
      value=var.dH[imu,iPphi,iH]
      output.write('%19.10E'%value)
      if count%4==0: output.write('\n')
if count%4!=0: output.write('\n')
output.write('%8d\n'%-1)
output.close()
