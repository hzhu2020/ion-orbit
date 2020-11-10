import numpy as np
import math
import setup
import myinterp
import orbit
from parameters import *
import variables as var

#initialize some global variables
Ta=setup.Tempix(temp_step)
vt=np.sqrt(qi*Ta/mi) #thermal speed

var.init(potfac,Nr,Nz,temp_step,pot00_step)

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
#determine orbit trajectories through RK4 integration
r_orb=np.zeros((nmu,nPphi,nH,nt),dtype=float)
z_orb=np.zeros((nmu,nPphi,nH,nt),dtype=float)
phi_orb=np.zeros((nmu,nPphi,nH,nt),dtype=float)
vp_orb=np.zeros((nmu,nPphi,nH,nt),dtype=float)
steps_orb=np.zeros((nmu,nPphi,nH),dtype=int)
for imu in range(nmu):
  mu=mu_arr[imu]
  for iPphi in range(nPphi):
    Pphi=Pphi_arr[iPphi]
    for iH in range(nH):
      x,y,z=r_beg[imu,iPphi,iH],0,z_beg[imu,iPphi,iH]
      step,r_orb1,z_orb1,phi_orb1,vp_orb1=orbit.tau_orb(qi,mi,x,y,z,\
        r_end[imu,iPphi,iH],z_end[imu,iPphi,iH],mu,Pphi,dt_orb,dt_xgc,nt)
      print(imu,iPphi,iH)
      r_orb[imu,iPphi,iH,:]=r_orb1
      z_orb[imu,iPphi,iH,:]=z_orb1
      phi_orb[imu,iPphi,iH,:]=phi_orb1
      vp_orb[imu,iPphi,iH,:]=vp_orb1
      steps_orb[imu,iPphi,iH]=step
#import matplotlib.pyplot as plt
#plt.imshow(Ez,origin='lower')
#plt.savefig('test.pdf')
output=open('orbit.txt','w')
output.write('%8d%8d%8d%8d\n'% (nmu,nPphi,nH,nt))
count=0
#orbit steps
for imu in range(nmu):
  for iPphi in range(nPphi):
    for iH in range(nH):
      count=count+1
      value=steps_orb[imu,iPphi,iH]
      output.write('%8d'%value)
      if count%4==0: output.write('\n')
if count%4!=0: output.write('\n')
#mu array
count=0
for imu in range(nmu):
  count=count+1
  value=mu_arr[imu]
  output.write('%19.10E '%value)
  if count%4==0: output.write('\n')
if count%4!=0: output.write('\n')
#R_orb
count=0
for imu in range(nmu):
  for iPphi in range(nPphi):
    for iH in range(nH):
      for it in range(nt):
        count=count+1
        value=r_orb[imu,iPphi,iH,it]
        output.write('%19.10E '%value)
        if count%4==0: output.write('\n')
if count%4!=0: output.write('\n')
#Z_orb
count=0
for imu in range(nmu):
  for iPphi in range(nPphi):
    for iH in range(nH):
      for it in range(nt):
        count=count+1
        value=z_orb[imu,iPphi,iH,it]
        output.write('%19.10E '%value)
        if count%4==0: output.write('\n')
if count%4!=0: output.write('\n')
#phi_orb
count=0
for imu in range(nmu):
  for iPphi in range(nPphi):
    for iH in range(nH):
      for it in range(nt):
        count=count+1
        value=phi_orb[imu,iPphi,iH,it]
        output.write('%19.10E '%value)
        if count%4==0: output.write('\n')
if count%4!=0: output.write('\n')
#vp_orb
count=0
for imu in range(nmu):
  for iPphi in range(nPphi):
    for iH in range(nH):
      for it in range(nt):
        count=count+1
        value=vp_orb[imu,iPphi,iH,it]
        output.write('%19.10E '%value)
        if count%4==0: output.write('\n')
if count%4!=0: output.write('\n')
#end flag
output.write('%8d\n'%-1)
output.close()
