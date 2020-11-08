import numpy as np
import math
import setup
import myinterp
import orbit
import variables as var

#parameters and initialize some global variables
qi=1.60217662E-19
mi=2*1.67262192369E-27 
f0_vp_max=4.0
f0_smu_max=4.0
potfac=0. #factor that reduces E field in case it is too strong
temp_step=0
Nr,Nz=200,200 #number of uniform grid points in r and z
pot00_step=-1 #-1 means the last step
Ta=setup.Tempix(temp_step)
vt=np.sqrt(qi*Ta/mi) #thermal speed

var.init(potfac,Nr,Nz,temp_step,pot00_step)

#setup orbit arrays
nmu,nvp,nH=3,1,4
nPphi=2*nvp+1
#mu array and P_phi array
mu_max=(f0_smu_max**2)*mi*(vt**2)/2/var.Ba
vp_max=f0_vp_max*vt
mu_arr=np.linspace(0,mu_max,nmu)
mu_arr[0]=mu_arr[1]/2 #to avoid zero mu
vphi_arr=np.linspace(-vp_max,vp_max,2*nvp+1)
#vphi_arr is the estimated value for v_\para; they also differ by a sign if Bphi<0
if var.Bphi[math.floor(Nz/2),math.floor(Nr/2)]>0:
  Pphi_arr=mi*var.Ra*vphi_arr+qi*var.psix
else:
  Pphi_arr=-mi*var.Ra*vphi_arr+qi*var.psix
#H array
r_beg,z_beg,r_end,z_end=var.H_arr(qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr)
#determine orbit trajectories through RK4 integration
dt_orb=1E-8
dt_xgc=3.9515E-7
nt=200
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
print(steps_orb)
#import matplotlib.pyplot as plt
#plt.imshow(Ez,origin='lower')
#plt.savefig('test.pdf')
