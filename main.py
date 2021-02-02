import numpy as np
import math
import setup
import myinterp
from parameters import qi,mi,Ti,f0_vp_max,f0_smu_max,pot00fac,pot0mfac,temp_step,pot00_step,\
                       Nr,Nz,nmu,nPphi,nH,nt,dt_orb,dt_xgc,debug,twod
import variables as var
from mpi4py import MPI
import plots
import time
if twod:
  import orbit2d as orbit
else:
  import orbit3d as orbit

#MPI is automatically initialized when imported; likewise, no need to call finalize.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#initialize some global variables
vt=np.sqrt(1.60217552E-19*Ti/mi) #thermal speed
var.init(pot00fac,pot0mfac,Nr,Nz,pot00_step,comm,rank)
myinterp.init(var.R,var.Z)
if (debug) and (rank==0):
  plots.plotEB()
  plots.write_surf()
#setup orbit arrays
mu_max=(f0_smu_max**2)*mi*(vt**2)/2/var.Ba
vp_max=f0_vp_max*vt
mu_arr=np.linspace(0,mu_max,nmu)
mu_arr[0]=mu_arr[1]/2 #to avoid zero mu
vphi_arr=np.linspace(-vp_max,vp_max,nPphi)
#vphi_arr is the estimated value for v_\para; they also differ by a sign if Bphi<0
if var.Bphi[math.floor(Nz/2),math.floor(Nr/2)]>0:
  Pphi_arr=mi*var.Ra*vphi_arr+qi*var.psi_surf
else:
  Pphi_arr=-mi*var.Ra*vphi_arr+qi*var.psi_surf
#H array
if rank==0:
  r_beg,z_beg,r_end,z_end=var.H_arr(qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr)
else:
  r_beg,z_beg,r_end,z_end=[None]*4
r_beg,z_beg,r_end,z_end=comm.bcast((r_beg,z_beg,r_end,z_end),root=0)
#output arrays for integration
if rank==0:
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
#partition orbits among processes
if rank==0:
  norb=nmu*nPphi*nH
  norb_avg=int(norb/size)
  norb_last=norb-norb_avg*(size-1)
  norb_list=np.zeros((size,1),dtype=int)
  norb_list[:]=norb_avg
  if norb_last>norb_avg:
    for irank in range(norb_last-norb_avg): norb_list[irank]=norb_list[irank]+1
else:
  norb_list=None

norb_list=comm.bcast(norb_list,root=0)
mynorb=int(norb_list[rank])
iorb1=int(sum(norb_list[0:rank]))
iorb2=iorb1+mynorb-1
#determine orbit trajectories through RK4 integration
r_orb=np.zeros((mynorb,nt),dtype=float,order='C')
z_orb=np.zeros((mynorb,nt),dtype=float,order='C')
#phi_orb=np.zeros((mynorb,nt),dtype=float,order='C')
vp_orb=np.zeros((mynorb,nt),dtype=float,order='C')
steps_orb=np.zeros((mynorb,),dtype=int)
dt_orb_out_orb=np.zeros((mynorb,),dtype=float)
if debug: tau_orb=np.zeros((mynorb,),dtype=float)
t_beg_tot=time.time()
for iorb in range(iorb1,iorb2+1):
  imu=int(iorb/(nPphi*nH))
  iPphi=int((iorb-imu*nPphi*nH)/nH)
  iH=iorb-imu*nPphi*nH-iPphi*nH
  mu=mu_arr[imu]
  Pphi=Pphi_arr[iPphi]
  x,y,z=r_beg[imu,iPphi,iH],0,z_beg[imu,iPphi,iH]
  t_beg=time.time()
  tau,dt_orb_out,step,r_orb1,z_orb1,vp_orb1=orbit.tau_orb(iorb,qi,mi,x,y,z,\
      r_end[imu,iPphi,iH],z_end[imu,iPphi,iH],mu,Pphi,dt_orb,dt_xgc,nt)
  t_end=time.time()
  print('rank=',rank,', orb=',iorb,', tau=',tau,', cpu time=',t_end-t_beg,'s',flush=True)
  r_orb[iorb-iorb1,:]=r_orb1
  z_orb[iorb-iorb1,:]=z_orb1
  #phi_orb[iorb-iorb1,:]=phi_orb1
  vp_orb[iorb-iorb1,:]=vp_orb1
  steps_orb[iorb-iorb1]=step
  dt_orb_out_orb[iorb-iorb1]=dt_orb_out
  if debug: tau_orb[iorb-iorb1]=tau
t_end_tot=time.time()
#rank=0 gather data and output
if rank==0:
  count1=norb_list
  count2=count1*nt
  steps_output=np.zeros((norb,),dtype=int)
  dt_orb_output=np.zeros((norb,),dtype=float)
  if debug: tau_output=np.zeros((norb,),dtype=float)
  r_output=np.zeros((norb,nt),dtype=float,order='C')
  z_output=np.zeros((norb,nt),dtype=float,order='C')
  #phi_output=np.zeros((norb,nt),dtype=float,order='C')
  vp_output=np.zeros((norb,nt),dtype=float,order='C')
else:
  if debug: tau_output=None
  steps_output,dt_orb_output,r_output,z_output,vp_output,count1,count2=[None]*7
comm.barrier()
print('rank=',rank,'total cpu time=',(t_end_tot-t_beg_tot)/60.0,'min')
comm.Gatherv(steps_orb,(steps_output,count1),root=0)
comm.Gatherv(dt_orb_out_orb,(dt_orb_output,count1),root=0)
if debug: comm.Gatherv(tau_orb,(tau_output,count1),root=0)
comm.Gatherv(r_orb,(r_output,count2),root=0)
comm.Gatherv(z_orb,(z_output,count2),root=0)
#comm.Gatherv(phi_orb,(phi_output,count2),root=0)
comm.Gatherv(vp_orb,(vp_output,count2),root=0)
if rank==0:
  #output tau separately for debug
  if debug:
    output=open('tau.txt','w')
    count=0
    for iorb in range(norb):
      count=count+1
      value=tau_output[iorb]
      output.write('%19.10E'%value)
      if count%4==0: output.write('\n')
  #the following are for xgc to read
  output=open('orbit.txt','w')
  output.write('%8d%8d%8d%8d\n'% (nmu,nPphi,nH,nt))
  #orbit steps
  count=0
  for iorb in range(norb):
    count=count+1
    value=steps_output[iorb]
    output.write('%8d'%value)
    if count%4==0: output.write('\n')
  if count%4!=0: output.write('\n')
  #orbit step size
  count=0
  for iorb in range(norb):
    count=count+1
    value=dt_orb_output[iorb]
    output.write('%19.10E '%value)
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
  #r_orb
  count=0
  for iorb in range(norb):
    for it in range(nt):
      count=count+1
      value=r_output[iorb,it]
      output.write('%19.10E '%value)
      if count%4==0: output.write('\n')
  if count%4!=0: output.write('\n')
  #z_orb
  count=0
  for iorb in range(norb):
    for it in range(nt):
      count=count+1
      value=z_output[iorb,it]
      output.write('%19.10E '%value)
      if count%4==0: output.write('\n')
  if count%4!=0: output.write('\n')
  #phi_orb
#  count=0
#  for iorb in range(norb):
#    for it in range(nt):
#      count=count+1
#      value=phi_output[iorb,it]
#      output.write('%19.10E '%value)
#      if count%4==0: output.write('\n')
#  if count%4!=0: output.write('\n')
  #vp_orb
  count=0
  for iorb in range(norb):
    for it in range(nt):
      count=count+1
      value=vp_output[iorb,it]
      output.write('%19.10E '%value)
      if count%4==0: output.write('\n')
  if count%4!=0: output.write('\n')
  #end flag
  output.write('%8d\n'%-1)
  output.close()
#end of output from rank=0
