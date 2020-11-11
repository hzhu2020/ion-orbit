import numpy as np
import math
import setup
import myinterp
import orbit
from parameters import qi,mi,f0_vp_max,f0_smu_max,potfac,temp_step,pot00_step,\
                       Nr,Nz,nmu,nPphi,nH,nt,dt_orb,dt_xgc,debug
import variables as var
from mpi4py import MPI
import plots

#MPI is automatically initialized when imported; likewise, no need to call finalize.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
#initialize some global variables
if rank==0: 
  Ta=setup.Tempix(temp_step)
else:
  Ta=None
Ta=comm.bcast(Ta,root=0)
vt=np.sqrt(qi*Ta/mi) #thermal speed
var.init(potfac,Nr,Nz,pot00_step,comm,rank)
if (debug) and (rank==0): plots.plotEB()
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
if rank==0:
  r_beg,z_beg,r_end,z_end=var.H_arr(qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr)
else:
  r_beg,z_beg,r_end,z_end=[None]*4
r_beg,z_beg,r_end,z_end=comm.bcast((r_beg,z_beg,r_end,z_end),root=0)
#partition orbits among processes
if rank==0:
  norb=nmu*nPphi*nH
  norb_avg=int(norb/size)
  norb_last=norb-norb_avg*(size-1)
  while (norb_last>norb_avg) and (norb_last>size-1):
    norb_avg=norb_avg+1
    norb_last=norb-norb_avg*(size-1)
else:
  norb,norb_avg,norb_last=[None]*3
norb,norb_avg,norb_last=comm.bcast((norb,norb_avg,norb_last),root=0)
if rank<size-1:
  iorb1=norb_avg*rank
  iorb2=iorb1+norb_avg-1
else:
  iorb2=norb-1
  iorb1=norb-norb_last
mynorb=iorb2-iorb1+1
#determine orbit trajectories through RK4 integration
r_orb=np.zeros((mynorb,nt),dtype=float,order='C')
z_orb=np.zeros((mynorb,nt),dtype=float,order='C')
phi_orb=np.zeros((mynorb,nt),dtype=float,order='C')
vp_orb=np.zeros((mynorb,nt),dtype=float,order='C')
steps_orb=np.zeros((mynorb,),dtype=int)
if debug: tau_orb=np.zeros((mynorb,),dtype=float)
for iorb in range(iorb1,iorb2+1):
  imu=int(iorb/(nPphi*nH))
  iPphi=int((iorb-imu*nPphi*nH)/nH)
  iH=iorb-imu*nPphi*nH-iPphi*nH
  mu=mu_arr[imu]
  Pphi=Pphi_arr[iPphi]
  x,y,z=r_beg[imu,iPphi,iH],0,z_beg[imu,iPphi,iH]
  tau,step,r_orb1,z_orb1,phi_orb1,vp_orb1=orbit.tau_orb(qi,mi,x,y,z,\
      r_end[imu,iPphi,iH],z_end[imu,iPphi,iH],mu,Pphi,dt_orb,dt_xgc,nt)
  print(imu,iPphi,iH)
  r_orb[iorb-iorb1,:]=r_orb1
  z_orb[iorb-iorb1,:]=z_orb1
  phi_orb[iorb-iorb1,:]=phi_orb1
  vp_orb[iorb-iorb1,:]=vp_orb1
  steps_orb[iorb-iorb1]=step
  if debug: tau_orb[iorb-iorb1]=tau
if debug: plots.plotOrbits(r_orb,z_orb,steps_orb,mynorb,rank)
#rank=0 gather data and output
if rank==0:
  count1=np.zeros((size,),dtype=int)
  count1[0:size-1]=norb_avg
  count1[size-1]=norb_last
  count2=count1*nt
  steps_output=np.zeros((norb,),dtype=int)
  if debug: tau_output=np.zeros((norb,),dtype=float)
  r_output=np.zeros((norb,nt),dtype=float,order='C')
  z_output=np.zeros((norb,nt),dtype=float,order='C')
  phi_output=np.zeros((norb,nt),dtype=float,order='C')
  vp_output=np.zeros((norb,nt),dtype=float,order='C')
else:
  if debug: tau_output=None
  steps_output,r_output,z_output,phi_output,vp_output,count1,count2=[None]*7
comm.barrier()
comm.Gatherv(steps_orb,(steps_output,count1),root=0)
if debug: comm.Gatherv(tau_orb,(tau_output,count1),root=0)
comm.Gatherv(r_orb,(r_output,count2),root=0)
comm.Gatherv(z_orb,(z_output,count2),root=0)
comm.Gatherv(phi_orb,(phi_output,count2),root=0)
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
  count=0
  for iorb in range(norb):
    for it in range(nt):
      count=count+1
      value=phi_output[iorb,it]
      output.write('%19.10E '%value)
      if count%4==0: output.write('\n')
  if count%4!=0: output.write('\n')
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
