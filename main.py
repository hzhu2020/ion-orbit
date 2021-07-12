import numpy as np
import math
import setup
import myinterp
from parameters import partition_opt,bp_write,qi,mi,Ti,f0_vp_max,f0_smu_max,pot0fac,dpotfac,\
                       Nr,Nz,nmu,nPphi,nH,nt,dt_xgc,nsteps,max_step,debug,determine_loss,\
                       gyro_E,write_parameters
if gyro_E: from parameters import ngyro
import variables as var
from mpi4py import MPI
if debug: import plots
import time
import orbit2d as orbit

#MPI is automatically initialized when imported; likewise, no need to call finalize.
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
try:
  import cupy as cp
  if rank==0: print('Using CuPy for GPU acceleration.\nOrbits are not outputed to debug_dir in GPU mode.')
  use_gpu=True
except:
  use_gpu=False

#initialize some global variables
vt=np.sqrt(1.60217552E-19*Ti/mi) #thermal speed
t_beg=time.time()
var.init(pot0fac,dpotfac,Nr,Nz,comm,rank)
myinterp.init(var.R,var.Z)
t_end=time.time()
if rank==0: print('Initialization took time:',t_end-t_beg,'s',flush=True)
if (debug) and (rank==0):
  plots.plotEB()
  plots.write_surf()
#setup orbit arrays
mu_max=(f0_smu_max**2)*mi*(vt**2)/2/var.Ba
vp_max=f0_vp_max*vt
mu_arr=np.linspace(0,mu_max,nmu)
mu_arr[0]=mu_arr[1]/2 #to avoid zero mu
vphi_arr=np.linspace(-vp_max,vp_max,nPphi)
#prepare gyro-averaged electric field
if (gyro_E):
  t_beg=time.time()
  if use_gpu:
    var.gyropot_gpu(mu_arr,qi,mi,ngyro)
  else:
    var.gyropot(comm,mu_arr,qi,mi,ngyro,MPI.SUM)
  t_end=time.time()
  if rank==0: print('Gyroavering electric field took time:',t_end-t_beg,'s',flush=True)
#vphi_arr is the estimated value for v_\para; they also differ by a sign if Bphi<0
if var.Bphi[math.floor(Nz/2),math.floor(Nr/2)]>0:
  Pphi_arr=mi*var.Ra*vphi_arr+qi*var.psi_surf
else:
  Pphi_arr=-mi*var.Ra*vphi_arr+qi*var.psi_surf
#H array
t_beg=time.time()
r_beg,z_beg,r_end,z_end=var.H_arr(comm,qi,mi,nmu,nPphi,nH,mu_arr,Pphi_arr,MPI.SUM)
t_end=time.time()
if rank==0: print('Setting up H array took time:',t_end-t_beg,'s',flush=True)
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
t_beg=time.time()
iorb1,iorb2,norb_list=var.partition_orbits(comm,partition_opt,nmu,nPphi,nH)
mynorb=iorb2-iorb1+1
norb=nmu*nPphi*nH
t_end=time.time()
if rank==0: print('Partition orbits took time:',t_end-t_beg,'s',flush=True)
#determine orbit trajectories through RK4 integration
r_orb=np.zeros((mynorb,nt),dtype=float,order='C')
z_orb=np.zeros((mynorb,nt),dtype=float,order='C')
vp_orb=np.zeros((mynorb,nt),dtype=float,order='C')
steps_orb=np.zeros((mynorb,),dtype=int)
dt_orb_out_orb=np.zeros((mynorb,),dtype=float)
if determine_loss: loss_orb=np.zeros((mynorb,),dtype=int)
tau_orb=np.zeros((mynorb,),dtype=float)
t_beg_tot=time.time()
pctg=0.01
for iorb in range(iorb1,iorb2+1):
  imu=int(iorb/(nPphi*nH))
  iPphi=int((iorb-imu*nPphi*nH)/nH)
  iH=iorb-imu*nPphi*nH-iPphi*nH
  mu=mu_arr[imu]
  Pphi=Pphi_arr[iPphi]
  r,z=r_beg[imu,iPphi,iH],z_beg[imu,iPphi,iH]
  if iorb==iorb1:
    calc_gyroE=True
    mu_old=mu
  elif mu!=mu_old:
    calc_gyroE=True
    mu_old=mu
  else:
    calc_gyroE=False
  if False:#use_gpu:
    lost,tau,dt_orb_out,step,r_orb1,z_orb1,vp_orb1=orbit.tau_orb_gpu(calc_gyroE,iorb,qi,mi,r,z,\
        r_end[imu,iPphi,iH],z_end[imu,iPphi,iH],mu,Pphi,dt_xgc,nt,nsteps,max_step)
  else:
    lost,tau,dt_orb_out,step,r_orb1,z_orb1,vp_orb1=orbit.tau_orb(calc_gyroE,iorb,qi,mi,r,z,\
        r_end[imu,iPphi,iH],z_end[imu,iPphi,iH],mu,Pphi,dt_xgc,nt,nsteps,max_step)
  if (float(iorb-iorb1)/float(mynorb))>pctg:
    t_end=time.time()
    print('rank=',rank,'finished',int(pctg*100),'% in ',t_end-t_beg_tot,'s',flush=True)
    pctg=pctg+0.01
  r_orb[iorb-iorb1,:]=r_orb1
  z_orb[iorb-iorb1,:]=z_orb1
  vp_orb[iorb-iorb1,:]=vp_orb1
  steps_orb[iorb-iorb1]=step
  dt_orb_out_orb[iorb-iorb1]=dt_orb_out
  if (lost)and(determine_loss): loss_orb[iorb-iorb1]=1
  tau_orb[iorb-iorb1]=tau
t_end_tot=time.time()
comm.barrier()
time.sleep(rank*0.001)
print('rank=',rank,'total time=',(t_end_tot-t_beg_tot)/60.0,'min',flush=True)
#output which orbits are lost
if determine_loss:
  if rank==0:
    count1=norb_list
    loss_output=np.zeros((norb,),dtype=int)
  else:
    count1=None
    loss_output=None
  comm.Gatherv(loss_orb,(loss_output,count1),root=0)
  if rank==0:
    output=open('lost.txt','w')
    output.write('%8d%8d%8d\n'% (nmu,nPphi,nH))
    for iorb in range(norb):
      value=loss_output[iorb]
      output.write('%1d\n'%value)
    output.write('%8d\n'%-1)
    output.close()
comm.barrier()
#output tau separately for debug and orbit partition optimization
if rank==0:
  count1=norb_list
  tau_output=np.zeros((norb,),dtype=float)
else:
  tau_output=None
  count1=None
comm.Gatherv(tau_orb,(tau_output,count1),root=0)
if rank==0:
  output=open('tau.txt','w')
  output.write('%8d%8d%8d\n'% (nmu,nPphi,nH))
  count=0
  for iorb in range(norb):
    count=count+1
    value=tau_output[iorb]
    output.write('%19.10E '%value)
    if count%4==0: output.write('\n')
comm.barrier()

#the following are for xgc to read
adios2_mpi=False
if bp_write:
  import adios2
  #Test if adios2 supports MPI. There should be a better way.
  try:
    output=adios2.open('orbit.bp','w',comm)
    adios2_mpi=True
  except:
    if rank==0: print('Adios2 does not support MPI.',flush=True)
    adios2_mpi=False

if (not bp_write)or(not adios2_mpi):
  #rank=0 gather data and output
  if rank==0:
    count1=norb_list
    count2=count1*nt
    steps_output=np.zeros((norb,),dtype=int)
    dt_orb_output=np.zeros((norb,),dtype=float)
    r_output=np.zeros((norb,nt),dtype=float,order='C')
    z_output=np.zeros((norb,nt),dtype=float,order='C')
    vp_output=np.zeros((norb,nt),dtype=float,order='C')
  else:
    steps_output,dt_orb_output,r_output,z_output,vp_output,count1,count2=[None]*7
  comm.Gatherv(steps_orb,(steps_output,count1),root=0)
  comm.Gatherv(dt_orb_out_orb,(dt_orb_output,count1),root=0)
  comm.Gatherv(r_orb,(r_output,count2),root=0)
  comm.Gatherv(z_orb,(z_output,count2),root=0)
  comm.Gatherv(vp_orb,(vp_output,count2),root=0)

if (rank==0)and(not bp_write):
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
elif (rank==0)and(bp_write)and(not adios2_mpi):
  output=adios2.open('orbit.bp','w')
  write_parameters(output)
  #nmu, nPphi, nH, nt
  value=np.array(nmu)
  start=np.zeros((value.ndim),dtype=int) 
  count=np.array((value.shape),dtype=int) 
  shape=count
  output.write('nmu',value,shape,start,count)
  value=np.array(nPphi)
  output.write('nPphi',value,shape,start,count)
  value=np.array(nH)
  output.write('nH',value,shape,start,count)
  value=np.array(nt)
  output.write('nt',value,shape,start,count)
  #steps_orb, dt_orb
  start=np.zeros((steps_output.ndim),dtype=int) 
  count=np.array((steps_output.shape),dtype=int) 
  shape=count
  output.write('steps_orb',steps_output,shape,start,count)
  output.write('dt_orb',dt_orb_output,shape,start,count)
  #mu_orb
  start=np.zeros((mu_arr.ndim),dtype=int) 
  count=np.array((mu_arr.shape),dtype=int) 
  shape=count
  output.write('mu_orb',mu_arr,shape,start,count)
  #R_orb, Z_orb, vp_orb
  start=np.zeros((r_output.ndim),dtype=int) 
  count=np.array((r_output.shape),dtype=int)
  shape=count
  output.write('R_orb',r_output,shape,start,count)
  output.write('Z_orb',z_output,shape,start,count)
  output.write('vp_orb',vp_output,shape,start,count)
  output.close()
elif (bp_write)and(adios2_mpi):
  if rank==0:
    write_parameters(output)
    value=np.array(nmu)
    start=np.zeros((value.ndim),dtype=int) 
    count=np.array((value.shape),dtype=int) 
    shape=count
    output.write('nmu',np.array(nmu),shape,start,count)
    output.write('nPphi',np.array(nPphi),shape,start,count)
    output.write('nH',np.array(nH),shape,start,count)
    output.write('nt',np.array(nt),shape,start,count)
    start=np.zeros((mu_arr.ndim),dtype=int) 
    count=np.array((mu_arr.shape),dtype=int) 
    shape=count
    output.write('mu_orb',mu_arr,shape,start,count)

  norb=nmu*nPphi*nH
  shape=np.array([norb,],dtype=int)
  start=np.array([iorb1,],dtype=int) 
  count=np.array([iorb2-iorb1+1,],dtype=int)
  output.write('steps_orb',steps_orb,shape,start,count)
  output.write('dt_orb',dt_orb_out_orb,shape,start,count)
  shape=np.array([norb,nt],dtype=int)
  start=np.array([iorb1,0],dtype=int) 
  count=np.array([iorb2-iorb1+1,nt],dtype=int)
  output.write('R_orb',r_orb,shape,start,count)
  output.write('Z_orb',z_orb,shape,start,count)
  output.write('vp_orb',vp_orb,shape,start,count)
  output.close()
#end of output
comm.barrier()
