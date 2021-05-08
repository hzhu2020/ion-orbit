import adios2 as ad
import numpy as np
xgc_dir='/path/to/XGCa'
#time indices for XGCa's 2d diagnostics
step_start=2020
step_end=4000
dstep=20
nsteps=(step_end-step_start)/dstep+1
for gstep in range(step_start,step_end+dstep,dstep):
  fname=xgc_dir+'/xgc.2d.'+'{:0>5d}'.format(gstep)+'.bp'
  fid=ad.open(fname,'r')
  pot0=fid.read('pot0')
  dpot=fid.read('dpot')
  fid.close()
  if gstep==step_start:
    nnodes=len(pot0)
    pot0avg=np.array(pot0)
    dpotavg=np.array(dpot)
  else:
    pot0avg=pot0avg+np.array(pot0)
    dpotavg=dpotavg+np.array(dpot)

pot0avg=pot0avg/float(nsteps)
dpotavg=dpotavg/float(nsteps)
#for generating orbit files
output=ad.open('avgpot.bp','w')
output.write_attribute('xgc_dir',xgc_dir)
output.write_attribute('step_start',np.array(step_start))
output.write_attribute('step_end',np.array(step_end))
output.write_attribute('dstep',np.array(dstep))
start=np.zeros((pot0avg.ndim),dtype=int) 
count=np.array((pot0avg.shape),dtype=int) 
shape=count
output.write('pot0',pot0avg,shape,start,count)
start=np.zeros((dpotavg.ndim),dtype=int) 
count=np.array((dpotavg.shape),dtype=int) 
shape=count
output.write('dpot',dpotavg,shape,start,count)
output.close()
#for turbulence orbit-loss diagnostics
output=open('pot0m.txt','w')
output.write('%8d\n'%nnodes)
for i in range(nnodes):
  output.write('%19.10E\n'%(pot0avg[i]+dpotavg[i]))
output.write('%8d\n'%-1)
output.close()
