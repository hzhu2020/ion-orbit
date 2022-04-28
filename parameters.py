#write to bp file using adios2
bp_write=True
#identify loss orbits
determine_loss=False
#gyroaverage the electric field
gyro_E=True
ngyro=8
#use XGC's grid derivatives to compute E
grid_E=True
if (grid_E)and(gyro_E): gyro_E=False
#All paramers are in mks unit except Ti
qi=1.60217662E-19#ion charge
mi=2*1.67262192369E-27#ion mass
Ti=270#ion temperature in eV
#f0_vp/smu_max could be the same as in the XGC input, but that's not required
f0_vp_max=5. #maximum v_\para normalized by vt
f0_smu_max=5. #maximum v_\perp normliaed by vt
pot0fac=1. #factor that reduces 00 E field
dpotfac=1. #factor that reduces turb E field
dpot_fourier_maxm=10 #maximum poloidal Fourier mode number for dpot
#input directory containing adios bp files
xgc='xgc1'#'xgc1' or 'xgca'
input_dir='input_dir'
adios_version=2
pot_file='xgc.2d.03000.bp'
#number of uniform grid points in r and z
Nr=1000
Nz=1000
#method to interpolate from XGC mesh to rectangular mesh
#options: 'linear', 'nearest', 'cubic'
interp_method='cubic'
#number of elements in mu, Pphi, H, and t
partition_opt=True#optimize orbit partition to reduce load imbalance, need input_dir/tau.txt
nmu=60
nPphi=60
nH=120
nt=200
max_step=1E4 #max number of time steps for orbit integration
dt_xgc=3.9515E-7 #simulation time step size of XGC
nsteps=1 #the orbit-integration timestep is dt_orb=dt_xgc/nsteps
#parameters for finding the surface
surf_psin=0.982 #normalized psi, psin=1 for the LCFS
surf_psitol=1E-5
surf_rztol=1E-4
#tolerance parameters for determing whether the orbit has crossed the LCFS
cross_psitol=0.0025 #percentage
cross_rztol=0.005 #meter
cross_disttol=0.005 #percentage
#for debug
debug=False
debug_dir='debug_dir'

def write_parameters(output):
  import numpy as np
  output.write_attribute('determine_loss',np.array(1*determine_loss))
  output.write_attribute('gyro_E',np.array(1*gyro_E))
  output.write_attribute('grid_E',np.array(1*grid_E))
  output.write_attribute('ngyro',np.array(ngyro))
  output.write_attribute('qi',np.array(qi))
  output.write_attribute('mi',np.array(mi))
  output.write_attribute('Ti',np.array(Ti))
  output.write_attribute('f0_vp_max',np.array(f0_vp_max))
  output.write_attribute('f0_smu_max',np.array(f0_smu_max))
  output.write_attribute('pot0fac',np.array(pot0fac))
  output.write_attribute('dpotfac',np.array(dpotfac))
  output.write_attribute('dpot_fourier_maxm',np.array(dpot_fourier_maxm))
  output.write_attribute('input_dir',input_dir)
  output.write_attribute('pot_file',pot_file)
  output.write_attribute('Nr',np.array(Nr))
  output.write_attribute('Nz',np.array(Nz))
  output.write_attribute('interp_method',interp_method)
  output.write_attribute('nmu',np.array(nmu))
  output.write_attribute('nPphi',np.array(nPphi))
  output.write_attribute('nH',np.array(nH))
  output.write_attribute('nt',np.array(nt))
  output.write_attribute('max_step',np.array(max_step))
  output.write_attribute('dt_xgc',np.array(dt_xgc))
  output.write_attribute('nsteps',np.array(nsteps))
  output.write_attribute('surf_psin',np.array(surf_psin))
  output.write_attribute('surf_rztol',np.array(surf_rztol))
  output.write_attribute('surf_psitol',np.array(surf_psitol))
  output.write_attribute('cross_psitol',np.array(cross_psitol))
  output.write_attribute('cross_rztol',np.array(cross_rztol))
  output.write_attribute('cross_disttol',np.array(cross_disttol))
