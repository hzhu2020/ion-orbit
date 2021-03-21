#write to bp file using adios2
bp_write=True
#identify loss orbits
determine_loss=False
#gyroaverage the electric field
gyro_E=True
ngyro=8
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
