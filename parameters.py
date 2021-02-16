#identify loss orbits
determine_loss=True
#All paramers are in mks unit except Ti
qi=1.60217662E-19#ion charge
mi=2*1.67262192369E-27#ion mass
Ti=300#ion temperature in eV
#f0_vp/smu_max could be the same as in the XGC input, but not's not required
f0_vp_max=4.0 #maximum v_\para normalized by vt
f0_smu_max=4.0 #maximum v_\perp normliaed by vt
pot0fac=0. #factor that reduces 00 E field
dpotfac=1. #factor that reduces turb E field
#input directory containing adios bp files
input_dir='input_dir'
adios_version=2
pot_file='xgc.2d.02000.bp'
#number of uniform grid points in r and z
Nr=500
Nz=500
#method to interpolate from XGC mesh to rectangular mesh
#options: 'linear', 'nearest', 'cubic'
interp_method='cubic'
#number of elements in mu, Pphi, H, and t
nmu=2
nPphi=2
nH=2
nt=50
dt_orb=5E-8 #time step size for the orbit integration
max_step=2E4 #max number of time steps for orbit integration
dt_xgc=3.9515E-7 #simulation time step size of XGC
#parameters for finding the surface
surf_psin=1 #normalized psi, psin=1 for the LCFS
surf_psitol=1E-5
surf_rztol=1E-4
#tolerance parameters for determing whether the orbit has crossed the LCFS
cross_psitol=0.0025 #percentage
cross_rztol=0.005 #meter
cross_disttol=0.005 #percentage
#for debug
debug=True
debug_dir='debug_dir'
