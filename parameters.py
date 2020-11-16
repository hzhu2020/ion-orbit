#All paramers are in mks unit
qi=1.60217662E-19#ion charge
mi=2*1.67262192369E-27#ion mass
#f0_vp/smu_max could be the same as in the XGC input, but not's not required
f0_vp_max=4.0 #maximum v_\para normalized by vt
f0_smu_max=4.0 #maximum v_\perp normliaed by vt
potfac=1 #factor that reduces E field in case it is too strong
#input directory containing adios bp files
input_dir='bp_dir'
#which time step from XGC output to use for the temperature and zonal potential;
#0 means the first step and -1 means the last step
temp_step=300 
pot00_step=300 
#number of uniform grid points in r and z
Nr=200
Nz=200
#number of elements in mu, Pphi, H, and t
nmu=8
nPphi=8
nH=8
nt=200
dt_orb=5E-9 #time step size for the orbit integration
max_step=5E4 #max number of time steps for orbit integration
dt_xgc=3.9515E-7 #simulation time step size of XGC
#tolerance parameters for finding LCFS
LCFS_psitol=1E-5
LCFS_rztol=1E-4
#tolerance parameters for determing whether the orbit has crossed the LCFS
cross_psitol=0.005
cross_rztol=0.01
#for debug
debug=True
debug_dir='debug_dir'
