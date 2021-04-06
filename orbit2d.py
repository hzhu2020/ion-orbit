import variables as var
import myinterp
import numpy as np
import math
import os
from parameters import cross_psitol,cross_rztol,cross_disttol,debug,debug_dir,determine_loss

def tau_orb(calc_gyroE,iorb,qi,mi,r_beg,z_beg,r_end,z_end,mu,Pphi,dt_xgc,nt,nsteps,max_step):
  global myEr00,myEz00,myEr0m,myEz0m
  if calc_gyroE: myEr00,myEz00,myEr0m,myEz0m=var.efield(iorb)
  
  dt_orb=dt_xgc/float(nsteps)
  dt_orb_out=0.0
  r=r_beg
  z=z_beg
  #prepare the axis position
  ra=var.rsurf[0]-var.dist[0]*math.cos(var.theta[0])
  za=var.zsurf[0]-var.dist[0]*math.sin(var.theta[0])

  ix,iy,wx,wy,Bmag=myinterp.TwoD(var.Bmag,r,z)
  if np.isnan(Bmag):
    print('Wrong initial orbit locations: iorb=',iorb,'r=',r_beg,'z=',z_beg)
    exit()
  Bphi=var.Bphi[iy,ix]*(1-wy)*(1-wx) + var.Bphi[iy+1,ix]*wy*(1-wx)\
      +var.Bphi[iy,ix+1]*(1-wy)*wx + var.Bphi[iy+1,ix+1]*wy*wx
  vp=(Pphi-qi*var.psi_surf)/mi/r/Bphi*Bmag
  #H,dHdr,dHdz=var.H2d(mu,Pphi,mi,qi)
  #H0=myinterp.TwoD(H,r,z)

  r_orb1=np.zeros((nt,),dtype=float)
  z_orb1=np.zeros((nt,),dtype=float)
  vp_orb1=np.zeros((nt,),dtype=float)

  r_tmp=np.zeros((np.int(max_step),),dtype=float)
  z_tmp=np.zeros((np.int(max_step),),dtype=float)
  vp_tmp=np.zeros((np.int(max_step),),dtype=float)

  tau=0
  step_count=0
  num_cross=0 #number of times the orbit has crossed the surface
  lost=False #whether the particle is lost to the wall
  if debug:
    debug_count=0
    if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
    output=open(debug_dir+'/'+str(iorb)+'.txt','w')
    output.write('%8d\n'%0)#a placeholder for debug_count
 
  for it in range(np.int(max_step)):
    if np.isnan(r+z):
      if num_cross==1: lost=True
      break
    if it==nsteps*step_count:
      if (step_count<nt)and(num_cross==0):
        r_orb1[step_count]=r
        z_orb1[step_count]=z
        vp_orb1[step_count]=vp

      if num_cross==0: step_count=step_count+1

    if(debug)and(it%nsteps==0):
      debug_count=debug_count+1
      output.write('%19.10E %19.10E\n'%(r,z))
      
    r_tmp[it]=r
    z_tmp[it]=z
    vp_tmp[it]=vp
    #RK4 1st step
    drdtc,dzdtc,dvpdtc=rhs(qi,mi,r,z,mu,vp)
    dvpdte=dvpdtc/6
    drdte=drdtc/6
    dzdte=dzdtc/6
    vpc=vp+dvpdtc*dt_orb/2
    rc=r+drdtc*dt_orb/2
    zc=z+dzdtc*dt_orb/2
    if np.isnan(rc+zc):
      if num_cross==1: lost=True
      break
    #RK4 2nd step
    drdtc,dzdtc,dvpdtc=rhs(qi,mi,rc,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/3
    drdte=drdte+drdtc/3
    dzdte=dzdte+dzdtc/3
    rc=r+drdtc*dt_orb/2
    zc=z+dzdtc*dt_orb/2
    if np.isnan(rc+zc): 
      if num_cross==1: lost=True
      break
    #RK4 3nd step
    drdtc,dzdtc,dvpdtc=rhs(qi,mi,rc,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/3
    drdte=drdte+drdtc/3
    dzdte=dzdte+dzdtc/3
    rc=r+drdtc*dt_orb
    zc=z+dzdtc*dt_orb
    if np.isnan(rc+zc): 
      if num_cross==1: lost=True
      break
    #Rk4 4th step
    drdtc,dzdtc,dvpdtc=rhs(qi,mi,rc,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/6
    drdte=drdte+drdtc/6
    dzdte=dzdte+dzdtc/6
    vp=vp+dvpdte*dt_orb
    r=r+drdte*dt_orb
    z=z+dzdte*dt_orb
    if np.isnan(r+z): 
      if num_cross==1: lost=True
      break

    tau=tau+dt_orb
    #check if the orbit has crossed the surface
    psi=myinterp.TwoD(var.psi2d,r,z)
    psi=psi[4]
    theta=math.atan2(z-za,r-ra)
    if theta<=var.theta[0]: theta=theta+2*np.pi
    dist=np.sqrt((r-ra)**2+(z-za)**2)
    dist_surf=myinterp.OneD_NL(var.theta,var.dist,theta)
    if (num_cross==0) and (\
       (psi>(1+cross_psitol)*var.psi_surf) \
       or (np.sqrt((r-r_end)**2+(z-z_end)**2)<cross_rztol)\
       or (dist>dist_surf*(1+cross_disttol))
       ): 
      #first time cross (leave) the surface: output orbits here
      num_cross=1 
      if step_count<=nt:
        dt_orb_out=dt_xgc
      else:
        dt_orb_out=tau/np.float(nt)
        step_count=nt
        for it2 in range(nt):
          t_ind=math.floor(np.float(it2)*dt_orb_out/dt_orb)
          wt=np.float(it2)*dt_orb_out/dt_orb - t_ind
          if t_ind==np.int(max_step)-1: #in case the right point is out of the boundary
            t_ind=t_ind-1
            wt=1.0
          if abs(r_tmp[t_ind+1])<1E-3: wt=0.0 #in case the right point has not been assgined value 
          r_orb1[it2]=(1-wt)*r_tmp[t_ind]+wt*r_tmp[t_ind+1]
          z_orb1[it2]=(1-wt)*z_tmp[t_ind]+wt*z_tmp[t_ind+1]
          vp_orb1[it2]=(1-wt)*vp_tmp[t_ind]+wt*vp_tmp[t_ind+1]
      if (not determine_loss): break

    if (num_cross==1) and \
       (psi<(1-cross_psitol)*var.psix) and\
       (z<var.zx):
      lost=True #assume lost if in the private region
      break

    if (num_cross==1) and (\
       (psi<(1-cross_psitol)*var.psi_surf) \
       or (np.sqrt((r-r_beg)**2+(z-z_beg)**2)<cross_rztol)\
       or (dist<dist_surf*(1-cross_disttol))
       ):
      num_cross=2 #second time cross (enter) the surface
      lost=False #redundantly setting lost=F
      break
    #end of the time loop

  if debug:  
    output.write('%8d\n'%-1)
    output.seek(0)
    output.write('%8d\n'%debug_count)
    output.close()
  if step_count>nt: step_count=1
  return lost,tau,dt_orb_out,step_count,r_orb1,z_orb1,vp_orb1

def rhs(qi,mi,r,z,mu,vp):
    #B
    ix,iy,wx,wy,Br=myinterp.TwoD(var.Br,r,z)
    if np.isnan(Br):
      return np.nan,np.nan,np.nan

    Bphi=var.Bphi[iy,ix]*(1-wy)*(1-wx) + var.Bphi[iy+1,ix]*wy*(1-wx)\
        +var.Bphi[iy,ix+1]*(1-wy)*wx + var.Bphi[iy+1,ix+1]*wy*wx
    Bz=var.Bz[iy,ix]*(1-wy)*(1-wx) + var.Bz[iy+1,ix]*wy*(1-wx)\
        +var.Bz[iy,ix+1]*(1-wy)*wx + var.Bz[iy+1,ix+1]*wy*wx
    Bmag=np.sqrt(Br**2+Bz**2+Bphi**2)
    #b
    br=Br/Bmag
    bphi=Bphi/Bmag
    bz=Bz/Bmag
    #E
    Er00=myEr00[iy,ix]*(1-wy)*(1-wx) + myEr00[iy+1,ix]*wy*(1-wx)\
        +myEr00[iy,ix+1]*(1-wy)*wx + myEr00[iy+1,ix+1]*wy*wx
    Ez00=myEz00[iy,ix]*(1-wy)*(1-wx) + myEz00[iy+1,ix]*wy*(1-wx)\
        +myEz00[iy,ix+1]*(1-wy)*wx + myEz00[iy+1,ix+1]*wy*wx
    Er0m=myEr0m[iy,ix]*(1-wy)*(1-wx) + myEr0m[iy+1,ix]*wy*(1-wx)\
        +myEr0m[iy,ix+1]*(1-wy)*wx + myEr0m[iy+1,ix+1]*wy*wx
    Ez0m=myEz0m[iy,ix]*(1-wy)*(1-wx) + myEz0m[iy+1,ix]*wy*(1-wx)\
        +myEz0m[iy,ix+1]*(1-wy)*wx + myEz0m[iy+1,ix+1]*wy*wx
    Er=Er00+Er0m
    Ez=Ez00+Ez0m
    #gradB
    gradBr=var.gradBr[iy,ix]*(1-wy)*(1-wx) + var.gradBr[iy+1,ix]*wy*(1-wx)\
        +var.gradBr[iy,ix+1]*(1-wy)*wx + var.gradBr[iy+1,ix+1]*wy*wx
    gradBz=var.gradBz[iy,ix]*(1-wy)*(1-wx) + var.gradBz[iy+1,ix]*wy*(1-wx)\
        +var.gradBz[iy,ix+1]*(1-wy)*wx + var.gradBz[iy+1,ix+1]*wy*wx
    #curlb
    curlbr=var.curlbr[iy,ix]*(1-wy)*(1-wx) + var.curlbr[iy+1,ix]*wy*(1-wx)\
        +var.curlbr[iy,ix+1]*(1-wy)*wx + var.curlbr[iy+1,ix+1]*wy*wx
    curlbphi=var.curlbphi[iy,ix]*(1-wy)*(1-wx) + var.curlbphi[iy+1,ix]*wy*(1-wx)\
        +var.curlbphi[iy,ix+1]*(1-wy)*wx + var.curlbphi[iy+1,ix+1]*wy*wx
    curlbz=var.curlbz[iy,ix]*(1-wy)*(1-wx) + var.curlbz[iy+1,ix]*wy*(1-wx)\
        +var.curlbz[iy,ix+1]*(1-wy)*wx + var.curlbz[iy+1,ix+1]*wy*wx
    #equation of motion
    rhop=mi*vp/qi/Bmag
    D=1.+rhop*(br*curlbr+bz*curlbz+bphi*curlbphi)
    #dvpdt does not contain dot(b,E00), which is by definition zero
    dvpdt=mu*(br*gradBr+bz*gradBz)+rhop*mu*(curlbr*gradBr+curlbz*gradBz)\
          -qi*rhop*(curlbr*Er+curlbz*Ez)-qi*(br*Er0m+bz*Ez0m)
    dvpdt=dvpdt/(-mi*D)
    drdt=vp*br+vp*rhop*curlbr+Bphi*(mu*gradBz-qi*Ez)/qi/Bmag**2
    drdt=drdt/D
    dzdt=vp*bz+vp*rhop*curlbz-Bphi*(mu*gradBr-qi*Er)/qi/Bmag**2
    dzdt=dzdt/D
    return drdt,dzdt,dvpdt
