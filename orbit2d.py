import variables as var
import myinterp
import numpy as np
import math
import os
from parameters import max_step,cross_psitol,cross_rztol,cross_disttol,debug,debug_dir

def tau_orb(iorb,qi,mi,x_beg,y,z_beg,r_end,z_end,mu,Pphi,dt_orb,dt_xgc,nt):
  x=x_beg
  z=z_beg
  #prepare the axis position
  ra=var.rsurf[0]-var.dist[0]*math.cos(var.theta[0])
  za=var.zsurf[0]-var.dist[0]*math.sin(var.theta[0])

  r=np.sqrt(x**2+y**2)
  Bmag=myinterp.TwoD(var.Bmag,r,z)
  Bphi=myinterp.TwoD(var.Bphi,r,z)
  vp=(Pphi-qi*var.psi_surf)/mi/r/Bphi*Bmag
  H,dHdr,dHdz=var.H2d(mu,Pphi,mi,qi)
  H0=myinterp.TwoD(H,r,z)

  r_orb1=np.zeros((nt,),dtype=float)
  z_orb1=np.zeros((nt,),dtype=float)
  vp_orb1=np.zeros((nt,),dtype=float)

  r_tmp=np.zeros((np.int(max_step),),dtype=float)
  z_tmp=np.zeros((np.int(max_step),),dtype=float)
  vp_tmp=np.zeros((np.int(max_step),),dtype=float)

  r=np.sqrt(x**2+y**2)
  tau=0
  step_count=0
  num_cross=0 #number of times the orbit has crossed the surface
  lost=False #whether the particle is lost to the wall
  #step_flag=True
  if debug:
    debug_count=0
    if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
    output=open(debug_dir+'/'+str(iorb)+'.txt','w')
    output.write('%8d\n'%0)#a placeholder for debug_count
 
  for it in range(np.int(max_step)):
    if np.isnan(r):
      if num_cross==1: lost=True
      break
    if math.floor(tau/dt_xgc)==step_count:
      #make a correction of position based on H deviation
      #did not work well. Turned it off until further update
      if False:
        psin=myinterp.TwoD(var.psi2d,r,z)/var.psix
        fac=0
        Hc=myinterp.TwoD(H,r,z)
        print(Hc)
        dHdrc=myinterp.TwoD(dHdr,r,z)
        dHdzc=myinterp.TwoD(dHdz,r,z)
        r=r+fac*(H0-Hc)*dHdrc/(dHdrc**2+dHdzc**2)
        z=z+fac*(H0-Hc)*dHdzc/(dHdrc**2+dHdzc**2)

      if step_count<nt:
        r_orb1[step_count]=r
        z_orb1[step_count]=z
        vp_orb1[step_count]=vp

      step_count=step_count+1
      if debug:
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
    if np.isnan(rc):
      if num_cross==1: lost=True
      break
    #RK4 2nd step
    drdtc,dzdtc,dvpdtc=rhs(qi,mi,rc,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/3
    drdte=drdte+drdtc/3
    dzdte=dzdte+dzdtc/3
    rc=r+drdtc*dt_orb/2
    zc=z+dzdtc*dt_orb/2
    if np.isnan(rc): 
      if num_cross==1: lost=True
      break
    #RK4 3nd step
    drdtc,dzdtc,dvpdtc=rhs(qi,mi,rc,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/3
    drdte=drdte+drdtc/3
    dzdte=dzdte+dzdtc/3
    rc=r+drdtc*dt_orb
    zc=z+dzdtc*dt_orb
    if np.isnan(rc): 
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
    if np.isnan(r): 
      if num_cross==1: lost=True
      break

    tau=tau+dt_orb
    #check if the orbit has crossed the surface
    psi=myinterp.TwoD(var.psi2d,r,z)
    theta=math.atan2(z-za,r-ra)
    if theta<=var.theta[0]: theta=theta+2*np.pi
    dist=np.sqrt((r-ra)**2+(z-za)**2)
    dist_surf=myinterp.OneD_NL(var.theta,var.dist,theta)
    if (num_cross==0) and (\
       (psi>(1+cross_psitol)*var.psi_surf) \
       or (np.sqrt((r-r_end)**2+(z-z_end)**2)<cross_rztol)\
       or (dist>dist_surf*(1+cross_disttol))
       ): num_cross==1 #first time cross (leave) the surface
      
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
      num_cross==2 #second time cross (enter) the surface
      break
    #end of the time loop
  print('lost:',lost)
  if step_count<=nt:
    dt_orb_out=dt_xgc
  else:
    dt_orb_out=tau/np.float(nt)
    step_count=nt
    for it in range(nt):
      t_ind=math.floor(np.float(it)*dt_orb_out/dt_orb)
      wt=np.float(it)*dt_orb_out/dt_orb - t_ind
      if t_ind==np.int(max_step)-1: #in case the right point is out of the boundary
        t_ind=t_ind-1
        wt=1.0
      if abs(r_tmp[t_ind+1])<1E-3: wt=0.0 #in case the right point has not been assgined value 
      r_orb1[it]=(1-wt)*r_tmp[t_ind]+wt*r_tmp[t_ind+1]
      z_orb1[it]=(1-wt)*z_tmp[t_ind]+wt*z_tmp[t_ind+1]
      vp_orb1[it]=(1-wt)*vp_tmp[t_ind]+wt*vp_tmp[t_ind+1]

  if debug:  
    output.write('%8d\n'%-1)
    output.seek(0)
    output.write('%8d\n'%debug_count)
    output.close()
  return tau,dt_orb_out,step_count,r_orb1,z_orb1,vp_orb1

def rhs(qi,mi,r,z,mu,vp):
    #B
    Br=myinterp.TwoD(var.Br,r,z)
    Bphi=myinterp.TwoD(var.Bphi,r,z)
    Bz=myinterp.TwoD(var.Bz,r,z)
    Bmag=np.sqrt(Br**2+Bz**2+Bphi**2)
    #b
    br=Br/Bmag
    bphi=Bphi/Bmag
    bz=Bz/Bmag
    #E
    Er00=myinterp.TwoD(var.Er00,r,z)
    Ez00=myinterp.TwoD(var.Ez00,r,z)
    Er0m=myinterp.TwoD(var.Er0m,r,z)
    Ez0m=myinterp.TwoD(var.Ez0m,r,z)
    Er=Er00+Er0m
    Ez=Ez00+Ez0m
    #gradB
    gradBr=myinterp.TwoD(var.gradBr,r,z)
    gradBz=myinterp.TwoD(var.gradBz,r,z)
    #curlb
    curlbr=myinterp.TwoD(var.curlbr,r,z)
    curlbphi=myinterp.TwoD(var.curlbphi,r,z)
    curlbz=myinterp.TwoD(var.curlbz,r,z)
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
