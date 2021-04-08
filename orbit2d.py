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

def tau_orb_gpu(calc_gyroE,iorb,qi,mi,r_beg,z_beg,r_end,z_end,mu,Pphi,dt_xgc,nt,nsteps,max_step):
  import cupy as cp
  orbit=cp.ElementwiseKernel(
  'float64 r_beg, float64 z_beg, float64 qi, float64 mi, float64 mu, int64 max_step, int64 nsteps, int64 nt,\
   raw float64 Br, raw float64 Bz, raw float64 Bphi, raw float64 gradBr, raw float64 gradBz,\
   raw float64 curlbr, raw float64 curlbz, raw float64 curlbphi, raw float64 Er00, raw float64 Ez00,\
   raw float64 Er0m, raw float64 Ez0m, raw float64 rlin, raw float64 zlin, float64 dt_orb,\
   raw float64 psi2d, raw float64 theta, float64 ra, float64 za, raw float64 dist,float64 psi_surf,\
   float64 r_end, float64 z_end, float64 cross_psitol, float64 cross_rztol, float64 cross_disttol,\
   bool determine_loss, float64 psix, float64 zx',
  'int64 num_cross, int64 step_count, bool lost, float64 vp, raw float64 r_orb1, raw float64 z_orb1,\
   raw float64 vp_orb1, float64 tau, float64 dt_orb_out',
  '''
  int it,it2,irk,iz,ir,Nr,Nz,Nsurf,itheta,t_ind;
  double r,z,rc,zc,vpc,drdtc,dzdtc,dvpdtc,drdte,dzdte,dvpdte;
  double dr,dz,dtheta,r0,z0,wr,wz,wtheta,wt,Er,Ez,Bmag,br,bz,bphi,rhop,D;
  double psi,theta_l,dist_l,dist_surf;
  double Br_l,Bz_l,Bphi_l,Er00_l,Ez00_l,Er0m_l,Ez0m_l,gradBr_l,gradBz_l,curlbr_l,curlbphi_l,curlbz_l;
  double *r_tmp,*z_tmp,*vp_tmp;
  r=r_beg;
  z=z_beg;
  r_tmp=new double[max_step];
  z_tmp=new double[max_step];
  vp_tmp=new double[max_step];
  r0=rlin[0];
  z0=zlin[0];
  dr=rlin[1]-rlin[0];
  dz=zlin[1]-zlin[0];
  Nr=rlin.size();
  Nz=zlin.size();
  Nsurf=theta.size();
  dtheta=theta[1]-theta[0];
  num_cross=0;
  step_count=0;
  lost=false;
  tau=0.;
  for (it=0;it<nt;it++){
    r_orb1[it]=0.;
    z_orb1[it]=0.;
    vp_orb1[it]=0.;
  }
  for (it=0;it<max_step;it++){
    r_tmp[it]=0.;
    z_tmp[it]=0.;
    vp_tmp[it]=0.;
  }
  for (it=0;it<max_step;it++){
    if (isnan(r+z)){
      if (num_cross==1) lost=true;
      break;
    }
    if (it==nsteps*step_count){
      if ((step_count<nt)&&(num_cross==0)){
        r_orb1[step_count]=r;
        z_orb1[step_count]=z;
        vp_orb1[step_count]=vp;
      }
      if (num_cross==0) step_count=step_count+1;
    }
    r_tmp[it]=r;
    z_tmp[it]=z;
    vp_tmp[it]=vp;
    rc=r;
    zc=z;
    vpc=vp;
    for (irk=1;irk<=4;irk++){
      ir=floor((rc-r0)/dr);
      iz=floor((zc-z0)/dz);
      wr=(rc-r0)/dr-ir;
      wz=(zc-z0)/dz-iz;
      if ((ir<0)||(ir>Nr-2)||(iz<0)||(iz>Nz-2)){
        drdtc=nan("");
        dzdtc=nan("");
        dvpdtc=nan("");
      }
      else{
        Br_l=Br[iz*Nz+ir]*(1-wz)*(1-wr)+Br[(iz+1)*Nz+ir]*wz*(1-wr)\
            +Br[iz*Nz+ir+1]*(1-wz)*wr+Br[(iz+1)*Nz+ir+1]*wz*wr;
        Bz_l=Bz[iz*Nz+ir]*(1-wz)*(1-wr)+Bz[(iz+1)*Nz+ir]*wz*(1-wr)\
            +Bz[iz*Nz+ir+1]*(1-wz)*wr+Bz[(iz+1)*Nz+ir+1]*wz*wr;
        Bphi_l=Bphi[iz*Nz+ir]*(1-wz)*(1-wr)+Bphi[(iz+1)*Nz+ir]*wz*(1-wr)\
              +Bphi[iz*Nz+ir+1]*(1-wz)*wr+Bphi[(iz+1)*Nz+ir+1]*wz*wr;
        gradBr_l=gradBr[iz*Nz+ir]*(1-wz)*(1-wr)+gradBr[(iz+1)*Nz+ir]*wz*(1-wr)\
                +gradBr[iz*Nz+ir+1]*(1-wz)*wr+gradBr[(iz+1)*Nz+ir+1]*wz*wr;
        gradBz_l=gradBz[iz*Nz+ir]*(1-wz)*(1-wr)+gradBz[(iz+1)*Nz+ir]*wz*(1-wr)\
                +gradBz[iz*Nz+ir+1]*(1-wz)*wr+gradBz[(iz+1)*Nz+ir+1]*wz*wr;
        curlbr_l=curlbr[iz*Nz+ir]*(1-wz)*(1-wr)+curlbr[(iz+1)*Nz+ir]*wz*(1-wr)\
                +curlbr[iz*Nz+ir+1]*(1-wz)*wr+curlbr[(iz+1)*Nz+ir+1]*wz*wr;
        curlbz_l=curlbz[iz*Nz+ir]*(1-wz)*(1-wr)+curlbz[(iz+1)*Nz+ir]*wz*(1-wr)\
                +curlbz[iz*Nz+ir+1]*(1-wz)*wr+curlbz[(iz+1)*Nz+ir+1]*wz*wr;
        curlbphi_l=curlbphi[iz*Nz+ir]*(1-wz)*(1-wr)+curlbphi[(iz+1)*Nz+ir]*wz*(1-wr)\
                  +curlbphi[iz*Nz+ir+1]*(1-wz)*wr+curlbphi[(iz+1)*Nz+ir+1]*wz*wr;
        Er00_l=Er00[iz*Nz+ir]*(1-wz)*(1-wr)+Er00[(iz+1)*Nz+ir]*wz*(1-wr)\
            +Er00[iz*Nz+ir+1]*(1-wz)*wr+Er00[(iz+1)*Nz+ir+1]*wz*wr;
        Ez00_l=Ez00[iz*Nz+ir]*(1-wz)*(1-wr)+Ez00[(iz+1)*Nz+ir]*wz*(1-wr)\
            +Ez00[iz*Nz+ir+1]*(1-wz)*wr+Ez00[(iz+1)*Nz+ir+1]*wz*wr;
        Er0m_l=Er0m[iz*Nz+ir]*(1-wz)*(1-wr)+Er0m[(iz+1)*Nz+ir]*wz*(1-wr)\
            +Er0m[iz*Nz+ir+1]*(1-wz)*wr+Er0m[(iz+1)*Nz+ir+1]*wz*wr;
        Ez0m_l=Ez0m[iz*Nz+ir]*(1-wz)*(1-wr)+Ez0m[(iz+1)*Nz+ir]*wz*(1-wr)\
            +Ez0m[iz*Nz+ir+1]*(1-wz)*wr+Ez0m[(iz+1)*Nz+ir+1]*wz*wr;
        Bmag=sqrt(Br_l*Br_l+Bz_l*Bz_l+Bphi_l*Bphi_l);
        br=Br_l/Bmag;
        bphi=Bphi_l/Bmag;
        bz=Bz_l/Bmag;
        Er=Er00_l+Er0m_l;
        Ez=Ez00_l+Ez0m_l;
        rhop=mi*vpc/qi/Bmag;
        D=1.+rhop*(br*curlbr_l+bz*curlbz_l+bphi*curlbphi_l);
        dvpdtc=mu*(br*gradBr_l+bz*gradBz_l)+rhop*mu*(curlbr_l*gradBr_l+curlbz_l*gradBz_l)\
              -qi*rhop*(curlbr_l*Er+curlbz_l*Ez)-qi*(br*Er0m_l+bz*Ez0m_l);
        dvpdtc=dvpdtc/(-mi*D);
        drdtc=vpc*br+vpc*rhop*curlbr_l+Bphi_l*(mu*gradBz_l-qi*Ez)/qi/Bmag/Bmag;
        drdtc=drdtc/D;
        dzdtc=vpc*bz+vpc*rhop*curlbz_l-Bphi_l*(mu*gradBr_l-qi*Er)/qi/Bmag/Bmag;
        dzdtc=dzdtc/D;
      }
      switch(irk){
        case 1:
          dvpdte=dvpdtc/6.;
          drdte=drdtc/6.;
          dzdte=dzdtc/6.;
          vpc=vp+dvpdtc*dt_orb/2.;
          rc=r+drdtc*dt_orb/2.;
          zc=z+dzdtc*dt_orb/2.;
          break;
        case 2:
          dvpdte=dvpdte+dvpdtc/3.;
          drdte=drdte+drdtc/3.;
          dzdte=dzdte+dzdtc/3.;
          rc=r+drdtc*dt_orb/2.;
          zc=z+dzdtc*dt_orb/2.;
          break;
        case 3:
          dvpdte=dvpdte+dvpdtc/3.;
          drdte=drdte+drdtc/3.;
          dzdte=dzdte+dzdtc/3.;
          rc=r+drdtc*dt_orb;
          zc=z+dzdtc*dt_orb;
          break;
        case 4:
          dvpdte=dvpdte+dvpdtc/6.;
          drdte=drdte+drdtc/6.;
          dzdte=dzdte+dzdtc/6.;
          vp=vp+dvpdte*dt_orb;
          r=r+drdte*dt_orb;
          z=z+dzdte*dt_orb;
          break;
      }
      if (isnan(rc+zc)||isnan(r+z)){
        if (num_cross==1) lost=true;
        break;
      }
    }//end of rk4 loop
    if (isnan(rc+zc)||isnan(r+z)){
      if (num_cross==1) lost=true;
      break;
    }
    tau=tau+dt_orb;
    ir=floor((r-r0)/dr);
    iz=floor((z-z0)/dz);
    wr=(r-r0)/dr-ir;
    wz=(z-z0)/dz-iz;
    if ((ir<0)||(ir>Nr-2)||(iz<0)||(iz>Nz-2)) break;
    psi=psi2d[iz*Nz+ir]*(1-wz)*(1-wr)+psi2d[(iz+1)*Nz+ir]*wz*(1-wr)\
       +psi2d[iz*Nz+ir+1]*(1-wz)*wr+psi2d[(iz+1)*Nz+ir+1]*wz*wr;
    theta_l=atan2(z-za,r-ra);
    if (theta_l<=theta[0]) theta_l+=8*atan(1.);
    dist_l=sqrt((r-ra)*(r-ra)+(z-za)*(z-za));
    if (theta_l>=theta[Nsurf-1]){
      wtheta=(theta_l-theta[Nsurf-1])/(theta[0]+8*atan(1.)-theta[Nsurf-1]);
      dist_surf=dist[Nsurf-1]*(1-wtheta)+dist[0]*wtheta;
    }
    else{
      itheta=floor((theta_l-theta[0])/dtheta);
      itheta=min(itheta,Nsurf-1);
      while (theta_l<theta[itheta]) itheta=itheta-1;
      while (theta_l>theta[itheta]) itheta=itheta+1;
      wtheta=(theta_l-theta[itheta])/(theta[itheta+1]-theta[itheta]);
      dist_surf=dist[itheta]*(1-wtheta)+dist[itheta+1]*wtheta;
    }
    if((num_cross==0)&&(\
      (psi>(1+cross_psitol)*psi_surf)||\
      (sqrt((r-r_end)*(r-r_end)+(z-z_end)*(z-z_end))<cross_rztol)||\
      (dist_l>dist_surf*(1+cross_disttol))\
     )){
       num_cross=1;
       if (step_count<=nt){
         dt_orb_out=dt_orb*double(nsteps);
       }
       else{
         dt_orb_out=tau/double(nt);
         step_count=nt;
         for (it2=0;it2<nt;it2++){
           t_ind=floor(double(it2)*dt_orb_out/dt_orb);
           wt=double(it2)*dt_orb_out/dt_orb-t_ind;
           if (t_ind==(max_step-1)){
             t_ind=t_ind-1;
             wt=1.0;
           }
           if (abs(r_tmp[t_ind+1])<1E-3) wt=0.0;
           r_orb1[it2]=(1-wt)*r_tmp[t_ind]+wt*r_tmp[t_ind+1];
           z_orb1[it2]=(1-wt)*z_tmp[t_ind]+wt*z_tmp[t_ind+1];
           vp_orb1[it2]=(1-wt)*vp_tmp[t_ind]+wt*vp_tmp[t_ind+1];
         }//for it2
       }//if step_count<=nt
       if (! determine_loss) break;
    }//if cross
    if((num_cross==1)&&\
      (psi<(1-cross_psitol)*psix)&&\
      (z<zx)){
        lost=true;
        break;
      }
    if((num_cross==1)&&(\
      (psi<(1-cross_psitol)*psi_surf)||\
      (sqrt((r-r_beg)*(r-r_beg)+(z-z_beg)*(z-z_beg))<cross_rztol)||\
      (dist_l<dist_surf*(1-cross_disttol))\
     )){
       num_cross=2;
       lost=false;
       break;
     }
  }//end of step loop
  if (step_count>nt) step_count=1;
  delete [] r_tmp;
  delete [] z_tmp;
  delete [] vp_tmp;
  ''',
  'orbit')
  global myEr00,myEz00,myEr0m,myEz0m
  if calc_gyroE: myEr00,myEz00,myEr0m,myEz0m=var.efield(iorb)
  
  dt_orb=dt_xgc/float(nsteps)
  #prepare the axis position
  ra=var.rsurf[0]-var.dist[0]*math.cos(var.theta[0])
  za=var.zsurf[0]-var.dist[0]*math.sin(var.theta[0])

  ix,iy,wx,wy,Bmag=myinterp.TwoD(var.Bmag,r_beg,z_beg)
  if np.isnan(Bmag):
    print('Wrong initial orbit locations: iorb=',iorb,'r=',r_beg,'z=',z_beg)
    exit()
  Bphi=var.Bphi[iy,ix]*(1-wy)*(1-wx) + var.Bphi[iy+1,ix]*wy*(1-wx)\
      +var.Bphi[iy,ix+1]*(1-wy)*wx + var.Bphi[iy+1,ix+1]*wy*wx
  vp=(Pphi-qi*var.psi_surf)/mi/r_beg/Bphi*Bmag

  r_beg_gpu=cp.asarray([r_beg],dtype=cp.float64)
  z_beg_gpu=cp.asarray([z_beg],dtype=cp.float64)
  vp_gpu=cp.asarray([vp],dtype=cp.float64)
  r_orb1_gpu=cp.zeros((nt,),dtype=cp.float64)
  z_orb1_gpu=cp.zeros((nt,),dtype=cp.float64)
  vp_orb1_gpu=cp.zeros((nt,),dtype=cp.float64)
  tau_gpu=cp.asarray([0.],dtype=cp.float64)
  dt_orb_out_gpu=cp.asarray([0.],dtype=cp.float64)
  step_count_gpu=cp.asarray([0],dtype=cp.int64)
  num_cross_gpu=cp.asarray([0],dtype=cp.int64)
  lost_gpu=cp.asarray([False],dtype=cp.bool_)
  Br_gpu=cp.asarray(var.Br,dtype=cp.float64).ravel(order='C')
  Bz_gpu=cp.asarray(var.Bz,dtype=cp.float64).ravel(order='C')
  Bphi_gpu=cp.asarray(var.Bphi,dtype=cp.float64).ravel(order='C')
  gradBr_gpu=cp.asarray(var.gradBr,dtype=cp.float64).ravel(order='C')
  gradBz_gpu=cp.asarray(var.gradBz,dtype=cp.float64).ravel(order='C')
  curlbr_gpu=cp.asarray(var.curlbr,dtype=cp.float64).ravel(order='C')
  curlbz_gpu=cp.asarray(var.curlbz,dtype=cp.float64).ravel(order='C')
  curlbphi_gpu=cp.asarray(var.curlbphi,dtype=cp.float64).ravel(order='C')
  Er00_gpu=cp.asarray(myEr00,dtype=cp.float64).ravel(order='C')
  Ez00_gpu=cp.asarray(myEz00,dtype=cp.float64).ravel(order='C')
  Er0m_gpu=cp.asarray(myEr0m,dtype=cp.float64).ravel(order='C')
  Ez0m_gpu=cp.asarray(myEz0m,dtype=cp.float64).ravel(order='C')
  rlin_gpu=cp.asarray(var.rlin,dtype=cp.float64)
  zlin_gpu=cp.asarray(var.zlin,dtype=cp.float64)
  psi2d_gpu=cp.asarray(var.psi2d,dtype=cp.float64)
  theta_gpu=cp.asarray(var.theta,dtype=cp.float64)
  dist_gpu=cp.asarray(var.dist,dtype=cp.float64)
  psix_gpu=cp.asarray(var.psix,dtype=cp.float64)
  zx_gpu=cp.asarray(var.zx,dtype=cp.float64)

  orbit(r_beg_gpu,z_beg_gpu,qi,mi,mu,max_step,nsteps,nt,Br_gpu,Bz_gpu,Bphi_gpu,gradBr_gpu,gradBz_gpu,\
       curlbr_gpu,curlbz_gpu,curlbphi_gpu,Er00_gpu,Ez00_gpu,Er0m_gpu,Ez0m_gpu,rlin_gpu,zlin_gpu,\
       dt_orb,psi2d_gpu,theta_gpu,ra,za,dist_gpu,var.psi_surf,r_end,z_end,cross_psitol,cross_rztol,\
       cross_disttol,determine_loss,psix_gpu,zx_gpu,\
       num_cross_gpu,step_count_gpu,lost_gpu,vp_gpu,r_orb1_gpu,z_orb1_gpu,vp_orb1_gpu,tau_gpu,dt_orb_out_gpu)
  r_orb1=cp.asnumpy(r_orb1_gpu)
  z_orb1=cp.asnumpy(z_orb1_gpu)
  vp_orb1=cp.asnumpy(vp_orb1_gpu)
  lost=cp.asnumpy(lost_gpu)
  tau=cp.asnumpy(tau_gpu)
  step_count=cp.asnumpy(step_count_gpu)
  num_cross=cp.asnumpy(num_cross_gpu)
  dt_orb_out=cp.asnumpy(dt_orb_out_gpu)
  return lost,tau,dt_orb_out,step_count,r_orb1,z_orb1,vp_orb1
