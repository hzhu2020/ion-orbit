import variables as var
import myinterp
import numpy as np
import math
import os
from parameters import cross_psitol,cross_rztol,cross_disttol,debug,debug_dir,determine_loss,\
     qi,mi,nmu,nPphi,nH,nt,max_step,dt_xgc,nsteps

def tau_orb(calc_gyroE,iorb,r_beg,z_beg,r_end,z_end,mu,Pphi):
  global myEr00,myEz00,myEr0m,myEz0m
  if calc_gyroE: myEr00,myEz00,myEr0m,myEz0m=var.efield(int(iorb/(nPphi*nH)))

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
      if (step_count<nt)and(nsteps==1):
        dt_orb_out=dt_xgc
        r_orb1[step_count]=r
        z_orb1[step_count]=z
        vp_orb1[step_count]=vp
        step_count=step_count+1
      else:
        if it<int(max_step)-1:
          r_tmp[it+1]=r
          z_tmp[it+1]=z
          vp_tmp[it+1]=vp
        step_count=min(step_count,nt-1)
        dt_orb_out=tau/np.float(step_count)
        for it2 in range(step_count+1):
          #t_ind=it2*(it+1)/step_count, so t_ind \in [0,it+1]
          t_ind=math.floor(float(it2)*dt_orb_out/dt_orb)
          wt=float(it2)*dt_orb_out/dt_orb - t_ind
          if t_ind==np.int(max_step)-1: #in case the right point is out of the boundary
            t_ind=t_ind-1
            wt=1.0
          if abs(r_tmp[t_ind+1])<1E-3: wt=0.0 #in case the right point has not been assgined value 
          r_orb1[it2]=(1-wt)*r_tmp[t_ind]+wt*r_tmp[t_ind+1]
          z_orb1[it2]=(1-wt)*z_tmp[t_ind]+wt*z_tmp[t_ind+1]
          vp_orb1[it2]=(1-wt)*vp_tmp[t_ind]+wt*vp_tmp[t_ind+1]
        #end for it2
        step_count=step_count+1
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

def tau_orb_gpu(iorb1,iorb2,r_beg,z_beg,r_end,z_end,mu_arr,Pphi_arr):
  import cupy as cp
  orbit_kernel=cp.RawKernel(r'''
  extern "C" __device__
  double TwoD(double* f2d,double xin,double yin,double dx,double dy,double x0,double y0,int Nx,int Ny)
  {
    int ix,iy;
    double wx,wy,fout;
    ix=floor((xin-x0)/dx);
    wx=(xin-x0)/dx-ix;
    iy=floor((yin-y0)/dy);
    wy=(yin-y0)/dy-iy;
    if ((ix<0)|| (ix>Nx-2)||(iy<0)||(iy>Ny-2)){
      fout=nan("");
    }
    else{
      fout=f2d[iy*Nx+ix]*(1-wy)*(1-wx)+f2d[(iy+1)*Nx+ix]*wy*(1-wx)\
          +f2d[iy*Nx+ix+1]*(1-wy)*wx+f2d[(iy+1)*Nx+ix+1]*wy*wx;
    }
    return fout;
  }
  extern "C" __device__
  void rhs(double qi,double mi,double r,double z,double mu,double vp,double* rhs_l,double* Br,double* Bz,\
       double* Bphi,double* Er00,double* Ez00,double* Er0m,double* Ez0m,double* gradBr,double* gradBz,\
       double* curlbr,double* curlbz,double* curlbphi,double r0,double z0,double dr,\
       double dz,int Nr,int Nz)
  { 
    double Br_l,Bz_l,Bphi_l,br_l,bz_l,bphi_l,Er00_l,Ez00_l,Er0m_l,Ez0m_l,Bmag,Er_l,Ez_l;
    double gradBr_l,gradBz_l,curlbr_l,curlbz_l,curlbphi_l,rhop,D,drdt,dzdt,dvpdt;
    Br_l=TwoD(Br,r,z,dr,dz,r0,z0,Nr,Nz);
    if (isnan(Br_l)){
      rhs_l[0]=nan("");
      rhs_l[1]=nan("");
      rhs_l[2]=nan("");
      return;
    }
    Bz_l=TwoD(Bz,r,z,dr,dz,r0,z0,Nr,Nz);
    Bphi_l=TwoD(Bphi,r,z,dr,dz,r0,z0,Nr,Nz);
    Bmag=sqrt(Br_l*Br_l+Bz_l*Bz_l+Bphi_l*Bphi_l);
    br_l=Br_l/Bmag;
    bz_l=Bz_l/Bmag;
    bphi_l=Bphi_l/Bmag;
    Er00_l=TwoD(Er00,r,z,dr,dz,r0,z0,Nr,Nz);
    Er0m_l=TwoD(Er0m,r,z,dr,dz,r0,z0,Nr,Nz);
    Ez00_l=TwoD(Ez00,r,z,dr,dz,r0,z0,Nr,Nz);
    Ez0m_l=TwoD(Ez0m,r,z,dr,dz,r0,z0,Nr,Nz);
    Er_l=Er00_l+Er0m_l;
    Ez_l=Ez00_l+Ez0m_l;
    gradBr_l=TwoD(gradBr,r,z,dr,dz,r0,z0,Nr,Nz);
    gradBz_l=TwoD(gradBz,r,z,dr,dz,r0,z0,Nr,Nz);
    curlbr_l=TwoD(curlbr,r,z,dr,dz,r0,z0,Nr,Nz);
    curlbz_l=TwoD(curlbz,r,z,dr,dz,r0,z0,Nr,Nz);
    curlbphi_l=TwoD(curlbphi,r,z,dr,dz,r0,z0,Nr,Nz);
    rhop=mi*vp/qi/Bmag;
    D=1.+rhop*(br_l*curlbr_l+bz_l*curlbz_l+bphi_l*curlbphi_l);
    dvpdt=mu*(br_l*gradBr_l+bz_l*gradBz_l)+rhop*mu*(curlbr_l*gradBr_l+curlbz_l*gradBz_l)\
          -qi*rhop*(curlbr_l*Er_l+curlbz_l*Ez_l)-qi*(br_l*Er0m_l+bz_l*Ez0m_l);
    dvpdt=dvpdt/(-mi*D);
    drdt=vp*br_l+vp*rhop*curlbr_l+Bphi_l*(mu*gradBz_l-qi*Ez_l)/qi/Bmag/Bmag;
    drdt=drdt/D;
    dzdt=vp*bz_l+vp*rhop*curlbz_l-Bphi_l*(mu*gradBr_l-qi*Er_l)/qi/Bmag/Bmag;
    dzdt=dzdt/D;
    rhs_l[0]=drdt;
    rhs_l[1]=dzdt;
    rhs_l[2]=dvpdt;
  }
  extern "C" __global__
  void orbit(int mynorb,int nblocks_max,double* r_orb1,double* z_orb1,double* vp_orb1,int* steps_orb,\
    double* dt_orb_out_orb,int* loss_orb,double* tau_orb,double* r_tmp,double* z_tmp,double* vp_tmp,\
    double* r_beg,double* z_beg,double* r_end,double* z_end,double* rlin,double* zlin,int Nr,int Nz,\
    int max_step,int nsteps,double* theta,double psi_surf,double qi,double mi,double mu,double dt_orb,\
    int Nsurf,int nt,double* Bmag,double* Br,double* Bz,double* Bphi,double* Er00,double* Ez00,\
    double* Er0m,double* Ez0m,double* gradBr,double* gradBz,double* curlbr,double* curlbz,double* curlbphi,\
    double* Pphi_arr,double* psi2d,double* dist,double psix,double zx,double ra,double za,double cross_psitol,\
    double cross_rztol,double cross_disttol,bool determine_loss)
  {
    int iorb0,iorb,num_cross,step_count,itheta,t_ind;
    double r,z,vp,r0,z0,dr,dz,rc,zc,vpc,drdtc,dzdtc,dvpdtc,drdte,dzdte,dvpdte,rhs_l[3];
    double theta_l,dtheta,wtheta,wt,tau,Bmag_l,Bphi_l,psi,dist_l,dist_surf;
    bool lost;
    r0=rlin[0];
    z0=zlin[0];
    dr=rlin[1]-rlin[0];
    dz=zlin[1]-zlin[0];
    dtheta=theta[1]-theta[0];
    iorb=blockIdx.x;
    iorb0=iorb;
    while(iorb<mynorb)
    {
      r=r_beg[iorb];
      z=z_beg[iorb];
      for (int it=0;it<nt;it++){
        r_orb1[iorb*nt+it]=0.;
        z_orb1[iorb*nt+it]=0.;
        vp_orb1[iorb*nt+it]=0.;
      }
      for (int it=0;it<max_step;it++){
        r_tmp[iorb0*max_step+it]=0.;
        z_tmp[iorb0*max_step+it]=0.;
        vp_tmp[iorb0*max_step+it]=0.;
      }
      num_cross=0;
      step_count=0;
      lost=false;
      tau=0.;
      Bmag_l=TwoD(Bmag,r,z,dr,dz,r0,z0,Nr,Nz);
      Bphi_l=TwoD(Bphi,r,z,dr,dz,r0,z0,Nr,Nz);
      vp=(Pphi_arr[iorb]-qi*psi_surf)/mi/r/Bphi_l*Bmag_l;
      for (int it=0;it<max_step;it++){
        if (isnan(r+z)){
         if (num_cross==1) lost=true;
         break;
        }
        if (it==nsteps*step_count){
          if ((step_count<nt)&&(num_cross==0)){
          r_orb1[iorb*nt+step_count]=r;
          z_orb1[iorb*nt+step_count]=z;
          vp_orb1[iorb*nt+step_count]=vp;
          }
          if (num_cross==0) step_count=step_count+1;
        }
        r_tmp[iorb0*max_step+it]=r;
        z_tmp[iorb0*max_step+it]=z;
        vp_tmp[iorb0*max_step+it]=vp;
        vpc=vp;
        //RK4 1st step
        rhs(qi,mi,r,z,mu,vp,rhs_l,Br,Bz,Bphi,Er00,Ez00,Er0m,Ez0m,gradBr,gradBz,\
            curlbr,curlbz,curlbphi,r0,z0,dr,dz,Nr,Nz);
        drdtc=rhs_l[0];dzdtc=rhs_l[1];dvpdtc=rhs_l[2];
        dvpdte=dvpdtc/6.;
        drdte=drdtc/6.;
        dzdte=dzdtc/6.;
        vpc=vp+dvpdtc*dt_orb/2.;
        rc=r+drdtc*dt_orb/2.;
        zc=z+dzdtc*dt_orb/2.;
        if (isnan(rc+zc)){
          if (num_cross==1) lost=true;
          break;
        }
        //RK4 2nd step
        rhs(qi,mi,rc,zc,mu,vpc,rhs_l,Br,Bz,Bphi,Er00,Ez00,Er0m,Ez0m,gradBr,gradBz,\
            curlbr,curlbz,curlbphi,r0,z0,dr,dz,Nr,Nz);
        drdtc=rhs_l[0];dzdtc=rhs_l[1];dvpdtc=rhs_l[2];
        dvpdte=dvpdte+dvpdtc/3.;
        drdte=drdte+drdtc/3.;
        dzdte=dzdte+dzdtc/3.;
        rc=r+drdtc*dt_orb/2.;
        zc=z+dzdtc*dt_orb/2.;
        if (isnan(rc+zc)){
          if (num_cross==1) lost=true;
          break;
        }
        //RK4 3nd step
        rhs(qi,mi,rc,zc,mu,vpc,rhs_l,Br,Bz,Bphi,Er00,Ez00,Er0m,Ez0m,gradBr,gradBz,\
            curlbr,curlbz,curlbphi,r0,z0,dr,dz,Nr,Nz);
        drdtc=rhs_l[0];dzdtc=rhs_l[1];dvpdtc=rhs_l[2];
        dvpdte=dvpdte+dvpdtc/3.;
        drdte=drdte+drdtc/3.;
        dzdte=dzdte+dzdtc/3.;
        rc=r+drdtc*dt_orb;
        zc=z+dzdtc*dt_orb;
        if (isnan(rc+zc)){ 
          if (num_cross==1) lost=true;
          break;
        }
        //Rk4 4th step
        rhs(qi,mi,rc,zc,mu,vpc,rhs_l,Br,Bz,Bphi,Er00,Ez00,Er0m,Ez0m,gradBr,gradBz,\
            curlbr,curlbz,curlbphi,r0,z0,dr,dz,Nr,Nz);
        drdtc=rhs_l[0];dzdtc=rhs_l[1];dvpdtc=rhs_l[2];
        dvpdte=dvpdte+dvpdtc/6.;
        drdte=drdte+drdtc/6.;
        dzdte=dzdte+dzdtc/6.;
        vp=vp+dvpdte*dt_orb;
        r=r+drdte*dt_orb;
        z=z+dzdte*dt_orb;
        if (isnan(r+z)){ 
          if (num_cross==1) lost=true;
          break;
        } 
        tau=tau+dt_orb;
        psi=TwoD(psi2d,r,z,dr,dz,r0,z0,Nr,Nz);
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
          while (theta_l>theta[itheta+1]) itheta=itheta+1;
          wtheta=(theta_l-theta[itheta])/(theta[itheta+1]-theta[itheta]);
          dist_surf=dist[itheta]*(1-wtheta)+dist[itheta+1]*wtheta;
        }
        if((num_cross==0)&&(\
          (psi>(1+cross_psitol)*psi_surf)||\
          (sqrt((r-r_end[iorb])*(r-r_end[iorb])+(z-z_end[iorb])*(z-z_end[iorb]))<cross_rztol)||\
          (dist_l>dist_surf*(1+cross_disttol))\
         )){
           num_cross=1;
           if ((step_count<nt)&&(nsteps==1)){
             dt_orb_out_orb[iorb]=dt_orb*double(nsteps);
             r_orb1[iorb*nt+step_count]=r;
             z_orb1[iorb*nt+step_count]=z;
             vp_orb1[iorb*nt+step_count]=vp;
             step_count=step_count+1;
           }
           else{
             if (it<max_step-1){
               r_tmp[iorb0*max_step+it+1]=r;
               z_tmp[iorb0*max_step+it+1]=z;
               vp_tmp[iorb0*max_step+it+1]=vp;
             }
             step_count=min(step_count,nt-1);
             dt_orb_out_orb[iorb]=tau/double(step_count);
             for (int it2=0;it2<step_count+1;it2++){
               t_ind=floor(double(it2)*dt_orb_out_orb[iorb]/dt_orb);
               wt=double(it2)*dt_orb_out_orb[iorb]/dt_orb-t_ind;
               if (t_ind==(max_step-1)){
                 t_ind=t_ind-1;
                 wt=1.0;
               }
               if (abs(r_tmp[iorb0*max_step+t_ind+1])<1E-3) wt=0.0;
               r_orb1[iorb*nt+it2]=(1-wt)*r_tmp[iorb0*max_step+t_ind]+wt*r_tmp[iorb0*max_step+t_ind+1];
               z_orb1[iorb*nt+it2]=(1-wt)*z_tmp[iorb0*max_step+t_ind]+wt*z_tmp[iorb0*max_step+t_ind+1];
               vp_orb1[iorb*nt+it2]=(1-wt)*vp_tmp[iorb0*max_step+t_ind]+wt*vp_tmp[iorb0*max_step+t_ind+1];
             }//for it2
             step_count=step_count+1;
           }//if step_count<nt
           if (! determine_loss) break;
        }//if cross
       if((num_cross==1)&&(psi<(1-cross_psitol)*psix)&&(z<zx)){
          lost=true;
          break;
         }
       if((num_cross==1)&&(\
         (psi<(1-cross_psitol)*psi_surf)||\
         (sqrt((r-r_beg[iorb])*(r-r_beg[iorb])+(z-z_beg[iorb])*(z-z_beg[iorb]))<cross_rztol)||\
         (dist_l<dist_surf*(1-cross_disttol))\
       )){
         num_cross=2;
         lost=false;
         break;
      }
      }//end for it
      if (step_count>nt) step_count=1;
      steps_orb[iorb]=step_count;
      tau_orb[iorb]=tau;
      loss_orb[iorb]=int(lost);
      iorb=iorb+nblocks_max;
    }
  }
  ''','orbit')
  from parameters import gyro_E,Nz,Nr
  r_beg=r_beg.ravel(order='C')
  z_beg=z_beg.ravel(order='C')
  r_end=r_end.ravel(order='C')
  z_end=z_end.ravel(order='C')
  nblocks_max=4096
  mynorb=iorb2-iorb1+1
  r_orb_gpu=cp.zeros((mynorb*nt,),dtype=cp.float64,order='C')
  z_orb_gpu=cp.zeros((mynorb*nt,),dtype=cp.float64,order='C')
  vp_orb_gpu=cp.zeros((mynorb*nt,),dtype=cp.float64,order='C')
  steps_orb_gpu=cp.zeros((mynorb,),dtype=cp.int32)
  dt_orb_out_orb_gpu=cp.zeros((mynorb,),dtype=cp.float64)
  loss_orb_gpu=cp.zeros((mynorb,),dtype=cp.int32)
  tau_orb_gpu=cp.zeros((mynorb,),dtype=cp.float64)
  dt_orb=dt_xgc/float(nsteps)
  rlin_gpu=cp.asarray(var.rlin,dtype=cp.float64)
  zlin_gpu=cp.asarray(var.zlin,dtype=cp.float64)
  psi2d_gpu=cp.asarray(var.psi2d,dtype=cp.float64).ravel(order='C')
  theta_gpu=cp.asarray(var.theta,dtype=cp.float64)
  dist_gpu=cp.asarray(var.dist,dtype=cp.float64)
  Bmag_gpu=cp.asarray(var.Bmag,dtype=cp.float64).ravel(order='C')
  Br_gpu=cp.asarray(var.Br,dtype=cp.float64).ravel(order='C')
  Bz_gpu=cp.asarray(var.Bz,dtype=cp.float64).ravel(order='C')
  Bphi_gpu=cp.asarray(var.Bphi,dtype=cp.float64).ravel(order='C')
  gradBr_gpu=cp.asarray(var.gradBr,dtype=cp.float64).ravel(order='C')
  gradBz_gpu=cp.asarray(var.gradBz,dtype=cp.float64).ravel(order='C')
  curlbr_gpu=cp.asarray(var.curlbr,dtype=cp.float64).ravel(order='C')
  curlbz_gpu=cp.asarray(var.curlbz,dtype=cp.float64).ravel(order='C')
  curlbphi_gpu=cp.asarray(var.curlbphi,dtype=cp.float64).ravel(order='C')
  Nsurf=np.size(var.theta);
  ra=var.rsurf[0]-var.dist[0]*math.cos(var.theta[0])
  za=var.zsurf[0]-var.dist[0]*math.sin(var.theta[0])
  imu1=int(iorb1/(nPphi*nH))
  imu2=int(iorb2/(nPphi*nH))
  for imu in range(imu1,imu2+1):
    iorb1_l=max(imu*nPphi*nH,iorb1)
    iorb2_l=min((imu+1)*nPphi*nH-1,iorb2)
    mynorb_l=iorb2_l-iorb1_l+1
    idx1=iorb1_l-iorb1
    idx2=iorb2_l+1-iorb1
    nblocks=min(nblocks_max,mynorb_l)
    myEr00,myEz00,myEr0m,myEz0m=var.efield(imu)
    Er00_gpu=cp.asarray(myEr00,dtype=cp.float64).ravel(order='C')
    Ez00_gpu=cp.asarray(myEz00,dtype=cp.float64).ravel(order='C')
    Er0m_gpu=cp.asarray(myEr0m,dtype=cp.float64).ravel(order='C')
    Ez0m_gpu=cp.asarray(myEz0m,dtype=cp.float64).ravel(order='C')
    Pphi_arr_gpu=cp.zeros((mynorb_l,),dtype=cp.float64)
    for iorb_l in range(iorb1_l,iorb2_l+1):
      iPphi=int((iorb_l-imu*nPphi*nH)/nH);
      Pphi_arr_gpu[iorb_l-iorb1_l]=Pphi_arr[iPphi]
    r_beg_gpu=cp.asarray(r_beg[iorb1_l:iorb2_l+1],dtype=cp.float64)
    z_beg_gpu=cp.asarray(z_beg[iorb1_l:iorb2_l+1],dtype=cp.float64)
    r_end_gpu=cp.asarray(r_end[iorb1_l:iorb2_l+1],dtype=cp.float64)
    z_end_gpu=cp.asarray(z_end[iorb1_l:iorb2_l+1],dtype=cp.float64)
    r_tmp_gpu=cp.zeros((int(nblocks*max_step),),dtype=cp.float64,order='C')
    z_tmp_gpu=cp.zeros((int(nblocks*max_step),),dtype=cp.float64,order='C')
    vp_tmp_gpu=cp.zeros((int(nblocks*max_step),),dtype=cp.float64,order='C')
    orbit_kernel((nblocks,),(1,),(int(mynorb_l),int(nblocks_max),r_orb_gpu[idx1*nt:idx2*nt],\
            z_orb_gpu[idx1*nt:idx2*nt],vp_orb_gpu[idx1*nt:idx2*nt],steps_orb_gpu[idx1:idx2],\
            dt_orb_out_orb_gpu[idx1:idx2],loss_orb_gpu[idx1:idx2],tau_orb_gpu[idx1:idx2],\
            r_tmp_gpu,z_tmp_gpu,vp_tmp_gpu,r_beg_gpu,z_beg_gpu,r_end_gpu,z_end_gpu,rlin_gpu,zlin_gpu,\
            int(Nr),int(Nz),int(max_step),int(nsteps),theta_gpu,var.psi_surf,qi,mi,mu_arr[imu],dt_orb,\
            int(Nsurf),int(nt),Bmag_gpu,Br_gpu,Bz_gpu,Bphi_gpu,Er00_gpu,Ez00_gpu,Er0m_gpu,Ez0m_gpu,\
            gradBr_gpu,gradBz_gpu,curlbr_gpu,curlbz_gpu,curlbphi_gpu,Pphi_arr_gpu,psi2d_gpu,dist_gpu,\
            float(var.psix),float(var.zx),ra,za,cross_psitol,cross_rztol,cross_disttol,determine_loss))
    #need to wait for GPU to finish before launching another kernel
    cp.cuda.Stream.null.synchronize()
 
  r_orb=cp.asnumpy(r_orb_gpu).reshape((mynorb,nt),order='C')
  z_orb=cp.asnumpy(z_orb_gpu).reshape((mynorb,nt),order='C')
  vp_orb=cp.asnumpy(vp_orb_gpu).reshape((mynorb,nt),order='C')
  steps_orb=cp.asnumpy(steps_orb_gpu)
  dt_orb_out_orb=cp.asnumpy(dt_orb_out_orb_gpu)
  loss_orb=cp.asnumpy(loss_orb_gpu)
  tau_orb=cp.asnumpy(tau_orb_gpu)
  #fot correct MPI communication if writing to orbit.txt
  steps_orb=np.array(steps_orb,dtype=int)
  loss_orb=np.array(loss_orb,dtype=int)
  del r_orb_gpu,z_orb_gpu,vp_orb_gpu,steps_orb_gpu,dt_orb_out_orb_gpu,loss_orb_gpu,tau_orb_gpu,\
      rlin_gpu,zlin_gpu,psi2d_gpu,theta_gpu,dist_gpu,Bmag_gpu,Br_gpu,Bz_gpu,Bphi_gpu,gradBr_gpu,\
      gradBz_gpu,curlbr_gpu,curlbz_gpu,curlbphi_gpu,Er00_gpu,Ez00_gpu,Er0m_gpu,Ez0m_gpu,Pphi_arr_gpu,\
      r_beg_gpu,z_beg_gpu,r_end_gpu,z_end_gpu,r_tmp_gpu,z_tmp_gpu,vp_tmp_gpu
  return loss_orb,tau_orb,dt_orb_out_orb,steps_orb,r_orb,z_orb,vp_orb
