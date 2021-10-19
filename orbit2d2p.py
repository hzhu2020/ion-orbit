import variables as var
import numpy as np
import math
from parameters import cross_psitol,cross_rztol,cross_disttol,debug,debug_dir,\
     qi,mi,nmu,nPphi,nH,nt,max_step,dt_xgc,nsteps

def calc_orb_gpu(iorb1,iorb2,r_beg,z_beg,r_end,z_end,mu_arr,Pphi_arr,bad_input):
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
    double cross_rztol,double cross_disttol,int* bad)
  {
    int iorb0,iorb,num_cross,step_count,itheta,t_ind,it_count[2],nsteps_local,iloop;
    double r,z,vp,r0,z0,dr,dz,rc,zc,vpc,drdtc,dzdtc,dvpdtc,drdte,dzdte,dvpdte,rhs_l[3];
    double theta_l,dtheta,wtheta,wt,tau[2],Bmag_l,Bphi_l,psi,dist_l,dist_surf,tmp;
    double dt_orb0,z_mid,r_old[2],z_old[2],vp_old[2];
    bool lost,unfinished;
    r0=rlin[0];
    z0=zlin[0];
    dr=rlin[1]-rlin[0];
    dz=zlin[1]-zlin[0];
    dtheta=theta[1]-theta[0];
    iorb=blockIdx.x;
    iorb0=iorb;
    dt_orb0=dt_orb;
    z_mid=(z_beg[iorb]+z_end[iorb])/2.0;
    while(iorb<mynorb)
    {
      for (int it=0;it<nt;it++){
        r_orb1[iorb*nt+it]=0.;
        z_orb1[iorb*nt+it]=0.;
        vp_orb1[iorb*nt+it]=0.;
      }
      for (int it=0;it<2*max_step;it++){
        r_tmp[iorb0*2*max_step+it]=0.;
        z_tmp[iorb0*2*max_step+it]=0.;
        vp_tmp[iorb0*2*max_step+it]=0.;
      }
      num_cross=0;
      it_count[0]=0;
      it_count[1]=0;
      lost=false;
      tau[0]=0.;
      tau[1]=0.;
      unfinished=false;
      if (bad[iorb]==0){
        iorb=iorb+nblocks_max;
        continue;
      }
      for (int it=0;it<max_step*nsteps;it++){
        if (it==nsteps*max_step-1) unfinished=true;
        for (iloop=0;iloop<2;iloop++){
        if (it==0){
          if (iloop==0){
            r_old[iloop]=r_beg[iorb];
            z_old[iloop]=z_beg[iorb];
          }else{
            r_old[iloop]=r_end[iorb];
            z_old[iloop]=z_end[iorb];
          }
          Bmag_l=TwoD(Bmag,r_old[iloop],z_old[iloop],dr,dz,r0,z0,Nr,Nz);
          Bphi_l=TwoD(Bphi,r_old[iloop],z_old[iloop],dr,dz,r0,z0,Nr,Nz);
          vp_old[iloop]=(Pphi_arr[iorb]-qi*psi_surf)/mi/r_old[iloop]/Bphi_l*Bmag_l;
        }
        r=r_old[iloop];
        z=z_old[iloop];
        vp=vp_old[iloop];
        if (isnan(r+z)) break;
        if (it%nsteps==0){
          r_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=r;
          z_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=z;
          vp_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=vp;
          it_count[iloop]=it_count[iloop]+1;
        }
        if (iloop==1) dt_orb=-dt_orb;
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
        if (isnan(rc+zc)) break;
        //RK4 2nd step
        rhs(qi,mi,rc,zc,mu,vpc,rhs_l,Br,Bz,Bphi,Er00,Ez00,Er0m,Ez0m,gradBr,gradBz,\
            curlbr,curlbz,curlbphi,r0,z0,dr,dz,Nr,Nz);
        drdtc=rhs_l[0];dzdtc=rhs_l[1];dvpdtc=rhs_l[2];
        dvpdte=dvpdte+dvpdtc/3.;
        drdte=drdte+drdtc/3.;
        dzdte=dzdte+dzdtc/3.;
        rc=r+drdtc*dt_orb/2.;
        zc=z+dzdtc*dt_orb/2.;
        if (isnan(rc+zc)) break;
        //RK4 3nd step
        rhs(qi,mi,rc,zc,mu,vpc,rhs_l,Br,Bz,Bphi,Er00,Ez00,Er0m,Ez0m,gradBr,gradBz,\
            curlbr,curlbz,curlbphi,r0,z0,dr,dz,Nr,Nz);
        drdtc=rhs_l[0];dzdtc=rhs_l[1];dvpdtc=rhs_l[2];
        dvpdte=dvpdte+dvpdtc/3.;
        drdte=drdte+drdtc/3.;
        dzdte=dzdte+dzdtc/3.;
        rc=r+drdtc*dt_orb;
        zc=z+dzdtc*dt_orb;
        if (isnan(rc+zc)) break;
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
        if (isnan(r+z)) break;
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
        if((psi>(1+cross_psitol)*psi_surf)||(dist_l>dist_surf*(1+cross_disttol))) num_cross=1;
 
        if (iloop==1) dt_orb=-dt_orb;
        tau[iloop]=tau[iloop]+dt_orb;
        r_old[iloop]=r;
        z_old[iloop]=z;
        vp_old[iloop]=vp;
       }//end for 1st iloop
       if (num_cross==1){
         step_count=0;
         dt_orb_out_orb[iorb]=0.;
         break;
       }
       if ((num_cross==0)&&(sqrt((r_old[1]-r_old[0])*(r_old[1]-r_old[0])+\
              (z_old[1]-z_old[0])*(z_old[1]-z_old[0]))<cross_rztol)){
        bad[iorb]=0;
        for(iloop=0;iloop<2;iloop++){
          r=r_old[iloop];
          z=z_old[iloop];
          vp=vp_old[iloop];
          if (it_count[iloop]<max_step){
            nsteps_local=(it+1)%nsteps;
            if (nsteps_local==0){
              r_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=r;
              z_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=z;
              vp_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=vp;
            }else{
              r_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=r_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]-1]\
                +(r-r_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]-1])*double(nsteps)/double(nsteps_local);
              z_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=z_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]-1]\
                +(z-z_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]-1])*double(nsteps)/double(nsteps_local);
              vp_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]]=vp_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]-1]\
                +(vp-vp_tmp[(2*iorb0+iloop)*max_step+it_count[iloop]-1])*double(nsteps)/double(nsteps_local);
            }
          }
        }//end for 2nd iloop
        step_count=min(it_count[0]+it_count[1],nt-1);
        dt_orb_out_orb[iorb]=(tau[0]+tau[1])/double(step_count);
        for (int it2=0;it2<step_count+1;it2++){
          tmp=double(it2)*dt_orb_out_orb[iorb];
          if (tmp<tau[0]){
            iloop=0;
            t_ind=floor(tmp/dt_orb/double(nsteps));
            wt=tmp/dt_orb/double(nsteps)-t_ind;
          }else{
            iloop=1;
            tmp=tau[0]+tau[1]-tmp;
            t_ind=floor(tmp/dt_orb/double(nsteps));
            wt=tmp/dt_orb/double(nsteps)-t_ind;
          }
          if (t_ind==(max_step-1)){
            t_ind=t_ind-1;
            wt=1.0;
          }
          if (abs(r_tmp[(2*iorb0+iloop)*max_step+t_ind+1])<1E-3) wt=0.0;
          r_orb1[iorb*nt+it2]=(1-wt)*r_tmp[(2*iorb0+iloop)*max_step+t_ind]\
                                 +wt*r_tmp[(2*iorb0+iloop)*max_step+t_ind+1];
          z_orb1[iorb*nt+it2]=(1-wt)*z_tmp[(2*iorb0+iloop)*max_step+t_ind]\
                                 +wt*z_tmp[(2*iorb0+iloop)*max_step+t_ind+1];
          vp_orb1[iorb*nt+it2]=(1-wt)*vp_tmp[(2*iorb0+iloop)*max_step+t_ind]+\
                                 wt*vp_tmp[(2*iorb0+iloop)*max_step+t_ind+1];
        }//for it2
        step_count=step_count+1; 
        break;
        }//end if close enough
     }//end for it
     if (unfinished){
        dt_orb=dt_orb*2;
      }else{
        steps_orb[iorb]=step_count;
        tau_orb[iorb]=tau[0]+tau[1];
        loss_orb[iorb]=int(lost);
        dt_orb=dt_orb0;//reset dt_orb and go to the next orbit
        iorb=iorb+nblocks_max;
      }
    }
  }
  ''','orbit')
  from parameters import gyro_E,Nz,Nr
  mempool = cp.get_default_memory_pool()
  mem_limit=mempool.get_limit()
  mempool.set_limit(fraction=1.)#use mem_limit as a guide, but do not actually place a cap
  r_beg=r_beg.ravel(order='C')
  z_beg=z_beg.ravel(order='C')
  r_end=r_end.ravel(order='C')
  z_end=z_end.ravel(order='C')
  mynorb=iorb2-iorb1+1
  r_orb_gpu=cp.zeros((mynorb*nt,),dtype=cp.float64,order='C')
  z_orb_gpu=cp.zeros((mynorb*nt,),dtype=cp.float64,order='C')
  vp_orb_gpu=cp.zeros((mynorb*nt,),dtype=cp.float64,order='C')
  steps_orb_gpu=cp.zeros((mynorb,),dtype=cp.int32)
  dt_orb_out_orb_gpu=cp.zeros((mynorb,),dtype=cp.float64)
  loss_orb_gpu=cp.zeros((mynorb,),dtype=cp.int32)
  tau_orb_gpu=cp.zeros((mynorb,),dtype=cp.float64)
  bad_gpu=cp.asarray(bad_input,dtype=cp.int32)

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
    myEr00,myEz00,myEr0m,myEz0m=var.efield(imu-var.imu1)
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
    nblocks_max=4096
    mem_used=mempool.used_bytes()
    while (nblocks_max*2*max_step*8*3)>=(mem_limit-mem_used): nblocks_max=int(nblocks_max/2)
    nblocks=min(nblocks_max,mynorb_l)
    r_tmp_gpu=cp.zeros((int(2*nblocks*max_step),),dtype=cp.float64,order='C')
    z_tmp_gpu=cp.zeros((int(2*nblocks*max_step),),dtype=cp.float64,order='C')
    vp_tmp_gpu=cp.zeros((int(2*nblocks*max_step),),dtype=cp.float64,order='C')
    orbit_kernel((nblocks,),(1,),(int(mynorb_l),int(nblocks_max),r_orb_gpu[idx1*nt:idx2*nt],\
            z_orb_gpu[idx1*nt:idx2*nt],vp_orb_gpu[idx1*nt:idx2*nt],steps_orb_gpu[idx1:idx2],\
            dt_orb_out_orb_gpu[idx1:idx2],loss_orb_gpu[idx1:idx2],tau_orb_gpu[idx1:idx2],\
            r_tmp_gpu,z_tmp_gpu,vp_tmp_gpu,r_beg_gpu,z_beg_gpu,r_end_gpu,z_end_gpu,rlin_gpu,zlin_gpu,\
            int(Nr),int(Nz),int(max_step),int(nsteps),theta_gpu,var.psi_surf,qi,mi,mu_arr[imu],dt_orb,\
            int(Nsurf),int(nt),Bmag_gpu,Br_gpu,Bz_gpu,Bphi_gpu,Er00_gpu,Ez00_gpu,Er0m_gpu,Ez0m_gpu,\
            gradBr_gpu,gradBz_gpu,curlbr_gpu,curlbz_gpu,curlbphi_gpu,Pphi_arr_gpu,psi2d_gpu,dist_gpu,\
            float(var.psix),float(var.zx),ra,za,cross_psitol,cross_rztol,cross_disttol,
            bad_gpu[idx1:idx2]))
    #need to wait for GPU to finish before launching another kernel
    cp.cuda.Stream.null.synchronize()
    del Er00_gpu,Ez00_gpu,Er0m_gpu,Ez0m_gpu,Pphi_arr_gpu,r_beg_gpu,z_beg_gpu,r_end_gpu,z_end_gpu,\
        r_tmp_gpu,z_tmp_gpu,vp_tmp_gpu

  global r_orb,z_orb,vp_orb,steps_orb,dt_orb_out_orb,loss_orb,tau_orb,bad
  r_orb=cp.asnumpy(r_orb_gpu).reshape((mynorb,nt),order='C')
  z_orb=cp.asnumpy(z_orb_gpu).reshape((mynorb,nt),order='C')
  vp_orb=cp.asnumpy(vp_orb_gpu).reshape((mynorb,nt),order='C')
  steps_orb=cp.asnumpy(steps_orb_gpu)
  dt_orb_out_orb=cp.asnumpy(dt_orb_out_orb_gpu)
  loss_orb=cp.asnumpy(loss_orb_gpu)
  tau_orb=cp.asnumpy(tau_orb_gpu)
  bad=cp.asnumpy(bad_gpu)
  #fot correct MPI communication if writing to orbit.txt
  steps_orb=np.array(steps_orb,dtype=int)
  loss_orb=np.array(loss_orb,dtype=int)
  del r_orb_gpu,z_orb_gpu,vp_orb_gpu,steps_orb_gpu,dt_orb_out_orb_gpu,loss_orb_gpu,tau_orb_gpu,bad_gpu
  del rlin_gpu,zlin_gpu,psi2d_gpu,theta_gpu,dist_gpu,Bmag_gpu,Br_gpu,Bz_gpu,Bphi_gpu,gradBr_gpu,\
      gradBz_gpu,curlbr_gpu,curlbz_gpu,curlbphi_gpu
  return
