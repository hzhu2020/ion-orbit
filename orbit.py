import variables as var
import myinterp
import numpy as np
import math
import os
from parameters import max_step,cross_psitol,cross_rztol,cross_disttol,debug,debug_dir

def tau_orb(iorb,qi,mi,x,y,z,r_end,z_end,mu,Pphi,dt_orb,dt_xgc,nt):
  #prepare the axis position
  ra=var.rLCFS[0]-var.dist[0]*math.cos(var.theta[0])
  za=var.zLCFS[0]-var.dist[0]*math.sin(var.theta[0])

  r=np.sqrt(x**2+y**2)
  Bmag=myinterp.TwoD(var.Bmag,r,z)
  Bphi=myinterp.TwoD(var.Bphi,r,z)
  vp=(Pphi-qi*var.psix)/mi/r/Bphi*Bmag
  H,dHdr,dHdz=var.H2d(mu,Pphi,mi,qi)
  H0=myinterp.TwoD(H,r,z)

  r_orb1=np.zeros((nt,),dtype=float)
  z_orb1=np.zeros((nt,),dtype=float)
  #phi_orb1=np.zeros((nt,),dtype=float)
  vp_orb1=np.zeros((nt,),dtype=float)

  r_tmp=np.zeros((np.int(max_step),),dtype=float)
  z_tmp=np.zeros((np.int(max_step),),dtype=float)
  vp_tmp=np.zeros((np.int(max_step),),dtype=float)

  tau=0
  step_count=0
  #step_flag=True
  if debug:
    debug_count=0
    if not(os.path.isdir(debug_dir)): os.mkdir(debug_dir)
    output=open(debug_dir+'/'+str(iorb)+'.txt','w')
    output.write('%8d\n'%0)#a placeholder for debug_count
 
  for it in range(np.int(max_step)):
    phi=math.atan2(y,x)
    r=np.sqrt(x**2+y**2)
    if np.isnan(r): break
    if math.floor(tau/dt_xgc)==step_count:
      #make a correction of position based on H deviation
      #did not work well. Turned it off until further update
      if False:
        psin=myinterp.TwoD(var.psi2d,r,z)/var.psix
        fac=0.01*math.exp(-5E3*(psin-0.98)**2)
        Hc=myinterp.TwoD(H,r,z)
        dHdrc=myinterp.TwoD(dHdr,r,z)
        dHdzc=myinterp.TwoD(dHdz,r,z)
        rtmp=r+fac*(H0-Hc)*dHdrc/(dHdrc**2+dHdzc**2)
        x=x*rtmp/r
        y=y*rtmp/r
        z=z+fac*(H0-Hc)*dHdzc/(dHdrc**2+dHdzc**2)

      if step_count<nt:
        r_orb1[step_count]=r
        z_orb1[step_count]=z
        vp_orb1[step_count]=vp
      step_count=step_count+1
      if debug:
        debug_count=debug_count+1
        output.write('%19.10E %19.10E\n'%(r,z))
      
    #if step_flag and (math.floor(tau/dt_xgc)==step_count):
    #  r_orb1[step_count]=r
    #  z_orb1[step_count]=z
    #  #phi_orb1[step_count]=phi
    #  vp_orb1[step_count]=vp
    #  step_count=step_count+1
    #  if step_count==nt: step_flag=False#only record nt steps of orbit information
    
    r_tmp[it]=r
    z_tmp[it]=z
    vp_tmp[it]=vp
    #RK4 1st step
    dxdtc,dvpdtc=rhs(qi,mi,r,phi,z,mu,vp)
    dvpdte=dvpdtc/6
    dxdte=dxdtc/6
    vpc=vp+dvpdtc*dt_orb/2
    xc=x+dxdtc[0]*dt_orb/2
    yc=y+dxdtc[1]*dt_orb/2
    zc=z+dxdtc[2]*dt_orb/2
    phic=math.atan2(yc,xc)
    rc=np.sqrt(xc**2+yc**2)
    if np.isnan(rc): break
    #RK4 2nd step
    dxdtc,dvpdtc=rhs(qi,mi,rc,phic,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/3
    dxdte=dxdte+dxdtc/3
    xc=x+dxdtc[0]*dt_orb/2
    yc=y+dxdtc[1]*dt_orb/2
    zc=z+dxdtc[2]*dt_orb/2
    phic=math.atan2(yc,xc)
    rc=np.sqrt(xc**2+yc**2)
    if np.isnan(rc): break
    #RK4 3nd step
    dxdtc,dvpdtc=rhs(qi,mi,rc,phic,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/3
    dxdte=dxdte+dxdtc/3
    xc=x+dxdtc[0]*dt_orb
    yc=y+dxdtc[1]*dt_orb
    zc=z+dxdtc[2]*dt_orb
    phic=math.atan2(yc,xc)
    rc=np.sqrt(xc**2+yc**2)
    if np.isnan(rc): break
    #Rk4 4th step
    dxdtc,dvpdtc=rhs(qi,mi,rc,phic,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/6
    dxdte=dxdte+dxdtc/6
    vp=vp+dvpdte*dt_orb
    x=x+dxdte[0]*dt_orb
    y=y+dxdte[1]*dt_orb
    z=z+dxdte[2]*dt_orb

    tau=tau+dt_orb
    #check if the orbit has crossed the LCFS
    r=np.sqrt(x**2+y**2)
    if np.isnan(r): break
    psi=myinterp.TwoD(var.psi2d,r,z)
    theta=math.atan2(z-za,r-ra)
    if theta<=var.theta[0]: theta=theta+2*np.pi
    dist=np.sqrt((r-ra)**2+(z-za)**2)
    dist_LCFS=myinterp.OneD_NL(var.theta,var.dist,theta)
    if (psi>(1+cross_psitol)*var.psix) \
       or (np.sqrt((r-r_end)**2+(z-z_end)**2)<cross_rztol)\
       or (dist>dist_LCFS*(1+cross_disttol)):
      break
    #end of the time loop

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

def mycross(a,b):
  out=np.zeros((3,),dtype=float)
  out[0]=a[1]*b[2]-a[2]*b[1]
  out[1]=a[2]*b[0]-a[0]*b[2]
  out[2]=a[0]*b[1]-a[1]*b[0]
  return out

def rhs(qi,mi,r,phi,z,mu,vp):
    #B
    Br=myinterp.TwoD(var.Br,r,z)
    Bphi=myinterp.TwoD(var.Bphi,r,z)
    Bz=myinterp.TwoD(var.Bz,r,z)
    Bx=Br*math.cos(phi)-Bphi*math.sin(phi)
    By=Br*math.sin(phi)+Bphi*math.cos(phi)
    B=np.array([Bx,By,Bz])
    Bmag=np.sqrt(sum(B**2));
    b=B/Bmag
    #E
    Er=myinterp.TwoD(var.Er,r,z)
    Ephi=myinterp.TwoD(var.Ephi,r,z)
    Ez=myinterp.TwoD(var.Ez,r,z)
    Ex=Er*math.cos(phi)-Ephi*math.sin(phi)
    Ey=Er*math.sin(phi)+Ephi*math.cos(phi)
    E=np.array([Ex,Ey,Ez])
    #gradB
    gradBr=myinterp.TwoD(var.gradBr,r,z)
    gradBphi=myinterp.TwoD(var.gradBphi,r,z)
    gradBz=myinterp.TwoD(var.gradBz,r,z)
    gradBx=gradBr*math.cos(phi)-gradBphi*math.sin(phi)
    gradBy=gradBr*math.sin(phi)+gradBphi*math.cos(phi)
    gradB=np.array([gradBx,gradBy,gradBz])
    #curlb
    curlbr=myinterp.TwoD(var.curlbr,r,z)
    curlbphi=myinterp.TwoD(var.curlbphi,r,z)
    curlbz=myinterp.TwoD(var.curlbz,r,z)
    curlbx=curlbr*math.cos(phi)-curlbphi*math.sin(phi)
    curlby=curlbr*math.sin(phi)+curlbphi*math.cos(phi)
    curlb=np.array([curlbx,curlby,curlbz])
    #equation of motion
    rhop=mi*vp/qi/Bmag
    D=1.+rhop*np.dot(b,curlb)
    #dvpdt=-np.dot(b+rhop*curlb,mu*gradB-qi*E)/(mi*D)
    #dot(b,E) is not zero even for zonal Er, that causes numerical problems
    #need to add dot(b,nonzonal E) in the future
    dvpdt=-np.dot(rhop*curlb,mu*gradB-qi*E)/(mi*D)
    dvpdt=dvpdt-np.dot(b,mu*gradB)/(mi*D)

    dxdt=vp*(b+rhop*curlb)+mycross(B,mu*gradB-qi*E)/qi/Bmag**2
    dxdt=dxdt/D
    return dxdt,dvpdt
