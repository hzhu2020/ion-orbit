import variables as var
import myinterp
import numpy as np
import math
from parameters import max_step,cross_psitol,cross_rztol

def tau_orb(qi,mi,x,y,z,r_end,z_end,mu,Pphi,dt_orb,dt_xgc,nt):
  r=np.sqrt(x**2+y**2)
  Bmag=myinterp.TwoD(var.R,var.Z,var.Bmag,r,z)
  Bphi=myinterp.TwoD(var.R,var.Z,var.Bphi,r,z)
  vp=(Pphi-qi*var.psix)/mi/r/Bphi*Bmag

  r_orb1=np.zeros((nt,),dtype=float)
  z_orb1=np.zeros((nt,),dtype=float)
  phi_orb1=np.zeros((nt,),dtype=float)
  vp_orb1=np.zeros((nt,),dtype=float)
  
  tau=0
  step_count=0
  step_flag=True
  for it in range(np.int(max_step)):
    phi=math.atan2(y,x)
    r=np.sqrt(x**2+y**2)
    if step_flag and (math.floor(tau/dt_xgc)==step_count):
      r_orb1[step_count]=r
      z_orb1[step_count]=z
      phi_orb1[step_count]=phi
      vp_orb1[step_count]=vp
      if step_count<nt-1:
        step_count=step_count+1
      else:
        step_flag=False#only record nt steps of orbit information
    
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
    #RK4 2nd step
    dxdtc,dvpdtc=rhs(qi,mi,rc,phic,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/3
    dxdte=dxdte+dxdtc/3
    xc=x+dxdtc[0]*dt_orb/2
    yc=y+dxdtc[1]*dt_orb/2
    zc=z+dxdtc[2]*dt_orb/2
    phic=math.atan2(yc,xc)
    rc=np.sqrt(xc**2+yc**2)
    #RK4 3nd step
    dxdtc,dvpdtc=rhs(qi,mi,rc,phic,zc,mu,vpc)
    dvpdte=dvpdte+dvpdtc/3
    dxdte=dxdte+dxdtc/3
    xc=x+dxdtc[0]*dt_orb
    yc=y+dxdtc[1]*dt_orb
    zc=z+dxdtc[2]*dt_orb
    phic=math.atan2(yc,xc)
    rc=np.sqrt(xc**2+yc**2)
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
    psi=myinterp.TwoD(var.R,var.Z,var.psi2d,r,z)
    if (psi>(1+cross_psitol)*var.psix) or (np.sqrt((r-r_end)**2+(z-z_end)**2)<cross_rztol) or (not step_flag):
      break
  #end of the time loop

  if step_count==nt-1: 
    step_count=step_count+1 
  return step_count,r_orb1,z_orb1,phi_orb1,vp_orb1

def rhs(qi,mi,r,phi,z,mu,vp):
    #B
    Br=myinterp.TwoD(var.R,var.Z,var.Br,r,z)
    Bphi=myinterp.TwoD(var.R,var.Z,var.Bphi,r,z)
    Bz=myinterp.TwoD(var.R,var.Z,var.Bz,r,z)
    Bx=Br*math.cos(phi)-Bphi*math.sin(phi)
    By=Br*math.sin(phi)+Bphi*math.cos(phi)
    B=np.array([Bx,By,Bz])
    Bmag=np.sqrt(sum(B**2));
    b=B/Bmag
    #E
    Er=myinterp.TwoD(var.R,var.Z,var.Er,r,z)
    Ephi=myinterp.TwoD(var.R,var.Z,var.Ephi,r,z)
    Ez=myinterp.TwoD(var.R,var.Z,var.Ez,r,z)
    Ex=Er*math.cos(phi)-Ephi*math.sin(phi)
    Ey=Er*math.sin(phi)+Ephi*math.cos(phi)
    E=np.array([Ex,Ey,Ez])
    #gradB
    gradBr=myinterp.TwoD(var.R,var.Z,var.gradBr,r,z)
    gradBphi=myinterp.TwoD(var.R,var.Z,var.gradBphi,r,z)
    gradBz=myinterp.TwoD(var.R,var.Z,var.gradBz,r,z)
    gradBx=gradBr*math.cos(phi)-gradBphi*math.sin(phi)
    gradBy=gradBr*math.sin(phi)+gradBphi*math.cos(phi)
    gradB=np.array([gradBx,gradBy,gradBz])
    #curlb
    curlbr=myinterp.TwoD(var.R,var.Z,var.curlbr,r,z)
    curlbphi=myinterp.TwoD(var.R,var.Z,var.curlbphi,r,z)
    curlbz=myinterp.TwoD(var.R,var.Z,var.curlbz,r,z)
    curlbx=curlbr*math.cos(phi)-curlbphi*math.sin(phi)
    curlby=curlbr*math.sin(phi)+curlbphi*math.cos(phi)
    curlb=np.array([curlbx,curlby,curlbz])
    #equation of motion
    rhop=mi*vp/qi/Bmag
    D=1.+rhop*np.dot(b,curlb)
    dvpdt=-np.dot(b+rhop*curlb,mu*gradB-qi*E)/(mi*D)
    dxdt=vp*(b+rhop*curlb)+np.cross(B,mu*gradB-qi*E)/qi/Bmag**2
    dxdt=dxdt/D
    return dxdt,dvpdt
