
Info<<"sensitivity analysis"<<endl;

fsens=-alphaMax*(1+qu)*qu/((qu+gamma)*(qu+gamma))*(U&Ub)+b1*(kf-ks)*(1+qu)*qu/((qu+gamma)*(qu+gamma))*(fvc::grad(T)&fvc::grad(Tb))/rhoc;

gsens_U=-alphaMax*(1+qu)*qu/((qu+gamma)*(qu+gamma))*(U&Uc);

#include "Heaviside_sens.H"
     
gx[0]=(DissPower/Foam::max(D0-opt*0.2,D1)-1.0);
gx[1]=(V);

for(i=0;i<allcells;i++)
{
  dfdx[i]=fsens[i]/scalef/nallcells;
  dgdx[0][i]=gsens_U[i]/scaleg/nallcells;
  dgdx[1][i]=gsens[i];
}

Info<< "\nrun mmasub\n" << endl;
mma.MMAsolver(xmma, dfdx, gx, dgdx);

for(i=0;i<allcells;i++)
{
  gamma[i]=xmma[i];
}

#include "Heaviside_rho.H"

opt++;

alphaMax=alphaMax*1.04;
alphaMax.value()=Foam::min(alphaMax.value(),1e5);//1e6,1e7

if(opt>=80)
{
  qu=0.005+(opt-80)*1e-4;
}
qu=Foam::min(qu,0.02);
qu.correctBoundaryConditions();
alpha=alphaMax*qu*(1-gamma)/(qu+gamma);
DT=(ks+(kf-ks)*gamma*(1+qu)/(qu+gamma))/rhoc;





