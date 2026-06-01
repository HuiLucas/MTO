// 
    std::remove("Voluse.txt");
    std::remove("meanT.txt");
    std::remove("Disspower.txt");
    std::remove("Time.txt");
    std::remove("massflow.txt");
    std::remove("deltaP.txt");
    std::remove("alphaMax.txt");
//    
    scalar C(0);
    scalar DissPower(0);
    scalar meanT(0);
    scalar V(0);
    scalar scalef(0),scaleg(0);
    int opt1(readScalar(transportProperties.lookup("opt1")));
    int opt2(readScalar(transportProperties.lookup("opt2")));
   alphaMax.value() *= Foam::pow(5, opt1-1);
   alphaMax.value()=alphaMax.value();//1e6,1e7 TODO: fix min Alphamax
   alphamax.value() *= Foam::pow(5, opt1-1);
   alphamax.value()=alphaMax.value();//1e6,1e7
   Info<< "alphaMax: recalc " << alphaMax.value() << endl;
   std::ofstream outfileam("alphaMax.txt",std::ios::app);
   outfileam<<alphaMax.value()<<"\n";
   outfileam.close();
    int i=0,j=0,k=0,jk,allcells,nallcells,mycells=0,iter=0;
    scalar allcell=gamma.size();//allcell:length of gamma in the i-th core
    allcells=static_cast<int>(allcell); 
    List<scalar> cellsize(Pstream::nProcs());
    for(i=0;i<Pstream::nProcs();i++)
    {
       cellsize[i]=0;
    }
    cellsize[Pstream::myProcNo()]=allcells;
    reduce(cellsize, sumOp<List<scalar>>()); 
    reduce(allcell, sumOp<scalar>()); 
    nallcells=static_cast<int>(allcell);//nallcells:total length of gamma
    //
    for(i=0;i<Pstream::myProcNo();i++)
    {
       mycells=mycells+cellsize[i];
    }     

    double add1=0,add2=0,len=std::pow(mesh.V()[0],1.0/3);
    double eta0,eta1,eta5=0,del=0.1,y0,y1,y5; 

    double *Faa =new double[nallcells];
    double *Gbb =new double[nallcells];
    double *Gcc =new double[nallcells];
    double *Gdd =new double[nallcells];
    double *gsens =new double[nallcells];
    double *gamma_densityfilter=new double[nallcells];
    double *drho=new double[nallcells];

 // Heaviside density filter (only works on uniform mesh)

    List<vectorField> AllcellList(Pstream::nProcs());
    vectorField localcell(mesh.C());
    AllcellList[Pstream::myProcNo()] = localcell;
    Pstream::gatherList(AllcellList);
    Pstream::scatterList(AllcellList);
    vectorField Allcellcenter 
    (
        ListListOps::combine<vectorField>
        (
            AllcellList,
            accessOp<vectorField>()
        )
    );  
    int FR=filter_Radius;
    int Nnum[nallcells];
    int (*Nlabel)[125]=new int [nallcells][125];
    double (*Nweight)[125]=new double [nallcells][125];
    double geo_xmax=-1000,geo_ymax=-1000,geo_zmax=-1000,geo_xmin=1000,geo_ymin=1000,geo_zmin=1000;
    for(i=0;i<nallcells;i++)
    {
       geo_xmax=std::max(geo_xmax,Allcellcenter[i][0]);
       geo_ymax=std::max(geo_ymax,Allcellcenter[i][1]);
       geo_zmax=std::max(geo_zmax,Allcellcenter[i][2]);
       geo_xmin=std::min(geo_xmin,Allcellcenter[i][0]);
       geo_ymin=std::min(geo_ymin,Allcellcenter[i][1]);
       geo_zmin=std::min(geo_zmin,Allcellcenter[i][2]);
    }

    int nx,ny,nz,Nx,Ny,Nz;
    nx=static_cast<int>((geo_xmax-geo_xmin+len/4.0)/len)+FR*2+1;
    ny=static_cast<int>((geo_ymax-geo_ymin+len/4.0)/len)+FR*2+1;
    nz=static_cast<int>((geo_zmax-geo_zmin+len/4.0)/len)+FR*2+1;


    int ***ALLlabel;
    ALLlabel = new int **[nx]; 
    for(i=0; i<nx; i++) 
    { 
       ALLlabel[i]=new int *[ny]; 
       for(j=0; j<ny; j++)
       { 
          ALLlabel[i][j]=new int[nz]; 
       }
    }
    for(i=0;i<nx;i++)
    {
       for(j=0;j<ny;j++)
       {
          for(k=0;k<nz;k++)
          {
             ALLlabel[i][j][k]=-1;
          }
       }
    }
   for(jk=0;jk<nallcells;jk++)
   {
      nx=static_cast<int>((Allcellcenter[jk][0]-geo_xmin+len/4.0)/len);
      ny=static_cast<int>((Allcellcenter[jk][1]-geo_ymin+len/4.0)/len);
      nz=static_cast<int>((Allcellcenter[jk][2]-geo_zmin+len/4.0)/len);
      ALLlabel[nx+FR][ny+FR][nz+FR]=jk;
   }

   for(jk=0;jk<nallcells;jk++)
   {
      Nnum[jk]=0;
      nx=static_cast<int>((Allcellcenter[jk][0]-geo_xmin+len/4.0)/len);
      ny=static_cast<int>((Allcellcenter[jk][1]-geo_ymin+len/4.0)/len);
      nz=static_cast<int>((Allcellcenter[jk][2]-geo_zmin+len/4.0)/len);
      for(i=-FR;i<=FR;i++)
      {
         for(j=-FR;j<=FR;j++)
         {
            for(k=-FR;k<=FR;k++)
            {
               Nx=nx+i;
               Ny=ny+j;
               Nz=nz+k; 
               if(ALLlabel[Nx+FR][Ny+FR][Nz+FR]>=0)
               {
                  Nlabel[jk][Nnum[jk]]=ALLlabel[Nx+FR][Ny+FR][Nz+FR];
                  Nweight[jk][Nnum[jk]]=(3.8-std::sqrt(i*i+j*j+k*k))*len;
                  Nnum[jk]++;
               }
            }
         }
      }
   }

// Each MTO_TF invocation is a fresh process; gamma_densityfilter and eta5 are
// plain C arrays/scalars that do not persist across runs.  Re-derive them here
// from the gamma field that was written to disk by the previous run so that
// Heaviside_sens.H has consistent values when opt > 1.
{
    List<scalarField> AllrInit(Pstream::nProcs());
    scalarField localrInit(gamma.internalField());
    AllrInit[Pstream::myProcNo()] = localrInit;
    Pstream::gatherList(AllrInit);
    Pstream::scatterList(AllrInit);
    scalarField AllgammaInit
    (
        ListListOps::combine<scalarField>(AllrInit, accessOp<scalarField>())
    );
    for(jk=0;jk<nallcells;jk++)
    {
        double ad1=0.0, ad2=0.0;
        for(int jj=0;jj<Nnum[jk];jj++)
        {
            ad1 += Nweight[jk][jj]*AllgammaInit[Nlabel[jk][jj]];
            ad2 += Nweight[jk][jj];
        }
        gamma_densityfilter[jk] = (ad2>0.0) ? ad1/ad2 : AllgammaInit[jk];
    }
    // Recompute eta5 via the same bisection used in Heaviside_rho.H
    double etaA=0.0001, etaB=1.0;
    double yA=fun(gamma_densityfilter,del,etaA,nallcells);
    for(int ii=1;ii<17;ii++)
    {
        eta5=(etaA+etaB)/2.0;
        double yC=fun(gamma_densityfilter,del,eta5,nallcells);
        if(yA*yC<0.0) { etaB=eta5; }
        else { etaA=eta5; yA=yC; }
    }
    Info << "[init] gamma_densityfilter[0]=" << gamma_densityfilter[0]
         << " eta5=" << eta5 << endl;
}

//***********************************//MMA
int m=2;//m:number of constraints
MMA mma(allcells,m);
mma.asyminit = mma_init;
mma.asymdec = mma_dec;
mma.asyminc = mma_inc;
mma.raa0 = raa0;
mma.movelimit = movlim;

std::vector<double> xmma(allcells);
std::vector<double> dfdx(allcells);
std::vector<double> gx(m);
std::vector<std::vector<double>> dgdx(m);
for (int ii=0; ii<m; ii++)
{
   dgdx[ii].resize(allcells);
}

for(i=0;i<allcells;i++)
{
   xmma[i]=gamma[i];
}

gamma.correctBoundaryConditions();
qu.correctBoundaryConditions();
alpha=alphaMax*qu*(1-gamma)/(qu+gamma);
DT=(ks+(kf-ks)*gamma*(1+qu)/(qu+gamma))/rhoc; 
