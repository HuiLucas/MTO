
// define scalar field Allfsens
List<scalarField> AllfList(Pstream::nProcs());
scalarField localfsens(fsens.internalField());
AllfList[Pstream::myProcNo()] = localfsens;
Pstream::gatherList(AllfList);
Pstream::scatterList(AllfList);
scalarField Allfsens 
(
    ListListOps::combine<scalarField>
    (
        AllfList,
        accessOp<scalarField>()
    )
);
// define scalar field GSENS_U
List<scalarField> AllguList(Pstream::nProcs());
scalarField localgusens(gsens_U.internalField());
AllguList[Pstream::myProcNo()] = localgusens;
Pstream::gatherList(AllguList);
Pstream::scatterList(AllguList);
scalarField Allgusens 
(
    ListListOps::combine<scalarField>
    (
        AllguList,
        accessOp<scalarField>()
    )
);  

scalef=0;
scaleg=0;    
for(i=0;i<allcells;i++)
{
   scalef=std::max(scalef,mag(fsens[i]));
   scaleg=std::max(scaleg,mag(gsens_U[i]));
}
Foam::reduce(scalef, Foam::maxOp<Foam::scalar>());
Foam::reduce(scaleg, Foam::maxOp<Foam::scalar>());

// Guard against zero sensitivities (e.g. near-uniform T field).
if (scalef < SMALL) { scalef = 1.0; Info << "Warning: scalef~0, sensitivities are degenerate" << endl; }
if (scaleg < SMALL) { scaleg = 1.0; Info << "Warning: scaleg~0, sensitivities are degenerate" << endl; }
Info << "[dbg-hs] scalef=" << scalef << " scaleg=" << scaleg << " eta5=" << eta5 << " del=" << del << endl;

// gsens must be zeroed for opt==1 (uninitialized heap memory → potential NaN → SIGFPE)
for(i=0;i<nallcells;i++) gsens[i]=0.0;

Info << "[dbg-hs] A: gsens zeroed, entering opt>1 drho block (opt=" << opt << ")" << endl;

if(opt>1)
{
  for(i=0;i<nallcells;i++)
  {
     if(gamma_densityfilter[i]<=eta5)
     {
       drho[i]=del*Foam::exp(-del*(1-gamma_densityfilter[i]/eta5))+Foam::exp(-del);
     }
     else
     {
       drho[i]=del*Foam::exp(-del*(gamma_densityfilter[i]-eta5)/(1-eta5))+Foam::exp(-del);
     }
     Allfsens[i]=Allfsens[i]*drho[i];
     Allgusens[i]=Allgusens[i]*drho[i];
     gsens[i]=1.0/nallcells*drho[i];
  }
}

Info << "[dbg-hs] B: drho block done, entering Faa loop" << endl;

for(i=0;i<nallcells;i++)//ith node
     {
        add1=0;
        Faa[i]=0;
        Gbb[i]=0;
        Gdd[i]=0;
        for(j=0;j<Nnum[i];j++)//j nodes around the ith node
        {
           add2=0;
           for(k=0;k<Nnum[Nlabel[i][j]];k++)//k nodes around the jth node
           {
              add2=add2+Nweight[Nlabel[i][j]][k];
           }
           add1=Nweight[i][j];
           Faa[i]=Faa[i]+Allfsens[Nlabel[i][j]]*add1/add2;
           Gbb[i]=Gbb[i]+Allgusens[Nlabel[i][j]]*add1/add2;
           Gdd[i]=Gdd[i]+gsens[Nlabel[i][j]]*add1/add2;
         }
      }

Info << "[dbg-hs] C: Faa loop done, entering copy-back block" << endl;

     if(opt>1)
     {
        for(i=0;i<allcells;i++)
        {
           fsens[i]=Faa[i+mycells];
           gsens_U[i]=Gbb[i+mycells];
           gsens[i]=Gdd[i+mycells];
        }
     }

Info << "[dbg-hs] D: copy-back done" << endl;