
 V=0;
 meanT=0;
 for (i=0;i<allcells;i++)
 {
     V=gamma[i]+V;
     meanT=meanT+T[i];
 }
 reduce(meanT, sumOp<scalar>());
 reduce(V, sumOp<scalar>());     
 V=V/nallcells-voluse;

 //DissPower
 DissPower=0;
 for (i=0; i<nObjPatch; i++)
 {
      DissPower =DissPower - sum(
      phi.boundaryField()[conPatchList[i]]*(p.boundaryField()[conPatchList[i]]
      + 0.5*magSqr(U.boundaryField()[conPatchList[i]]))); 
 }
 DissPower=DissPower/D_normalization;
 if(geo_dim==2) 
 {
   DissPower=DissPower/len;
 }
 reduce(DissPower, sumOp<scalar>()); 
 Info<<"constraint function (DissPower): "<<DissPower<<endl; 

 // ── Mass flow rate (volumetric, m³/s) ──────────────────────────────────────
 // phi is the outward face flux; at the inlet the outward normal opposes flow
 // so sum(phi_inlet) is negative – take the magnitude.
 label inletPatchID  = mesh.boundaryMesh().findPatchID("inlet");
 label outletPatchID = mesh.boundaryMesh().findPatchID("outlet");

 scalar massflow = 0;
 if (inletPatchID >= 0)
 {
     massflow = mag(sum(phi.boundaryField()[inletPatchID]));
 }
 reduce(massflow, sumOp<scalar>());

 // ── Pressure drop (kinematic, m²/s²) ───────────────────────────────────────
 // Area-weighted average p at inlet minus area-weighted average p at outlet.
 // p_outlet is fixed at 0 in the BC, so deltaP ≈ <p>_inlet.
 scalar pInletSum  = 0, aInlet  = 0;
 scalar pOutletSum = 0, aOutlet = 0;
 if (inletPatchID >= 0)
 {
     const scalarField& patchP    = p.boundaryField()[inletPatchID];
     const scalarField& patchArea = mesh.magSf().boundaryField()[inletPatchID];
     pInletSum = sum(patchP * patchArea);
     aInlet    = sum(patchArea);
 }
 if (outletPatchID >= 0)
 {
     const scalarField& patchP    = p.boundaryField()[outletPatchID];
     const scalarField& patchArea = mesh.magSf().boundaryField()[outletPatchID];
     pOutletSum = sum(patchP * patchArea);
     aOutlet    = sum(patchArea);
 }
 reduce(pInletSum,  sumOp<scalar>());
 reduce(aInlet,     sumOp<scalar>());
 reduce(pOutletSum, sumOp<scalar>());
 reduce(aOutlet,    sumOp<scalar>());

 scalar deltaP = 0;
 if (aInlet > SMALL)  deltaP += pInletSum  / aInlet;
 if (aOutlet > SMALL) deltaP -= pOutletSum / aOutlet;

 Info << "massflow (m3/s): " << massflow << "  deltaP (m2/s2): " << deltaP << endl;

 if (Pstream::master())
{
std::ofstream outfile1("Voluse.txt",std::ios::app);
outfile1<<V<<"\n";
outfile1.close();

std::ofstream outfile2("meanT.txt",std::ios::app);
outfile2<<meanT/nallcells<<"\n";
outfile2.close();

std::ofstream outfile3("Disspower.txt",std::ios::app);
outfile3<<DissPower<<"\n";
outfile3.close();

std::ofstream outfile4("Time.txt",std::ios::app);
outfile4<<runTime.elapsedClockTime()<<"\n";
outfile4.close();

std::ofstream outfile5("massflow.txt",std::ios::app);
outfile5<<massflow<<"\n";
outfile5.close();

std::ofstream outfile6("deltaP.txt",std::ios::app);
outfile6<<deltaP<<"\n";
outfile6.close();

}