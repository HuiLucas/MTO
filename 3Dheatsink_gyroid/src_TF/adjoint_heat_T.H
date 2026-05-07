Info<< "\nslove adj-heat-b\n" << endl;
     fvScalarMatrix TEqnb
     (
        -fvm::div(-phi,Tb)
        +fvm::laplacian(DT, Tb)
      ==
         MTO_FV_SOURCE(Tb)+Q
     );

     // Adjoint Robin BC (homogeneous): -DT*dTb/dn = h_eff*Tb at walls
     const scalar h_eff = hconv.value() / rhoc.value();
     forAll(mesh.boundary(), patchI)
     {
         if (mesh.boundary()[patchI].type() == "wall")
         {
             const scalarField& magSf    = mesh.magSf().boundaryField()[patchI];
             const labelList&   faceCells = mesh.boundary()[patchI].faceCells();
             forAll(faceCells, faceI)
             {
                 const label  cellI = faceCells[faceI];
                 const scalar Av    = magSf[faceI] / mesh.V()[cellI];
                 TEqnb.diag()[cellI] += h_eff * Av;
             }
         }
     }

     TEqnb.relax();
     MTO_FV_CONSTRAIN(TEqnb);
     TEqnb.solve();
     MTO_FV_CORRECT(Tb);
