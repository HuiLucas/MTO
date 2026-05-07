Info<< "\nslove pri-heat\n" << endl;
      const scalar h_eff = hconv.value() / rhoc.value(); // [m/s]
      const scalar T_ext = Texterior.value();

      fvScalarMatrix TEqn
      (
          fvm::div(phi,T)
         -fvm::laplacian(DT, T)
       ==
          MTO_FV_SOURCE(T)
      );

      // Robin BC at walls: q = hconv*(T_wall - Texterior)
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
                  TEqn.diag()[cellI]   += h_eff * Av;
                  TEqn.source()[cellI] += h_eff * T_ext * Av;
              }
          }
      }

      TEqn.relax();
      MTO_FV_CONSTRAIN(TEqn);
      TEqn.solve();
      MTO_FV_CORRECT(T);

