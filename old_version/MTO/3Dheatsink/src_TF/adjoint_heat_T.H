Info<< "\nslove adj-heat-b\n" << endl;
     fvScalarMatrix TEqnb
     (
        -fvm::div(-phi,Tb) 
        +fvm::laplacian(DT, Tb)
      ==
         MTO_FV_SOURCE(Tb)+Q
     );
     TEqnb.relax();
     MTO_FV_CONSTRAIN(TEqnb);
     TEqnb.solve();
     MTO_FV_CORRECT(Tb);
