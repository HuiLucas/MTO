    Info<< "Reading thermal properties\n" << endl;

    //************************************************************************//    
    IOdictionary thermalProperties
    (
        IOobject
        (
            "thermalProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );
   dimensionedScalar rhoc(thermalProperties.lookup("rhoc"));//rho_fluid*heat_capacity
   dimensionedScalar kf(thermalProperties.lookup("kf"));//Thermal Conductivity of fluid
   dimensionedScalar ks(thermalProperties.lookup("ks"));//Thermal Conductivity of solid
   //Switch thermalStress(thermalProperties.lookup("thermalStress"));
   //dimensionedScalar Talpha(thermalProperties.lookup("Talpha"));//Thermal expansion coefficient
    dimensionedScalar Texterior(thermalProperties.lookup("Texterior"));//Ambient temperature for thermal expansion
   dimensionedScalar hconv(thermalProperties.lookup("hconv"));//Convective HTC at walls [W/(m^2 K)]

 //***************************************************************************//adjoint source
     // Q acts as the adjoint source dJ/dT for the cost function J = integral(Q*T dV).
     // It is NOT added to the primal equation (heat enters via Robin BC at walls instead).
     volScalarField Q
     (
         IOobject
         (
             "Q",
             runTime.timeName(),
             mesh,
             IOobject::READ_IF_PRESENT,
             IOobject::NO_WRITE
         ),
         mesh,
         dimensionedScalar("Q", dimensionSet(0,0,-1,1,0,0,0),100000000.0/4180000),
         calculatedFvPatchScalarField::typeName
     );

   volScalarField DT
   (
       IOobject
       (
           "DT",
           runTime.timeName(),
           mesh,
           IOobject::READ_IF_PRESENT,
           IOobject::AUTO_WRITE
       ),
      (ks+(kf-ks)*gamma*(1+qu)/(qu+gamma))/rhoc
   );

 //***************************************************************************//pri-heat
    Info<< "Reading field T\n" << endl;
        volScalarField T
        (
            IOobject
            (
                "T",
                runTime.timeName(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        );
     Info<< "Reading field Tb\n" << endl;
        volScalarField Tb
        (
            IOobject
            (
                "Tb",
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::AUTO_WRITE
            ),
            mesh
        ); 
              
