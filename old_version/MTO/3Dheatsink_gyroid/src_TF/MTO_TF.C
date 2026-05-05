/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | Copyright (C) 2011-2016 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.
\*---------------------------------------------------------------------------*/

#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvc.H"
#include "fvm.H"
#include "zeroGradientFvPatchFields.H"
#include "argList.H"
#include "IOMRFZoneList.H"
#include "adjustPhi.H"
#include "constrainPressure.H"
#include "constrainHbyA.H"
#include "../../../../new_2022.12.15/common/OpenFOAMCompat.H"
#include "simpleControl.H"
#include "fvOptionCompat.H"
#include <math.h>
#include <fstream>
#include <iostream>
#include <iosfwd>
#include <stdio.h>
#include <mpi.h>
#include <MMA.h>

using namespace Foam;

template<class Type>
void setCells
(
    GeometricField<Type, fvPatchField, volMesh>& vf,
    const labelList& cells,
    double value
)
{
    forAll(cells, i)
    {
        vf[cells[i]] = value;
    }
}
double fun(double gamma[],double del,double eta,int allcells);
int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #if __has_include("createFvOptions.H")
    #include "createFvOptions.H"
    #else
    #include "createFvModels.H"
    #include "createFvConstraints.H"
    #endif
    #include "createFields.H"
    #include "readThermalProperties.H"
    #include "initContinuityErrs.H"
    #include "SIMP_initialize.H"
    while (simple.loop(runTime))
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
        #include "primal_equation.H"
        #include "adjoint_equation_T.H"
        for(i=0;i<30;i++)
        {
           #include "adjoint_flow_U.H" 
        }   
        #include "costfunction.H"                           
        #include "sensitivity.H"        
        if(runTime.writeTime())
        {
        gamma.write();
        fsens.write();    // dJ_thermal/dgamma  – used by heat-minimisation mode
        gsens_U.write();  // dDissPower/dgamma  – used by pressure-drop mode
        T.write();
        U.write();
        p.write();
        }

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
        << "  ClockTime = " << runTime.elapsedClockTime() << " s"
        << nl << endl;
    }
    Info<< "End\n" << endl;
    return 0;
}


//*********************************************************//function
double fun(double gamma[],double del,double eta,int allcells)
{
     int i;
     double z=0;
     double *fg =new double[allcells];
     
     for(i=0;i<allcells;i++)
     {
        if(gamma[i]<=eta)
        {
          fg[i]=eta*(Foam::exp(-del*(1-gamma[i]/eta))-(1-gamma[i]/eta)*Foam::exp(-del));
        }
        else
        {
          fg[i]=eta+(1-eta)*(1-Foam::exp(-del*(gamma[i]-eta)/(1-eta))+(gamma[i]-eta)*Foam::exp(-del)/(1-eta));
        }
     }
     for(i=0;i<allcells;i++)
     {
        z=z+gamma[i]-fg[i];
     }
     delete fg;
     return {z};
}

