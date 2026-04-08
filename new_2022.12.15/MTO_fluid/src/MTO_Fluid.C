//Author: Yu Minghao    Updated: May 2020 

static char help[] = "topology optimization of fluid problem\n";
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
#include "../../common/OpenFOAMCompat.H"
#include "simpleControl.H"
#include "fvModels.H"
#include "fvConstraints.H"
#include "MMA/MMA.h"
#include <fstream>

using namespace Foam;

#include <diff.c>

int main(int argc, char *argv[])
{
    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFvModels.H"
    #include "createFvConstraints.H"
    #include "createFields.H"
    #include "readTransportProperties.H" 
    #include "initContinuityErrs.H"
    #include "opt_initialization.H"
    while (simple.loop(runTime))
    {
        #include "update.H"
        #include "Primal_U.H"
        #include "AdjointFlow_Ua.H"
        #include "costfunction.H"              
        #include "sensitivity.H"
    }
    #include "finalize.H"
    return 0;
}
