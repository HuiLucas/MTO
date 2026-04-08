#ifndef MTO_OPENFOAM_COMPAT_H
#define MTO_OPENFOAM_COMPAT_H

#if __has_include("singlePhaseTransportModel.H")
    #include "singlePhaseTransportModel.H"
    #include "turbulentTransportModel.H"

namespace Foam
{
namespace mtoCompat
{
    typedef singlePhaseTransportModel transportModel;
    typedef incompressible::turbulenceModel momentumTransportModel;

    inline autoPtr<momentumTransportModel> NewMomentumTransport
    (
        volVectorField& U,
        surfaceScalarField& phi,
        transportModel& transport
    )
    {
        return momentumTransportModel::New(U, phi, transport);
    }

    inline tmp<fvVectorMatrix> divDevMomentum
    (
        momentumTransportModel& turbulence,
        volVectorField& U
    )
    {
        return turbulence.divDevReff(U);
    }
}
}

#elif __has_include("viscosityModel.H") && __has_include("incompressibleMomentumTransportModels.H")
    #include "viscosityModel.H"
    #include "incompressibleMomentumTransportModels.H"

namespace Foam
{
namespace mtoCompat
{
    typedef viscosityModel transportModel;
    typedef incompressible::momentumTransportModel momentumTransportModel;

    inline autoPtr<transportModel> NewTransportModel(const fvMesh& mesh)
    {
        return transportModel::New(mesh);
    }

    inline autoPtr<momentumTransportModel> NewMomentumTransport
    (
        volVectorField& U,
        surfaceScalarField& phi,
        transportModel& transport
    )
    {
        return momentumTransportModel::New(U, phi, transport);
    }

    inline tmp<fvVectorMatrix> divDevMomentum
    (
        momentumTransportModel& turbulence,
        volVectorField& U
    )
    {
        return turbulence.divDevSigma(U);
    }
}
}

#else
    #error "Unsupported OpenFOAM version for MTO compatibility layer."
#endif

#endif
