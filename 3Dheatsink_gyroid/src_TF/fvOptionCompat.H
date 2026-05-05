#ifndef MTO_FV_OPTION_COMPAT_H
#define MTO_FV_OPTION_COMPAT_H

#if __has_include("fvOptions.H")
    #include "fvOptions.H"
    #define MTO_FV_SOURCE(field) fvOptions(field)
    #define MTO_FV_CONSTRAIN(eqn) fvOptions.constrain(eqn)
    #define MTO_FV_CORRECT(field) fvOptions.correct(field)
#else
    #include "fvModels.H"
    #include "fvConstraints.H"
    #define MTO_FV_SOURCE(field) fvModels.source(field)
    #define MTO_FV_CONSTRAIN(eqn) fvConstraints.constrain(eqn)
    #define MTO_FV_CORRECT(field) fvConstraints.constrain(field)
#endif

#endif
