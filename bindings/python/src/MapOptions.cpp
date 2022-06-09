#include "CommonPybindUtilities.h"
#include "MParT/MapOptions.h"
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::MapOptionsWrapper(py::module &m)
{

    // BasisTypes
    py::enum_<BasisTypes>(m, "BasisTypes")
    .value("ProbabilistHermite",BasisTypes::ProbabilistHermite)
    .value("PhysicistHermite",BasisTypes::PhysicistHermite)
    .value("HermiteFunctions",BasisTypes::HermiteFunctions);
    //.export_values();

    // PosFuncTypes
    py::enum_<PosFuncTypes>(m, "PosFuncTypes")
    .value("Exp",PosFuncTypes::Exp)
    .value("SoftPlus",PosFuncTypes::SoftPlus);
    //.export_values();

    // QuadTypes
    py::enum_<QuadTypes>(m, "QuadTypes")
    .value("ClenshawCurtis",QuadTypes::ClenshawCurtis)
    .value("AdaptiveSimpson",QuadTypes::AdaptiveSimpson)
    .value("AdaptiveClenshawCurtis",QuadTypes::AdaptiveClenshawCurtis);
    //.export_values();

    // MapOptions
    py::class_<MapOptions, KokkosCustomPointer<MapOptions>>(m, "MapOptions")
    .def(py::init<>())
    .def_readwrite("basisType", &MapOptions::basisType)
    .def_readwrite("posFuncType", &MapOptions::posFuncType)
    .def_readwrite("quadType", &MapOptions::quadType)
    .def_readwrite("quadAbsTol", &MapOptions::quadAbsTol)
    .def_readwrite("quadRelTol", &MapOptions::quadRelTol)
    .def_readwrite("quadMaxSub", &MapOptions::quadMaxSub)
    .def_readwrite("quadPts", &MapOptions::quadPts);
    
        


}