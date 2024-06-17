#include "CommonPybindUtilities.h"
#include "MParT/Sigmoid.h"
#include "MParT/Utilities/ArrayConversions.h"
#include <Eigen/Core>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mpart::binding;

void mpart::binding::Sigmoid1dWrapper(py::module &m)
{
    typedef Sigmoid1d<Kokkos::HostSpace, mpart::Logistic, mpart::SoftPlus> SigmoidType;

    py::class_<SigmoidType, std::shared_ptr<SigmoidType>>(m, "Sigmoid1d")
        .def(py::init([](Eigen::Ref<Eigen::VectorXd> centers, 
                         Eigen::Ref<Eigen::VectorXd> widths,
                         Eigen::Ref<Eigen::VectorXd> weights, 
                         Eigen::Ref<Eigen::Matrix<unsigned int, Eigen::Dynamic, 1>> startInds) {
                return std::make_shared<SigmoidType>(mpart::VecToKokkos<double>(centers),
                                                     mpart::VecToKokkos<double>(widths),
                                                     mpart::VecToKokkos<double>(weights),
                                                     mpart::VecToKokkos<unsigned int>(startInds));
            }))
        .def("EvaluateAll", [](const SigmoidType &sig,
                               Eigen::Ref<Eigen::VectorXd> output, 
                               double x){sig.EvaluateAll(output.data(),output.size()-1,x);})
        .def("EvaluateDerivatives", [](const SigmoidType &sig,
                               Eigen::Ref<Eigen::VectorXd> output, 
                               Eigen::Ref<Eigen::VectorXd> derivs,
                               double x){
                                if(output.size()!=derivs.size()){
                                    throw std::invalid_argument("Vector dimensions do not match.");
                                }else{
                                   sig.EvaluateDerivatives(output.data(),derivs.data(),output.size()-1,x);
                                }})
        .def("EvaluateSecondDerivatives", [](const SigmoidType &sig,
                               Eigen::Ref<Eigen::VectorXd> output, 
                               Eigen::Ref<Eigen::VectorXd> derivs,
                               Eigen::Ref<Eigen::VectorXd> derivs2,
                               double x){
                                if((output.size()!=derivs.size())||(derivs.size()!=derivs2.size())){
                                    throw std::invalid_argument("Vector dimensions do not match.");
                                }else{
                                   sig.EvaluateSecondDerivatives(output.data(),derivs.data(),derivs2.data(),output.size()-1,x);
                                }})
        ;

}
