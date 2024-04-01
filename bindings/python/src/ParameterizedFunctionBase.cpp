#include "CommonPybindUtilities.h"
#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/Utilities/ArrayConversions.h"

#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <Kokkos_Core.hpp>
#include <pybind11/pybind11.h>

#if defined(MPART_HAS_CEREAL)
#include <fstream>
#endif

namespace py = pybind11;
using namespace mpart::binding;

template<typename MemorySpace>
void mpart::binding::ParameterizedFunctionBaseWrapper<MemorySpace>(py::module &m)
{
    // ParameterizedFunctionBase
    py::class_<ParameterizedFunctionBase<MemorySpace>, std::shared_ptr<ParameterizedFunctionBase<MemorySpace>>>(m, "ParameterizedFunctionBase")
        .def("CoeffMap", [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> pfb) {
            Kokkos::View<const double*, MemorySpace> coeffs = pfb->CoeffMap();
            Kokkos::View<const double*, Kokkos::HostSpace> host_coeffs = ToHost(coeffs);
            return Eigen::VectorXd(Eigen::Map<Eigen::VectorXd>(host_coeffs.data(), host_coeffs.size()));
        })
        .def("SetCoeffs", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ParameterizedFunctionBase<MemorySpace>::SetCoeffs))
        .def("WrapCoeffs", [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> obj, std::tuple<long,int,int> coeffs){
            obj->WrapCoeffs(ToKokkos<double,MemorySpace>(coeffs));
        })
        .def("ParamMap", &ParameterizedFunctionBase<MemorySpace>::ParamMap)
        .def("SetParams", py::overload_cast<Eigen::Ref<Eigen::VectorXd>>(&ParameterizedFunctionBase<MemorySpace>::SetParams))
        .def("WrapParams", [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> obj, std::tuple<long,int,int> params){
            obj->WrapParams(ToKokkos<double,MemorySpace>(params));
        })
        .def("Evaluate", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<MemorySpace>::Evaluate))
        .def("EvaluateImpl", [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->EvaluateImpl(ToKokkos<double,MemorySpace>(input),ToKokkos<double,MemorySpace>(output));
        })
        .def("Gradient", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<MemorySpace>::Gradient))
        .def("GradientImpl", [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> sens, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->GradientImpl(ToKokkos<double,MemorySpace>(input),ToKokkos<double,MemorySpace>(sens), ToKokkos<double,MemorySpace>(output));
        })
        .def("InputGrad", static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<MemorySpace>::InputGrad))
        .def("InputGradImpl", [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> sens, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->InputGradImpl(ToKokkos<double,MemorySpace>(input),ToKokkos<double,MemorySpace>(sens), ToKokkos<double,MemorySpace>(output));
        })
        .def("CoeffGrad",static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<MemorySpace>::CoeffGrad))
        .def("CoeffGradImpl",[](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> sens, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->CoeffGradImpl(ToKokkos<double,MemorySpace>(input),ToKokkos<double,MemorySpace>(sens), ToKokkos<double,MemorySpace>(output));
        })
        .def("ParamGrad",static_cast<Eigen::RowMatrixXd (ParameterizedFunctionBase<MemorySpace>::*)(Eigen::Ref<const Eigen::RowMatrixXd> const&, Eigen::Ref<const Eigen::RowMatrixXd> const&)>(&ParameterizedFunctionBase<MemorySpace>::ParamGrad))
        .def("ParamGradImpl",[](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> obj, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> input, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> sens, std::tuple<long,std::tuple<int,int>,std::tuple<int,int>> output){
            obj->ParamGradImpl(ToKokkos<double,MemorySpace>(input),ToKokkos<double,MemorySpace>(sens), ToKokkos<double,MemorySpace>(output));
        })
        .def("torch", [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> obj, bool store_coeffs){
            auto mpart = py::module::import("mpart");
            if(!mpart.attr("mpart_has_torch").cast<bool>()){
                throw std::runtime_error("MParT could not import pytorch.");
            }
            return mpart.attr("TorchParameterizedFunctionBase")(obj, store_coeffs);
        }, py::arg("store_coeffs")=true)
        .def_property_readonly("numCoeffs", [](ParameterizedFunctionBase<MemorySpace>& pfb){
            // TODO: Create deprecation warning
            return pfb.numParams;
        })
        .def_readonly("numParams", &ParameterizedFunctionBase<MemorySpace>::numParams)
        .def_readonly("inputDim", &ParameterizedFunctionBase<MemorySpace>::inputDim)
        .def_readonly("outputDim", &ParameterizedFunctionBase<MemorySpace>::outputDim)
#if defined(MPART_HAS_CEREAL)
        .def("Serialize", [](ParameterizedFunctionBase<MemorySpace> const &obj, std::string const &filename){
            std::ofstream os(filename);
            cereal::BinaryOutputArchive archive(os);
            archive(obj.inputDim, obj.outputDim, obj.numCoeffs);
            archive(obj.Coeffs());
        })
        .def("ToBytes", [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const &ptr) {
            std::stringstream ss;
            ptr->Save(ss);
            return py::bytes(ss.str());
        })
        .def_static("FromBytes", [](std::string input) {
            std::stringstream ss;
            ss.str(input);

            auto ptr = ParameterizedFunctionBase<MemorySpace>::Load(ss);
            return ptr;  
        })
        .def(py::pickle(
            [](std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> const& ptr) { // __getstate__
                std::stringstream ss;
                ptr->Save(ss);
                return py::bytes(ss.str());
            },
            [](py::bytes input) {
                
                std::stringstream ss;
                ss.str(input);

                auto ptr = ParameterizedFunctionBase<MemorySpace>::Load(ss);
                return ptr;
            }
        ))
        #endif
        ;
}

template void mpart::binding::ParameterizedFunctionBaseWrapper<Kokkos::HostSpace>(py::module&);
#if defined(MPART_ENABLE_GPU)
template void mpart::binding::ParameterizedFunctionBaseWrapper<mpart::DeviceSpace>(py::module&);
#endif // MPART_ENABLE_GPU