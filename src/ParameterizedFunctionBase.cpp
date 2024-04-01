#include "MParT/ParameterizedFunctionBase.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/Utilities/GPUtils.h"

using namespace mpart;

template<>
void ParameterizedFunctionBase<Kokkos::HostSpace>::CheckDeviceMismatch(std::string) const
{
}

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::CheckDeviceMismatch(std::string functionName) const
{
    std::stringstream msg;
    msg << "Error in call to \"" << functionName << "\".  This function is only valid on the host space,";
    msg << " but called on a DeviceSpace ParameterizedFunctionBase object.   You must manually copy the input";
    msg << " argument to device space if you want to call this function.";
    throw std::runtime_error(msg.str());
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate(StridedMatrix<const double, Kokkos::HostSpace> const& pts)
{
    CheckParameters("Evaluate");

    Kokkos::View<double**, Kokkos::HostSpace> output("Map Evaluations", outputDim, pts.extent(1));
    EvaluateImpl(pts, output);
    return output;
}

template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckParameters("Evaluate");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);
    StridedMatrix<double, Kokkos::HostSpace> outView = MatToKokkos<double,Kokkos::HostSpace>(output);
    EvaluateImpl(ptsView, outView);
    return output;
}


#if defined(MPART_ENABLE_GPU)
template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::Evaluate(StridedMatrix<const double, mpart::DeviceSpace> const& pts)
{
    CheckParameters("Evaluate");

    Kokkos::View<double**, mpart::DeviceSpace> output("Map Evaluations", outputDim, pts.extent(1));
    EvaluateImpl(pts, output);
    return output;
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::Evaluate(StridedMatrix<const double, Kokkos::HostSpace> const& pts)
{
    // Copy the points to the device space
    StridedMatrix<const double, mpart::DeviceSpace> pts_device = ToDevice<mpart::DeviceSpace>(pts);

    // Evaluate on the device space
    StridedMatrix<double, mpart::DeviceSpace> evals_device = this->Evaluate(pts_device);

    // Copy back to the host space
    return ToHost(evals_device);
}

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::Evaluate(StridedMatrix<const double, mpart::DeviceSpace> const& pts)
{
    // Copy the points to host
    StridedMatrix<const double, Kokkos::HostSpace> pts_host = ToHost(pts);

    // Evaluate on the host
    StridedMatrix<double, Kokkos::HostSpace> evals_host = this->Evaluate(pts_host);

    // Copy back to the device
    return ToDevice<mpart::DeviceSpace>(evals_host);
}


template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<mpart::DeviceSpace>::Evaluate(Eigen::Ref<const Eigen::RowMatrixXd> const& pts)
{
    CheckParameters("Evaluate");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    StridedMatrix<const double, mpart::DeviceSpace> ptsView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts));
    return KokkosToMat( ToHost(this->Evaluate(ptsView)));
}

#endif



template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::InputGrad(StridedMatrix<const double, Kokkos::HostSpace> const& pts, StridedMatrix<const double, Kokkos::HostSpace> const& sens)
{
    CheckParameters("Gradient");

    Kokkos::View<double**, Kokkos::HostSpace> output("Gradients", inputDim, pts.extent(1));
    InputGradImpl(pts, sens, output);
    return output;
}

template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<Kokkos::HostSpace>::InputGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts, Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    CheckParameters("Gradient");

    Eigen::RowMatrixXd output(inputDim, pts.cols());
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);
    StridedMatrix<const double, Kokkos::HostSpace> sensView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(sens);
    StridedMatrix<double, Kokkos::HostSpace> outView = MatToKokkos<double,Kokkos::HostSpace>(output);
    InputGradImpl(ptsView, sensView, outView);
    return output;
}


#if defined(MPART_ENABLE_GPU)
template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::InputGrad(StridedMatrix<const double, mpart::DeviceSpace> const& pts, StridedMatrix<const double, mpart::DeviceSpace> const& sens)
{
    CheckParameters("Gradient");

    Kokkos::View<double**, mpart::DeviceSpace> output("Map Evaluations", outputDim, pts.extent(1));
    InputGradImpl(pts, sens, output);
    return output;
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::InputGrad(StridedMatrix<const double, Kokkos::HostSpace> const& pts, StridedMatrix<const double, Kokkos::HostSpace> const& sens)
{
    // Copy the points to the device space
    StridedMatrix<const double, mpart::DeviceSpace> pts_device = ToDevice<mpart::DeviceSpace>(pts);
    StridedMatrix<const double, mpart::DeviceSpace> sens_device = ToDevice<mpart::DeviceSpace>(sens);
    // Evaluate on the device space
    StridedMatrix<double, mpart::DeviceSpace> evals_device = this->InputGrad(pts_device, sens_device);

    // Copy back to the host space
    return ToHost(evals_device);
}

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::InputGrad(StridedMatrix<const double, mpart::DeviceSpace> const& pts, StridedMatrix<const double, mpart::DeviceSpace> const& sens)
{
    // Copy the points to host
    StridedMatrix<const double, Kokkos::HostSpace> pts_host = ToHost(pts);
    StridedMatrix<const double, Kokkos::HostSpace> sens_host = ToHost(sens);

    // Evaluate on the host
    StridedMatrix<double, Kokkos::HostSpace> evals_host = this->InputGrad(pts_host, sens_host);

    // Copy back to the device
    return ToDevice<mpart::DeviceSpace>(evals_host);
}


template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<mpart::DeviceSpace>::InputGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts, Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    CheckParameters("Evaluate");

    Eigen::RowMatrixXd output(outputDim, pts.cols());
    StridedMatrix<const double, mpart::DeviceSpace> ptsView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts));
    StridedMatrix<const double, mpart::DeviceSpace> sensView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(sens));

    return KokkosToMat( ToHost(this->InputGrad(ptsView, sensView)));
}

#endif

template<typename MemorySpaceSrc, typename MemorySpaceDest>
void SetParamsInternal(unsigned int numParams,
        Kokkos::View<double*,MemorySpaceDest>& coeffsDest,
        Kokkos::View<const double*,MemorySpaceSrc> coeffsSrc) {
    // If coefficients already exist, make sure the sizes match
    if(coeffsDest.is_allocated()){
        if(coeffsSrc.size() != numParams){
            std::stringstream msg;
            msg << "Error in ParameterizedFunctionBase<MemorySpace>::SetParams.  Expected coefficient vector with size " << numParams << ", but new coefficients have size " << coeffsSrc.size() << ".";
            throw std::invalid_argument(msg.str());
        }

        if(coeffsDest.size() != numParams)
            Kokkos::resize(coeffsDest, numParams);
    }else{
        coeffsDest = Kokkos::View<double*, MemorySpaceDest>("ParameterizedFunctionBase<MemorySpace> Coefficients", numParams);
    }

    Kokkos::deep_copy(coeffsDest, coeffsSrc);
}

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::SetParams(Kokkos::View<const double*, MemorySpace> coeffs){
    SetParamsInternal(this->numParams, this->savedParams, coeffs);
}

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::SetParams(Kokkos::View<double*, MemorySpace> coeffs){
    Kokkos::View<const double*, MemorySpace> ConstCoeffs = coeffs;
    SetParams(ConstCoeffs);
}



template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::WrapParams(Kokkos::View<double*, MemorySpace> coeffs){

    if(coeffs.size() != numParams){
        std::stringstream msg;
        msg << "Error in ParameterizedFunctionBase<MemorySpace>::WrapParams.  Expected coefficient vector with size " << numParams << ", but new coefficients have size " << coeffs.size() << ".";
        throw std::invalid_argument(msg.str());
    }
    this->savedParams = coeffs;

}

#if defined(MPART_ENABLE_GPU)
template<>
void ParameterizedFunctionBase<mpart::DeviceSpace>::SetParams(Kokkos::View<const double*, Kokkos::HostSpace> coeffs)
{
    SetParamsInternal(this->numParams, this->savedParams, coeffs);
}
template<>
void ParameterizedFunctionBase<Kokkos::HostSpace>::SetParams(Kokkos::View<const double*, mpart::DeviceSpace> coeffs)
{
    SetParamsInternal(this->numParams, this->savedParams, coeffs);
}

template<>
void ParameterizedFunctionBase<mpart::DeviceSpace>::SetParams(Kokkos::View<double*, Kokkos::HostSpace> coeffs)
{
    Kokkos::View<const double*, Kokkos::HostSpace> ConstCoeffs = coeffs;
    SetParams(ConstCoeffs);
}
template<>
void ParameterizedFunctionBase<Kokkos::HostSpace>::SetParams(Kokkos::View<double*, mpart::DeviceSpace> coeffs)
{   
    Kokkos::View<const double*, mpart::DeviceSpace> ConstCoeffs = coeffs;
    SetParams(ConstCoeffs);
}



#endif

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::SetParams(Eigen::Ref<Eigen::VectorXd> coeffs)
{
    Kokkos::View<const double*, MemorySpace> inputCoeffs = VecToKokkos<double, MemorySpace>(coeffs);
    SetParams(inputCoeffs);
}

template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::WrapParams(Eigen::Ref<Eigen::VectorXd> coeffs) {
    CheckDeviceMismatch("WrapParams(Eigen::Ref<Eigen::VectorXd");
}

template<>
void ParameterizedFunctionBase<Kokkos::HostSpace>::WrapParams(Eigen::Ref<Eigen::VectorXd> coeffs)
{
    WrapParams(Kokkos::View<double*, Kokkos::HostSpace>(VecToKokkos<double, Kokkos::HostSpace>(coeffs)));
}

template<>
Eigen::Map<Eigen::VectorXd> ParameterizedFunctionBase<Kokkos::HostSpace>::ParamMap()
{
    CheckParameters("ParamMap");
    return KokkosToVec(this->savedParams);
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::ParamGrad(StridedMatrix<const double, Kokkos::HostSpace> const& pts,
                                                                                                 StridedMatrix<const double, Kokkos::HostSpace> const& sens)
{
    CheckParameters("ParamGrad");
    Kokkos::View<double**, Kokkos::HostSpace> output("Param Grad", numParams, pts.extent(1));
    ParamGradImpl(pts,sens, output);
    return output;
}

template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<Kokkos::HostSpace>::ParamGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts,
                                                                           Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    CheckParameters("ParamGrad");
    StridedMatrix<const double, Kokkos::HostSpace> ptsView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts);
    StridedMatrix<const double, Kokkos::HostSpace> sensView = ConstRowMatToKokkos<double,Kokkos::HostSpace>(sens);

    Kokkos::View<double**,Kokkos::LayoutRight,Kokkos::HostSpace> outView = ParamGrad(ptsView, sensView);
    return KokkosToMat(outView);
}


#if defined(MPART_ENABLE_GPU)
template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::ParamGrad(StridedMatrix<const double, mpart::DeviceSpace> const& pts,
                                                                                                   StridedMatrix<const double, mpart::DeviceSpace> const& sens)
{
    CheckParameters("ParamGrad");
    Kokkos::View<double**, mpart::DeviceSpace> output("Coeff Grad", numParams, pts.extent(1));
    ParamGradImpl(pts,sens, output);
    return output;
}

template<>
template<>
StridedMatrix<double, mpart::DeviceSpace> ParameterizedFunctionBase<Kokkos::HostSpace>::ParamGrad(StridedMatrix<const double, mpart::DeviceSpace> const& pts,
                                                                                                  StridedMatrix<const double, mpart::DeviceSpace> const& sens)
{
    return ToDevice<mpart::DeviceSpace>( this->ParamGrad(ToHost(pts), ToHost(sens)));
}

template<>
template<>
StridedMatrix<double, Kokkos::HostSpace> ParameterizedFunctionBase<mpart::DeviceSpace>::ParamGrad(StridedMatrix<const double, Kokkos::HostSpace> const& pts,
                                                                                                  StridedMatrix<const double, Kokkos::HostSpace> const& sens)
{
    return ToHost( this->ParamGrad(ToDevice<mpart::DeviceSpace>(pts), ToDevice<mpart::DeviceSpace>(sens)));
}

template<>
Eigen::RowMatrixXd ParameterizedFunctionBase<mpart::DeviceSpace>::ParamGrad(Eigen::Ref<const Eigen::RowMatrixXd> const& pts,
                                                                            Eigen::Ref<const Eigen::RowMatrixXd> const& sens)
{
    StridedMatrix<const double, mpart::DeviceSpace> ptsView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(pts) );
    StridedMatrix<const double, mpart::DeviceSpace> sensView = ToDevice<mpart::DeviceSpace>( ConstRowMatToKokkos<double,Kokkos::HostSpace>(sens) );

    return KokkosToMat( ToHost( ParamGrad(ptsView, sensView) ));
}


#endif
template<typename MemorySpace>
bool ParameterizedFunctionBase<MemorySpace>::CheckParameters() const
{
    if(this->numParams==0)
        return true;

    bool good = true;

    if(!this->savedParams.is_allocated()){
        good = false;
    }else if(this->savedParams.size()!=this->numParams){
        good = false;
    }

    return good;

}


template<typename MemorySpace>
void ParameterizedFunctionBase<MemorySpace>::CheckParameters(std::string const& functionName) const
{

    bool good = CheckParameters();

    if(!good){
        std::stringstream msg;
        msg << "Error in \"" << functionName << "\", the coefficients have not been set yet.  Make sure to call SetParams() before calling this function.";
        throw std::runtime_error(msg.str());
    }
}


// Explicit template instantiation
template class mpart::ParameterizedFunctionBase<Kokkos::HostSpace>;
#if defined(MPART_ENABLE_GPU)

    template<>
    Eigen::Map<Eigen::VectorXd> ParameterizedFunctionBase<DeviceSpace>::CoeffMap()
    {
        return KokkosToVec(ToHost(this->savedParams));
    }

    template class mpart::ParameterizedFunctionBase<DeviceSpace>;
#endif
