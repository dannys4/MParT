#include "MParT/MapFactory.h"

#include "MParT/MultiIndices/MultiIndexLimiter.h"
#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/SummarizedMap.h"
#include "MParT/AffineFunction.h"
#include "MParT/IdentityMap.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/HermiteFunction.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/RectifiedMultivariateExpansion.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/LinearizedBasis.h"
#include "MParT/Sigmoid.h"
#include "MParT/UnivariateExpansion.h"

using namespace mpart;


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateComponent(FixedMultiIndexSet<MemorySpace> const& mset,
                                                           MapOptions                                   opts)
{
    return CompFactoryImpl<MemorySpace>::GetFactoryFunction(opts)(mset,opts);
}


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int dim,
                                                                                         unsigned int activeInd,
                                                                                         std::shared_ptr<ConditionalMapBase<MemorySpace>> const &comp)
{

    // Check that active index is not greater than map dimension
    if(dim < activeInd){
        std::stringstream msg;
        msg << "In CreateSingleEntryMap, the active index can't be greater than map dimension. Got dim = " << dim << " and activeInd = " << activeInd << ".";
        throw std::invalid_argument(msg.str());
    }

    // Check that the input dimension of the component matches the activeInd
    if(activeInd != comp->inputDim){
        std::stringstream msg;
        msg << "In CreateSingleEntryMap, the component input dimension must be equal to the active index. Got dim = " << comp->inputDim << " and activeInd = " << activeInd << ".";
        throw std::invalid_argument(msg.str());
    }

    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;
    // Construct map using TriangularMap constructor

    if(activeInd == 1){  // special case if activeInd = 1, map is of form [T_1; Id]

        // Bottom identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> botIdMap = std::make_shared<IdentityMap<MemorySpace>>(dim, dim-activeInd);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(2);
        blocks.at(0) = comp;
        blocks.at(1) = botIdMap;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);


    }
    else if (activeInd == dim){  // special case if activeInd = dim, map is of form [Id; T_d]
        // Top identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> topIdMap = std::make_shared<IdentityMap<MemorySpace>>(activeInd-1, activeInd-1);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(2);
        blocks.at(0) = topIdMap;
        blocks.at(1) = comp;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);
    }
    else{ // general case, map is of form [Id; T_i; Id]

        // Top identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> topIdMap = std::make_shared<IdentityMap<MemorySpace>>(activeInd-1, activeInd-1);

        // Bottom identity map
        std::shared_ptr<ConditionalMapBase<MemorySpace>> botIdMap = std::make_shared<IdentityMap<MemorySpace>>(dim, dim-activeInd);

        // fill a vector of components with identity, active component, identity
        std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> blocks(3);
        blocks.at(0) = topIdMap;
        blocks.at(1) = comp;
        blocks.at(2) = botIdMap;

        // make map
        output = std::make_shared<TriangularMap<MemorySpace>>(blocks);

    }
    Kokkos::View<const double*, MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs);
    output->SetCoeffs(coeffs);
    return output;

}


template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> mpart::MapFactory::CreateTriangular(unsigned int inputDim,
                                                                         unsigned int outputDim,
                                                                         unsigned int totalOrder,
                                                                         MapOptions options)
{

    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>>> comps(outputDim);

    unsigned int extraInputs = inputDim - outputDim;

    for(unsigned int i=0; i<outputDim; ++i){
        FixedMultiIndexSet<Kokkos::HostSpace> mset(i+extraInputs+1, totalOrder);
        comps.at(i) = CreateComponent<MemorySpace>(mset.ToDevice<MemorySpace>(), options);
    }
    auto output = std::make_shared<TriangularMap<MemorySpace>>(comps);
    
    Kokkos::View<const double*, MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs);
    output->SetCoeffs(coeffs);
    return output;
}


template<typename MemorySpace>
std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> mpart::MapFactory::CreateExpansion(unsigned int outputDim,
                                                                                           FixedMultiIndexSet<MemorySpace> const& mset,
                                                                                           MapOptions                                   opts)
{
    std::shared_ptr<ParameterizedFunctionBase<MemorySpace>> output;

    if(opts.basisType==BasisTypes::ProbabilistHermite){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite> basis1d(opts.basisNorm);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }else{
            BasisEvaluator<BasisHomogeneity::Homogeneous,LinearizedBasis<ProbabilistHermite>> basis1d(LinearizedBasis(ProbabilistHermite(opts.basisNorm), opts.basisLB, opts.basisUB));
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }else if(opts.basisType==BasisTypes::PhysicistHermite){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            BasisEvaluator<BasisHomogeneity::Homogeneous,PhysicistHermite> basis1d(opts.basisNorm);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }else{
            BasisEvaluator<BasisHomogeneity::Homogeneous,LinearizedBasis<PhysicistHermite>> basis1d(PhysicistHermite(opts.basisNorm), opts.basisLB, opts.basisUB);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }else if(opts.basisType==BasisTypes::HermiteFunctions){

        if(isinf(opts.basisLB) && isinf(opts.basisUB)){
            BasisEvaluator<BasisHomogeneity::Homogeneous,HermiteFunction> basis1d;
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }else{
            BasisEvaluator<BasisHomogeneity::Homogeneous,LinearizedBasis<HermiteFunction>> basis1d(opts.basisLB, opts.basisUB);
            output = std::make_shared<MultivariateExpansion<decltype(basis1d), MemorySpace>>(outputDim, mset, basis1d);
        }
    }

    if(output){
        Kokkos::View<const double*, MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs);
    	output->SetCoeffs(coeffs);
        return output;
    }

    std::stringstream msg;
    msg << "Could not parse options in CreateExpansion.  Unknown 1d basis type.";
    throw std::runtime_error(msg.str());

    return nullptr;
}

/** Constructs a sigmoid basis from specified centers.  Chooses the widths of the 
 *  sigmoid functions using a rule of thumb based on the distance between neighboring
 *  centers.
*/
template <typename MemorySpace, typename SigmoidType, typename EdgeType>
Sigmoid1d<MemorySpace, SigmoidType, EdgeType> CreateSigmoid(unsigned int inputDim, 
                                                            StridedVector<double, MemorySpace> centers, 
                                                            double edgeShape,
                                                            SigmoidSumSizeType sumType) 
{
    int numSigmoids = Sigmoid1d<MemorySpace, SigmoidType, EdgeType>::ComputeNumSigmoids(centers.size(), sumType);
    
    // Fill vectors on host and then copy to device if necessary
    Kokkos::View<double*, Kokkos::HostSpace> hWidths("Widths", centers.extent(0));
    Kokkos::View<double*, Kokkos::HostSpace> hWeights("Weights", centers.extent(0));
    Kokkos::View<double*, Kokkos::HostSpace> hCenters("Centers", centers.extent(0));
    Kokkos::deep_copy(hCenters, centers);
    Kokkos::deep_copy(hWeights, 1.0);

    // Set widths and weights for edge terms
    hWidths(0) = edgeShape;
    hWeights(0) = 1.0/edgeShape;
    hWidths(1) = edgeShape;
    hWeights(1) = 1.0/edgeShape;
    
    // Now set sigmoid widths/weights
    if(sumType==SigmoidSumSizeType::Linear){
        
        for(int i=0; i<numSigmoids; i++){
            int start_idx = 2+(i*(i+1))/2;

            for(unsigned int j = 0; j < i; j++) {
                double prev_center, next_center;
                if(j == 0 || i == 0) {// Use center for left edge term
                    prev_center = centers(0);
                } else {
                    prev_center = centers(start_idx+j-1);
                }
                if(j == i-1 || i == 0) { // Use center for right edge term
                    next_center = centers(1);
                } else {
                    next_center = centers(start_idx+j);
                }
                hWidths(start_idx + j) = 2/(next_center - prev_center);
                hWeights(start_idx + j) = 1.;
            }
        }
    }else{
        if(numSigmoids>0){
            // Make sure the non-edge centers are sorted
            std::sort(&hCenters(2), &hCenters(hCenters.extent(0)-1)+1);

            // Set first sigmoid
            if(numSigmoids>1){
                hWidths(2) = 4.0/(hCenters(3)-hCenters(2)); // Second sigmoid center minus first
            }else{
                hWidths(2) = 8.0/std::abs(hCenters(1)-hCenters(0)); // Distance between edge centers
            }

            // Set "interior" sigmoids
            for(int i=1; i<numSigmoids-1; i++){
                hWidths[2+i] = 8.0/(hCenters[3+i]-hCenters[1+i]);
            }

            // If there is more than one sigmoid, set the last one
            if(numSigmoids>1){
                hWidths(2+numSigmoids-1) = 4.0/(hCenters(2+numSigmoids-1)-hCenters(2+numSigmoids-2));
            }
        }
    }

    Kokkos::View<double*, MemorySpace> widths = Kokkos::create_mirror_view_and_copy(MemorySpace(), hWidths);
    Kokkos::View<double*, MemorySpace> weights = Kokkos::create_mirror_view_and_copy(MemorySpace(), hWeights);
    Kokkos::View<double*, MemorySpace> dCenters = Kokkos::create_mirror_view_and_copy(MemorySpace(), hCenters);
    
    Sigmoid1d<MemorySpace, SigmoidType, EdgeType> sig {dCenters, widths, weights, sumType};
    return sig;  
}

template<typename MemorySpace, typename OffdiagEval, typename Rectifier, typename SigmoidType, typename EdgeType>
using SigmoidBasisEval = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<OffdiagEval, Sigmoid1d<MemorySpace, SigmoidType, EdgeType>>>;

template <typename MemorySpace, typename OffdiagEval, typename Rectifier, typename SigmoidType, typename EdgeType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSigmoidExpansionTemplate(FixedMultiIndexSet<MemorySpace> mset_diag,
                                                                                StridedVector<double, MemorySpace> centers, 
                                                                                double edgeWidth,
                                                                                SigmoidSumSizeType sumType)
{
    unsigned int inputDim = mset_diag.Length();
    using Sigmoid_T = Sigmoid1d<MemorySpace, SigmoidType, EdgeType>;
    using Eval_T = BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous, Kokkos::pair<OffdiagEval, Sigmoid_T>, Rectifier>;
    auto sigmoid = CreateSigmoid<MemorySpace, SigmoidType, EdgeType>(inputDim, centers, edgeWidth, sumType);
    if(inputDim == 1) {
        unsigned int maxOrder = mset_diag.Size() - 1;
        auto output = std::make_shared<UnivariateExpansion<MemorySpace, Sigmoid_T>>(maxOrder, sigmoid);
        output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
        return output;
    }
    Eval_T BasisEval(inputDim, OffdiagEval(), sigmoid);
    using RMVE = RectifiedMultivariateExpansion<MemorySpace, OffdiagEval, Sigmoid_T, Rectifier>;
    unsigned int maxOrder = BasisEval.diag_.GetOrder();
    MultivariateExpansionWorker<Eval_T,MemorySpace> worker(mset_diag, BasisEval);
    auto output = std::make_shared<RMVE>(worker);
    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;
}

template <typename MemorySpace, typename OffdiagEval, typename Rectifier, typename SigmoidType, typename EdgeType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateSigmoidExpansionTemplate(unsigned int inputDim, 
                                                                                unsigned int totalOrder, 
                                                                                unsigned int crossOrder,
                                                                                StridedVector<double, MemorySpace> centers, 
                                                                                double edgeWidth,
                                                                                SigmoidSumSizeType sumType)
{
    using Sigmoid_T = Sigmoid1d<MemorySpace, SigmoidType, EdgeType>;
    int numSigmoids = Sigmoid_T::ComputeNumSigmoids(centers.size(), sumType);

    if(inputDim == 1) {
        FixedMultiIndexSet<Kokkos::HostSpace> fmset_h(1, numSigmoids+3);
        FixedMultiIndexSet<MemorySpace> fmset_d = fmset_h.ToDevice<MemorySpace>();
        return CreateSigmoidExpansionTemplate<MemorySpace, OffdiagEval, Rectifier, SigmoidType, EdgeType>(
            fmset_d, centers, edgeWidth, sumType);
    }

    // Build multiindex with off diagonal terms
    FixedMultiIndexSet<Kokkos::HostSpace> moff1(inputDim-1, totalOrder);
    FixedMultiIndexSet<Kokkos::HostSpace> mdiag1(1, 0);
    FixedMultiIndexSet<Kokkos::HostSpace> m1 = moff1.Cartesian(mdiag1);

    // Create terms that mix sigmoid and off diagonal terms
    FixedMultiIndexSet<Kokkos::HostSpace> moff2(inputDim-1, crossOrder);
    FixedMultiIndexSet<Kokkos::HostSpace> mdiag2(1, numSigmoids+3, 1); // Do not include "order 0" sigmoid (i.e., constant) because that's already in m1
    FixedMultiIndexSet<Kokkos::HostSpace> m2 = moff2.Cartesian(mdiag2);
    
    // Concatenate the off-diagonal only and mixed terms
    FixedMultiIndexSet<Kokkos::HostSpace> fmset_h = m1.Concatenate(m2);

    FixedMultiIndexSet<MemorySpace> fmset_d = fmset_h.ToDevice<MemorySpace>();
    return CreateSigmoidExpansionTemplate<MemorySpace, OffdiagEval, Rectifier, SigmoidType, EdgeType>(
        fmset_d, centers, edgeWidth, sumType);
}

template<typename MemorySpace>
void HandleSigmoidComponentErrors(MapOptions const& opts) {
    // Check that the opts are valid
    if (opts.basisType != BasisTypes::HermiteFunctions) {
        std::string basisString = MapOptions::btypes[static_cast<unsigned int>(opts.basisType)];
        std::stringstream ss;
        ss << "Unsupported basis type for sigmoid expansion: " << basisString;
        ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
    }
    if(opts.posFuncType != PosFuncTypes::Exp && opts.posFuncType != PosFuncTypes::SoftPlus) {
        std::string posString = MapOptions::pftypes[static_cast<unsigned int>(opts.posFuncType)];
        std::stringstream ss;
        ss << "Unsupported positive function type for sigmoid expansion: " << posString;
        ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
    }
    if(opts.edgeType != EdgeTypes::SoftPlus) {
        std::string edgeString = MapOptions::etypes[static_cast<unsigned int>(opts.edgeType)];
        std::stringstream ss;
        ss << "Unsupported edge type for sigmoid expansion: " << edgeString;
        ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
    }
    if(opts.sigmoidType != SigmoidTypes::Logistic) {
        std::string sigmoidString = MapOptions::stypes[static_cast<unsigned int>(opts.sigmoidType)];
        std::stringstream ss;
        ss << "Unsupported sigmoid type for sigmoid expansion: " << sigmoidString;
        ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
    }
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> MapFactory::CreateSigmoidComponent(
            unsigned int inputDim, 
            unsigned int offDiagOrder, 
            unsigned int crossOrder,
            StridedVector<const double, MemorySpace> centers, 
            MapOptions opts) 
{
    HandleSigmoidComponentErrors<MemorySpace>(opts);
    Kokkos::View<double*, MemorySpace> centers_copy("Centers Copy", centers.size());
    Kokkos::deep_copy(centers_copy, centers);
    // Dispatch to the correct sigmoid expansion template
    if(opts.posFuncType == PosFuncTypes::Exp) {
        return CreateSigmoidExpansionTemplate<MemorySpace, HermiteFunction, Exp, SigmoidTypeSpace::Logistic, SoftPlus>(inputDim, offDiagOrder, crossOrder, centers_copy, opts.edgeShape, opts.sigmoidBasisSumType);
    } else if(opts.posFuncType == PosFuncTypes::SoftPlus) {
        return CreateSigmoidExpansionTemplate<MemorySpace, HermiteFunction, SoftPlus, SigmoidTypeSpace::Logistic, SoftPlus>(inputDim, offDiagOrder, crossOrder, centers_copy, opts.edgeShape, opts.sigmoidBasisSumType);
    }
    else {
        return nullptr;
    }
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> MapFactory::CreateSigmoidComponent(
            unsigned int inputDim, 
            unsigned int totalOrder,
            StridedVector<const double, MemorySpace> centers, 
            MapOptions opts) 
{
    return MapFactory::CreateSigmoidComponent(inputDim, totalOrder, totalOrder, centers, opts);
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> MapFactory::CreateSigmoidComponent(
    FixedMultiIndexSet<MemorySpace> mset_diag,
    StridedVector<const double, MemorySpace> centers, MapOptions opts) {
    HandleSigmoidComponentErrors<MemorySpace>(opts);
    Kokkos::View<double*, MemorySpace> centers_copy("Centers Copy", centers.size());
    Kokkos::deep_copy(centers_copy, centers);
    // Dispatch to the correct sigmoid expansion template
    if(opts.posFuncType == PosFuncTypes::Exp) {
        return CreateSigmoidExpansionTemplate<MemorySpace, HermiteFunction, Exp, SigmoidTypeSpace::Logistic, SoftPlus>(mset_diag, centers_copy, opts.edgeShape, opts.sigmoidBasisSumType);
    } else if(opts.posFuncType == PosFuncTypes::SoftPlus) {
        return CreateSigmoidExpansionTemplate<MemorySpace, HermiteFunction, SoftPlus, SigmoidTypeSpace::Logistic, SoftPlus>(mset_diag, centers_copy, opts.edgeShape, opts.sigmoidBasisSumType);
    }
    else {
        return nullptr;
    }
}

template<typename MemorySpace>
std::shared_ptr<ConditionalMapBase<MemorySpace>> MapFactory::CreateSigmoidTriangular(unsigned int inputDim,
    unsigned int outputDim, unsigned int totalOrder, std::vector<StridedVector<const double, MemorySpace>> const& centers, MapOptions opts) {
    if(outputDim > inputDim) {
        std::stringstream ss;
        ss << "CreateSigmoidTriangular: Output dimension " << outputDim << " cannot be greater than input dimension " << inputDim;
        ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
    }
    std::vector<std::shared_ptr<ConditionalMapBase<MemorySpace>> > comps(outputDim);
    for(int i = 0; i < outputDim; i++) {
        StridedVector<const double, MemorySpace> center_view = centers[i];
        unsigned int inputDim_i = (inputDim - outputDim) + i+1;
        auto comp = CreateSigmoidComponent<MemorySpace>(inputDim_i, totalOrder, center_view, opts);
        comps[i] = comp;
    }
    auto output = std::make_shared<TriangularMap<MemorySpace>>(comps);
    output->SetCoeffs(Kokkos::View<double*,MemorySpace>("Component Coefficients", output->numCoeffs));
    return output;
}


template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateComponent<Kokkos::HostSpace>(FixedMultiIndexSet<Kokkos::HostSpace> const&, MapOptions);
template std::shared_ptr<ParameterizedFunctionBase<Kokkos::HostSpace>> mpart::MapFactory::CreateExpansion<Kokkos::HostSpace>(unsigned int, FixedMultiIndexSet<Kokkos::HostSpace> const&, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateTriangular<Kokkos::HostSpace>(unsigned int, unsigned int, unsigned int, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int, unsigned int, std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> const&);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(unsigned int, unsigned int, StridedVector<const double, Kokkos::HostSpace>, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(unsigned int, unsigned int, unsigned int, StridedVector<const double, Kokkos::HostSpace>, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(unsigned int, unsigned int, Eigen::Ref<const Eigen::RowVectorXd>, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(unsigned int, unsigned int, unsigned int, Eigen::Ref<const Eigen::RowVectorXd>, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(FixedMultiIndexSet<Kokkos::HostSpace>, StridedVector<const double, Kokkos::HostSpace>, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidComponent<Kokkos::HostSpace>(FixedMultiIndexSet<Kokkos::HostSpace>, Eigen::Ref<const Eigen::RowVectorXd>, MapOptions);
template std::shared_ptr<ConditionalMapBase<Kokkos::HostSpace>> mpart::MapFactory::CreateSigmoidTriangular<Kokkos::HostSpace>(unsigned int, unsigned int, unsigned int, std::vector<StridedVector<const double, Kokkos::HostSpace>> const&, MapOptions);
#if defined(MPART_ENABLE_GPU)
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateComponent<DeviceSpace>(FixedMultiIndexSet<DeviceSpace> const&, MapOptions);
    template std::shared_ptr<ParameterizedFunctionBase<DeviceSpace>> mpart::MapFactory::CreateExpansion<DeviceSpace>(unsigned int, FixedMultiIndexSet<DeviceSpace> const&, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateTriangular<DeviceSpace>(unsigned int, unsigned int, unsigned int, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSingleEntryMap(unsigned int, unsigned int, std::shared_ptr<ConditionalMapBase<DeviceSpace>> const&);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidComponent<DeviceSpace>(unsigned int, unsigned int, StridedVector<const double, DeviceSpace>, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidComponent<DeviceSpace>(unsigned int, unsigned int, unsigned int, StridedVector<const double, DeviceSpace>, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidComponent<DeviceSpace>(unsigned int, unsigned int, Eigen::Ref<const Eigen::RowVectorXd>, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidComponent<DeviceSpace>(unsigned int, unsigned int, unsigned int, Eigen::Ref<const Eigen::RowVectorXd>, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidComponent<DeviceSpace>(FixedMultiIndexSet<DeviceSpace>, StridedVector<const double, DeviceSpace>, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidComponent<DeviceSpace>(FixedMultiIndexSet<DeviceSpace>, Eigen::Ref<const Eigen::RowVectorXd>, MapOptions);
    template std::shared_ptr<ConditionalMapBase<DeviceSpace>> mpart::MapFactory::CreateSigmoidTriangular<DeviceSpace>(unsigned int, unsigned int, unsigned int, std::vector<StridedVector<const double, DeviceSpace>> const&, MapOptions);
#endif
