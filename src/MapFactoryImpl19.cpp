#include "MParT/MapFactory.h"

#include "MParT/MonotoneComponent.h"
#include "MParT/TriangularMap.h"
#include "MParT/Quadrature.h"

#include "MParT/UnivariateBases.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/PositiveBijectors.h"

using namespace mpart;

using BasisType = Kokkos::pair<SineBasis, ShiftedLegendre>;

template<typename MemorySpace, typename PosFuncType>
std::shared_ptr<ConditionalMapBase<MemorySpace>> CreateComponentImpl_SL_CC(FixedMultiIndexSet<MemorySpace> const& mset, MapOptions opts)
{
    BasisEvaluator<BasisHomogeneity::OffdiagHomogeneous,BasisType,Identity> basis1d(mset.Length());
    ClenshawCurtisQuadrature<MemorySpace> quad(opts.quadPts, 1);

    MultivariateExpansionWorker<decltype(basis1d),MemorySpace> expansion(mset, basis1d);
    std::shared_ptr<ConditionalMapBase<MemorySpace>> output;

    output = std::make_shared<MonotoneComponent<decltype(expansion), PosFuncType, decltype(quad), MemorySpace, true>>(expansion, quad, opts.contDeriv, opts.nugget);

    Kokkos::View<const double*,MemorySpace> coeffs = Kokkos::View<double*,MemorySpace>("Component Coefficients", mset.Size());
    output->SetCoeffs(coeffs);

    return output;
}

static auto reg_host_sl_cc_exp_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::SinusoidLegendre, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis, true), CreateComponentImpl_SL_CC<Kokkos::HostSpace, Exp>));
static auto reg_host_sl_cc_splus_compact = mpart::MapFactory::CompFactoryImpl<Kokkos::HostSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::SinusoidLegendre, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis, true), CreateComponentImpl_SL_CC<Kokkos::HostSpace, SoftPlus>));
#if defined(MPART_ENABLE_GPU)
    static auto reg_device_sl_cc_exp_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::SinusoidLegendre, false, PosFuncTypes::Exp, QuadTypes::ClenshawCurtis, true), CreateComponentImpl_SL_CC<mpart::DeviceSpace, Exp>));
    static auto reg_device_sl_cc_splus_compact = mpart::MapFactory::CompFactoryImpl<mpart::DeviceSpace>::GetFactoryMap()->insert(std::make_pair(std::make_tuple(BasisTypes::SinusoidLegendre, false, PosFuncTypes::SoftPlus, QuadTypes::ClenshawCurtis, true), CreateComponentImpl_SL_CC<mpart::DeviceSpace, SoftPlus>));
#endif

#if defined(MPART_HAS_CEREAL)
REGISTER_OFFDIAGHOMOGENEOUS_MONO_COMP(SineBasis, ShiftedLegendre, Identity, Exp, ClenshawCurtisQuadrature, Kokkos::HostSpace, true)
REGISTER_OFFDIAGHOMOGENEOUS_MONO_COMP(SineBasis, ShiftedLegendre, Identity, SoftPlus, ClenshawCurtisQuadrature, Kokkos::HostSpace, true)
#if defined(MPART_ENABLE_GPU)
REGISTER_OFFDIAGHOMOGENEOUS_MONO_COMP(SineBasis, ShiftedLegendre, Identity, Exp, ClenshawCurtisQuadrature, mpart::DeviceSpace, true)
REGISTER_OFFDIAGHOMOGENEOUS_MONO_COMP(SineBasis, ShiftedLegendre, Identity, SoftPlus, ClenshawCurtisQuadrature, mpart::DeviceSpace, true)
#endif
CEREAL_REGISTER_DYNAMIC_INIT(mpartInitMapFactory19)
#endif