#include <catch2/catch_all.hpp>

#include "MParT/MonotoneComponent.h"
#include "MParT/PositiveBijectors.h"
#include "MParT/Quadrature.h"
#include "MParT/OrthogonalPolynomial.h"
#include "MParT/MultivariateExpansionWorker.h"
#include "MParT/MonotoneIntegrand.h"

#include "MParT/Utilities/ArrayConversions.h"

#include <Eigen/Dense>

using namespace mpart;
using namespace Catch::Matchers;
using HostSpace = Kokkos::HostSpace;

TEST_CASE( "Testing compact monotone component evaluation in 1d", "[MonotoneComponentCompact1d]" ) {

    const double testTol = 1e-7;
    unsigned int dim = 1;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 20;

    Kokkos::View<double**,HostSpace> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        evalPts(0,i) = (i/double(numPts-0.5));

    /* Create and evaluate an affine map
       - Set coefficients so that f(x) = 1.0 + x
       - (f(0) + int_0^x exp( d f(t) ) dt)/(f(0) + int_0^1 exp( d f(t) ) dt ) =  (1 + |x| * exp(1))/(1 + exp(1))
    */
    SECTION("Affine Map"){
        unsigned int maxDegree = 1;
        // This only will be okay for compact map since dim = 1
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);

        MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

        Kokkos::View<double*,HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(0) = 1.0; // Constant term
        coeffs(1) = 1.0; // Linear term

        unsigned int maxSub = 3;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

        MonotoneComponent<MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace>, Exp, AdaptiveSimpson<HostSpace>, HostSpace, true> comp(expansion, quad);

        Kokkos::View<double*,HostSpace> output("output", numPts);
        comp.EvaluateImpl(evalPts, coeffs, output);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK_THAT(output(i), WithinRel( (1+exp(1)*evalPts(0,i))/(1 + exp(1)), testTol));
            bool isInbounds = output(i) < 1. && output(i) > 0.;
            CHECK(isInbounds);
        }
    }

    /* Create and evaluate a quadratic map
       - Set coefficients so that f(x) = 1.0 + x + 0.5*x^2
       - df/dt = 1.0 + t
       - f(0) + int_0^x exp( df/dt ) dt = ( 1.0 + int_0^x exp(1+t) dt )/( 1.0 + int_0^1 exp(1+t) dt) = (1+exp(1+x)-exp(1))/(1+exp(2)-exp(1))
    */
    SECTION("Quadratic Map"){
        unsigned int maxDegree = 2;
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);

        MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

        Kokkos::View<double*,HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol,QuadError::First);

        MonotoneComponent<MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace>, Exp, AdaptiveSimpson<HostSpace>, HostSpace, true> comp(expansion, quad);

        Kokkos::View<double*,HostSpace> output("Output", numPts);
        comp.EvaluateImpl(evalPts, coeffs, output);

        for(unsigned int i=0; i<numPts; ++i){
            double x = evalPts(0,i);
            CHECK_THAT(output(i), WithinRel( (1 + exp(1+x) - exp(1))/(1 + exp(2) - exp(1)) , testTol));
        }
    }
}

TEST_CASE( "Testing bracket-based inversion of compact monotone component", "[CompactMonotoneBracketInverse]" ) {

    const double testTol = 1e-6;
    unsigned int dim = 1;

    // Create points evently space on [lb,ub]
    unsigned int numPts = 20;

    Kokkos::View<double**, HostSpace> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        evalPts(0,i) = (i/double(numPts-1));

    /* Create and evaluate an affine map
       - Set coefficients so that f(x) = 1.0 + x
       - ( f(0) + int_0^x exp( d f(t) ) dt ) / ( f(0) + int_0^1 exp( d f(t) ) dt )=  (1 + x * exp(1))/(1 + exp(1))
    */
    SECTION("Affine Map"){
        unsigned int maxDegree = 1;
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);

        MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

        Kokkos::View<double*, HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(0) = 1.0; // Constant term
        coeffs(1) = 1.0; // Linear term

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

        MonotoneComponent<decltype(expansion), Exp, AdaptiveSimpson<HostSpace>, HostSpace, true> comp(expansion, quad);

        Kokkos::View<double*, HostSpace> ys("ys", numPts);
        comp.EvaluateImpl(evalPts, coeffs, ys);

        Kokkos::View<double*, HostSpace> testInverse("Test output", numPts);
        for(int i = 0; i < 100; i++)
            comp.InverseImpl(evalPts, ys, coeffs, testInverse);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK_THAT(testInverse(i), WithinRel(evalPts(0,i), testTol));
        }
    }

    /* Create and evaluate a quadratic map
       - Set coefficients so that f(x) = 1.0 + x + 0.5*x^2
       - df/dt = 1.0 + t
       - f(0) + int_0^x exp( df/dt ) dt =  1.0 + int_0^x exp(1+t) dt = 1+exp(1+x)
    */
    SECTION("Quadratic Map"){
        unsigned int maxDegree = 2;
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);

        MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

        Kokkos::View<double*, HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol,QuadError::First);

        MonotoneComponent<decltype(expansion), Exp, AdaptiveSimpson<HostSpace>, HostSpace, true> comp(expansion, quad);

        Kokkos::View<double*, HostSpace> ys("ys",numPts);
        comp.EvaluateImpl(evalPts, coeffs,ys);

        Kokkos::View<double*, HostSpace> testInverse("inverse", numPts);
        comp.InverseImpl(evalPts, ys, coeffs, testInverse);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK_THAT(testInverse(i), WithinAbs(evalPts(0,i), testTol));
        }
    }

    SECTION("Same x, multiple ys"){

        Kokkos::View<double**, HostSpace> x("Evaluate Points", dim, 1);
        x(0,0) = 0.5;

        unsigned int maxDegree = 2;
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
        MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

        Kokkos::View<double*, HostSpace> coeffs("Expansion coefficients", mset.Size());
        coeffs(1) = 1.0; // Linear term = x ^1
        coeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
        coeffs(0) = 1.0 + coeffs(2); // Constant term = x^0

        unsigned int maxSub = 30;
        double relTol = 1e-7;
        double absTol = 1e-7;
        AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol,QuadError::First);

        MonotoneComponent<decltype(expansion), Exp, AdaptiveSimpson<HostSpace>, HostSpace, true> comp(expansion, quad);

        Kokkos::View<double*, HostSpace> ys("ys", numPts);
        comp.EvaluateImpl(evalPts, coeffs, ys);

        Kokkos::View<double*, HostSpace> testInverse("inverse", numPts);
        comp.InverseImpl(x, ys, coeffs,testInverse);

        for(unsigned int i=0; i<numPts; ++i){
            CHECK_THAT(testInverse(i), WithinAbs(evalPts(0,i), testTol));
        }
    }
}

TEST_CASE( "Testing compact monotone component derivative", "[CompactMonotoneComponentDerivative]" ) {

    const double testTol = 1e-4;
    unsigned int dim = 2;
    const double fdStep = 1e-4;

    // Create points evently spaced on [lb,ub]
    unsigned int numPts = 20;
    double lb = 1e-3;
    double ub = 1.- 1e-3;

    Kokkos::View<double**, HostSpace> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        evalPts(0,i) = (i/double(numPts-1))*(ub-lb) + lb;
        evalPts(1,i) = evalPts(0,i);
    }

    Kokkos::View<double**, HostSpace> rightEvalPts("Finite difference points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        rightEvalPts(0,i) = evalPts(0,i);
        rightEvalPts(1,i) = evalPts(1,i) + fdStep;
    }

    unsigned int maxDegree = 2;

    // Need nonzero end to midx's for compact since we need 0 to map to 0
    // i.e. T(x,y) = (f(x,0) + int_0^y g(d_y f(x,t)) dt )/(f(x,0) + int_0^1 g(d_y f(x,t)) dt )
    // so T(x,0) = f(x,0)/(f(x,0) + int_0^1 g(d_y f(x,t)) dt)
    // Since we assume support on the entire domain, then we must have f(x,0) == 0
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree, [](MultiIndex m){return m.HasNonzeroEnd();});
    MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

    unsigned int numTerms = mset.Size();

    unsigned int maxSub = 30;
    double relTol = 1e-7;
    double absTol = 1e-7;
    AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol,QuadError::First);

    MonotoneComponent<decltype(expansion), Exp, AdaptiveSimpson<HostSpace>, HostSpace, true> comp(expansion, quad);

    // Create some arbitrary coefficients
    Kokkos::View<double*, HostSpace> coeffs("Expansion coefficients", mset.Size());
    for(unsigned int i=0; i<coeffs.extent(0); ++i)
        coeffs(i) = 0.1*std::cos( 0.01*i );

    Kokkos::View<double*, HostSpace> evals("evals",numPts);
    comp.EvaluateImpl(evalPts, coeffs, evals);
    Kokkos::View<double*, HostSpace> rightEvals("revals", numPts);
    comp.EvaluateImpl(rightEvalPts, coeffs, rightEvals);
    Kokkos::View<double*, HostSpace> contDerivs = comp.ContinuousDerivative(evalPts, coeffs);

    for(unsigned int i=0; i<numPts; ++i){
        double fdDeriv = (rightEvals(i)-evals(i))/fdStep;
        CHECK_THAT( contDerivs(i), WithinRel(fdDeriv, testTol) );
    }
}
//     SECTION("Coefficient Jacobian"){

//         Kokkos::View<double*, HostSpace> evals2("FD Evals", numPts);
//         Kokkos::View<double**, HostSpace> jac("Jacobian", numTerms, numPts);

//         comp.CoeffJacobian(evalPts, coeffs, evals2, jac);

//         for(unsigned int i=0; i<numPts; ++i)
//             CHECK(evals2(i) == Approx(evals(i)).epsilon(1e-12));

//         const double fdStep = 1e-4;

//         for(unsigned j=0; j<numTerms; ++j){
//             coeffs(j) += fdStep;
//             comp.EvaluateImpl(evalPts, coeffs, evals2);

//             for(unsigned int i=0; i<numPts; ++i)
//                 CHECK(jac(j,i) == Approx((evals2(i)-evals(i))/fdStep).epsilon(5e-4).margin(1e-4));

//             coeffs(j) -= fdStep;
//         }
//     }


//      SECTION("Mixed Discrete Jacobian"){

//         const double fdStep = 1e-4;

//         Kokkos::View<double*, HostSpace> derivs("Derivatives", numPts);
//         Kokkos::View<double*, HostSpace> derivs2("Derivatives2", numPts);

//         Kokkos::View<double**, HostSpace> jac("Jacobian", numTerms,numPts);

//         comp.DiscreteMixedJacobian(evalPts, coeffs, jac);
//         derivs = comp.DiscreteDerivative(evalPts, coeffs);

//         // Perturb the coefficients and recompute
//         Kokkos::View<double*, HostSpace> coeffs2("Coefficients2", numTerms);
//         Kokkos::deep_copy(coeffs2, coeffs);

//         for(unsigned int j=0; j<coeffs.extent(0); ++j){
//             coeffs2(j) += fdStep;
//             derivs2 = comp.DiscreteDerivative(evalPts, coeffs2);

//             for(unsigned int i=0; i<derivs2.extent(0); ++i){
//                 CHECK(jac(j,i)==Approx((derivs2(i) - derivs(i))/fdStep).epsilon(1e-4).margin(1e-3));
//             }

//             coeffs2(j) = coeffs(j);
//         }

//     }

//     SECTION("Mixed Continuous Jacobian"){

//         const double fdStep = 1e-4;

//         Kokkos::View<double*, HostSpace> derivs("Derivatives", numPts);
//         Kokkos::View<double*, HostSpace> derivs2("Derivatives2", numPts);

//         Kokkos::View<double**, HostSpace> jac("Jacobian", numTerms,numPts);

//         comp.ContinuousMixedJacobian(evalPts, coeffs, jac);
//         derivs = comp.ContinuousDerivative(evalPts, coeffs);

//         // Perturb the coefficients and recompute
//         Kokkos::View<double*, HostSpace> coeffs2("Coefficients2", numTerms);
//         Kokkos::deep_copy(coeffs2, coeffs);

//         for(unsigned int j=0; j<coeffs.extent(0); ++j){
//             coeffs2(j) += fdStep;
//             derivs2 = comp.ContinuousDerivative(evalPts, coeffs2);

//             for(unsigned int i=0; i<derivs2.extent(0); ++i){
//                 CHECK(jac(j,i)==Approx((derivs2(i) - derivs(i))/fdStep).epsilon(1e-4).margin(1e-3));
//             }

//             coeffs2(j) = coeffs(j);
//         }

//     }

//     SECTION("Input Jacobian"){

//         const double fdStep = 1e-4;

//         Kokkos::View<double*, HostSpace> evals("Evaluations", numPts);
//         Kokkos::View<double*, HostSpace> evals2("Evaluations 2", numPts);

//         Kokkos::View<double**, HostSpace> jac("Jacobian", dim, numPts);

//         comp.InputJacobian(evalPts, coeffs, evals, jac);

//         Kokkos::View<double**, HostSpace> evalPts2("Points2", evalPts.extent(0), evalPts.extent(1));
//         Kokkos::deep_copy(evalPts2, evalPts);

//         for(unsigned int j=0; j<dim; ++j){
//             for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
//                 evalPts2(j,ptInd) += fdStep;

//             comp.EvaluateImpl(evalPts2, coeffs, evals2);

//             for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
//                 CHECK(jac(j,ptInd)==Approx((evals2(ptInd) - evals(ptInd))/fdStep).epsilon(1e-4).margin(1e-3));

//             for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
//                 evalPts2(j,ptInd) = evalPts(j,ptInd);
//         }

//     }

//     SECTION("GradientImpl") {

//             Kokkos::View<double**, HostSpace> evals("Evaluations", 1, numPts);

//             Kokkos::View<double**, HostSpace> sens("Jacobian", dim+1, numPts);
//             REQUIRE_THROWS_AS(comp.GradientImpl(evalPts, sens, evals), std::invalid_argument);

//     }
// }


// TEST_CASE( "Least squares test", "[MonotoneComponentRegression]" ) {

//     unsigned int numPts = 100;
//     Kokkos::View<double**,HostSpace> pts("Training Points", 1,numPts);
//     for(unsigned int i=0; i<numPts; ++i)
//         pts(0,i) = i/(numPts-1.0);


//     Kokkos::View<double*,HostSpace> fvals("Training Vales", numPts);
//     for(unsigned int i=0; i<numPts; ++i)
//         fvals(i) = pts(0,i)*pts(0,i) + pts(0,i);


//     MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(1, 6);
//     MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

//     unsigned int maxSub = 30;
//     double relTol = 1e-3;
//     double absTol = 1e-3;
//     AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

//     MonotoneComponent<decltype(expansion), SoftPlus, AdaptiveSimpson<HostSpace>, HostSpace> comp(expansion, quad);

//     unsigned int numTerms = mset.Size();
//     Kokkos::View<double*,HostSpace> coeffs("Coefficients", numTerms);
//     Kokkos::View<double**,HostSpace> jac("Gradient", numTerms,numPts);
//     Kokkos::View<double*,HostSpace> preds("Predictions", numPts);


//     double objective;

//     Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>> jacMat(&jac(0,0), numTerms,numPts);

//     Eigen::Map<Eigen::VectorXd> predVec(&preds(0), numPts);
//     Eigen::Map<Eigen::VectorXd> obsVec(&fvals(0), numPts);
//     Eigen::Map<Eigen::VectorXd> coeffVec(&coeffs(0), numTerms);

//     Eigen::VectorXd objGrad;

//     for(unsigned int optIt=0; optIt<5; ++optIt){

//         comp.CoeffJacobian(pts, coeffs, preds, jac);

//         objGrad = predVec-obsVec;

//         objective = 0.5*objGrad.squaredNorm();
//         coeffVec -= jacMat.transpose().colPivHouseholderQr().solve(objGrad);
//     }

//     CHECK(objective<1e-3);

// }


// TEST_CASE("Testing MonotoneComponent CoeffGrad and LogDeterminantCoeffGrad", "[MonotoneComponent_CoeffGrad]")
// {
//     //const double testTol = 1e-4;
//     unsigned int dim = 2;

//     // Create points evently spaced on [lb,ub]
//     unsigned int numPts = 20;
//     //double lb = -0.5;
//     //double ub = 0.5;

//     Kokkos::View<double**, HostSpace> evalPts("Evaluate Points", dim, numPts);
//     for(unsigned int i=0; i<numPts; ++i){
//         evalPts(0,i) = 0.3;
//         evalPts(1,i) = 0.5;
//     }

//     unsigned int maxDegree = 3;
//     MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);
//     MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

//     unsigned int maxSub = 20;
//     double relTol = 1e-7;
//     double absTol = 1e-7;
//     AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

//     MonotoneComponent<decltype(expansion), Exp, AdaptiveSimpson<HostSpace>, HostSpace> comp(expansion, quad);

//     Kokkos::View<double*, HostSpace> coeffs("Expansion coefficients", mset.Size());
//     for(unsigned int i=0; i<coeffs.extent(0); ++i)
//         coeffs(i) = 0.1*std::cos( 0.01*i );

//     comp.SetCoeffs(coeffs);

//     SECTION("CoeffGrad"){

//         Kokkos::View<double**, HostSpace> sens("Sensitivity", 1, numPts);
//         for(unsigned int i=0; i<numPts; ++i)
//             sens(0,i) = 0.25*(i+1);

//         Kokkos::View<double**, HostSpace> grads = comp.CoeffGrad(evalPts, sens);
//         REQUIRE(grads.extent(0)==comp.numCoeffs);
//         REQUIRE(grads.extent(1)==numPts);

//         for(unsigned int j=1; j<numPts; ++j){
//             for(unsigned int i=0; i<comp.numCoeffs; ++i){
//                 CHECK(grads(i,j) == (j+1.0)*grads(i,0));
//             }
//         }
//     }

//     SECTION("LogDeterminantCoeffGrad"){

//         for(unsigned int i=0; i<numPts; ++i){
//             evalPts(0,i) = 0.03*i;
//             evalPts(1,i) = -0.05*i;
//         }

//         Kokkos::View<double*, HostSpace> logDets = comp.LogDeterminant(evalPts);
//         Kokkos::View<double*, HostSpace> logDets2;
//         Kokkos::View<double**, HostSpace> grads = comp.LogDeterminantCoeffGrad(evalPts);
//         REQUIRE(grads.extent(0)==comp.numCoeffs);
//         REQUIRE(grads.extent(1)==numPts);

//         // Compare with finite difference derivatives
//         const double fdstep = 1e-4;

//         for(unsigned int i=0; i<coeffs.extent(0); ++i){
//             coeffs(i) += fdstep;

//             comp.SetCoeffs(coeffs);
//             logDets2 = comp.LogDeterminant(evalPts);
//             for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
//                 CHECK( grads(i,ptInd) == Approx((logDets2(ptInd)-logDets(ptInd))/fdstep).epsilon(1e-5));

//             coeffs(i) -= fdstep;
//         }

//     }

//     SECTION("DiagonalCoeffIndices") {
//         std::vector<unsigned int> indices = comp.DiagonalCoeffIndices();
//         std::vector<unsigned int> indices_ref = expansion.NonzeroDiagonalEntries();
//         bool same_indices = indices == indices_ref;
//         REQUIRE(same_indices);
//     }
// }

// #if defined(KOKKOS_ENABLE_CUDA ) || defined(KOKKOS_ENABLE_SYCL)

// TEST_CASE( "MonotoneIntegrand1d on device", "[MonotoneIntegrandDevice]") {

//     typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

//     const double testTol = 1e-7;

//     unsigned int dim = 1;
//     unsigned int maxDegree = 1;
//     FixedMultiIndexSet<HostSpace> hset(dim, maxDegree);
//     FixedMultiIndexSet<DeviceSpace> mset = hset.ToDevice<DeviceSpace>(); // Create a total order limited fixed multindex set

//     MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,DeviceSpace> expansion(mset);

//     MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> hexpansion(hset);

//     // Make room for the cache
//     unsigned int cacheSize = hexpansion.CacheSize();
//     Kokkos::View<double*, DeviceSpace> dcache("device cache", cacheSize);

//     Kokkos::View<double*, HostSpace> hcoeffs("Expansion coefficients", mset.Size());
//     hcoeffs(0) = 1.0; // Constant term
//     hcoeffs(1) = 1.0; // Linear term

//     Kokkos::View<double*, DeviceSpace> dcoeffs = ToDevice<DeviceSpace>(hcoeffs);

//     Kokkos::View<double*, HostSpace> hpt("evaluation point", dim);
//     hpt(0) = 1.0;

//     Kokkos::View<double*, DeviceSpace> dpt = ToDevice<DeviceSpace>(hpt);

//     SECTION("Integrand Only") {
//         Kokkos::View<double*, DeviceSpace> dres("result", 1);

//     	Kokkos::RangePolicy<typename MemoryToExecution<DeviceSpace>::Space> policy(0,1);
//         Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i){
//             MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs), DeviceSpace> integrand(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::None, 0.0);
//             integrand(0.5, &dres(0));
//         });

//         Kokkos::View<double*, HostSpace> hres = ToHost(dres);

//         CHECK(hres(0) == Approx(exp(1)).epsilon(testTol));
//     }

//     SECTION("Integrand Derivative") {
//         Kokkos::View<double*, DeviceSpace> dres("result", 2);

//     	Kokkos::RangePolicy<typename MemoryToExecution<DeviceSpace>::Space> policy(0,1);
//         Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i){
//             MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs), DeviceSpace> integrand(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::Diagonal, 0.0);
//             integrand(0.5, dres.data());
//         });

//         Kokkos::View<double*, HostSpace> hres = ToHost(dres);

//         CHECK(hres(0) == Approx(exp(1)).epsilon(testTol));
//     }

//     SECTION("Integrand Parameters Gradient") {

//         Kokkos::View<double*, DeviceSpace> dres("result", hset.Size());
//         Kokkos::View<double*, DeviceSpace> dres_fd("result_fd", hset.Size());
//         Kokkos::View<double*, DeviceSpace> testVal("integrand", 1+hset.Size());

//     	Kokkos::RangePolicy<typename MemoryToExecution<DeviceSpace>::Space> policy(0,1);
//         Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i){

//             MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs), DeviceSpace> integrand(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::Parameters, 0.0);
//             MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs), DeviceSpace> integrand2(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::None, 0.0);

//             integrand(0.5, testVal.data());

//             const double fdStep = 1e-4;
//             double testVal2;
//             for(unsigned int termInd=0; termInd<dres.extent(0); ++termInd){
//                 dcoeffs(termInd) += fdStep;
//                 integrand2(0.5, &testVal2);
//                 dres_fd(termInd) = (testVal2 - testVal(0))/fdStep;

//                 dres(termInd) = testVal(1 + termInd);
//                 dcoeffs(termInd) -= fdStep;
//             }

//         });

//         Kokkos::View<double*, HostSpace> hres = ToHost(dres);
//         Kokkos::View<double*, HostSpace> hres_fd = ToHost(dres_fd);

//         for(unsigned int termInd=0; termInd<hres.extent(0); ++termInd)
//             CHECK(hres(termInd) == Approx(hres_fd(termInd)).epsilon(1e-4));
//     }


//     SECTION("Integrand Mixed Gradient") {

//         Kokkos::View<double*, DeviceSpace> dres("result", hset.Size());
//         Kokkos::View<double*, DeviceSpace> dres_fd("result_fd", hset.Size());
//         Kokkos::View<double*, DeviceSpace> testVal("integrand", 1+hset.Size());
//         Kokkos::View<double*, DeviceSpace> workspace("workspace", hset.Size());
	
// 	Kokkos::RangePolicy<typename MemoryToExecution<DeviceSpace>::Space> policy(0,1);
//         Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i){

//             MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs), DeviceSpace> integrand(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::MixedCoeff, 0.0, workspace);
//             MonotoneIntegrand<decltype(expansion), Exp, decltype(dpt), decltype(dcoeffs), DeviceSpace> integrand2(dcache.data(), expansion, dpt, dcoeffs, DerivativeFlags::Diagonal, 0.0);

//             integrand(0.5, testVal.data());

//             const double fdStep = 1e-4;
//             double testVal2[2];
//             for(unsigned int termInd=0; termInd<dres.extent(0); ++termInd){
//                 dcoeffs(termInd) += fdStep;
//                 integrand2(0.5, testVal2);
//                 dres_fd(termInd) = (testVal2[0] - testVal(0))/fdStep;
//                 dres(termInd) = testVal(1 + termInd);
//             }
//         });

//         Kokkos::View<double*, HostSpace> hres = ToHost(dres);
//         Kokkos::View<double*, HostSpace> hres_fd = ToHost(dres_fd);

//         for(unsigned int termInd=0; termInd<hres.extent(0); ++termInd)
//             CHECK(hres(termInd) == Approx(hres_fd(termInd)).epsilon(1e-4));

//     }
// }



// TEST_CASE( "Testing MonotoneComponent::EvaluateSingle on Device", "[MonotoneComponentSingle_Device]") {

//     typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

//     unsigned int dim = 2;
//     unsigned int maxDegree = 1;
//     FixedMultiIndexSet<HostSpace> hset(dim,maxDegree);
//     FixedMultiIndexSet<DeviceSpace> dset = hset.ToDevice<DeviceSpace>(); // Create a total order limited fixed multindex set

//     MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,DeviceSpace> dexpansion(dset);

//     // define f(x1,x2) = c0 + c1*x1 + c2*x2
//     Kokkos::View<double*,HostSpace> hcoeffs("Expansion coefficients", hset.Size());
//     hcoeffs(0) = 1.0; // Constant term
//     hcoeffs(1) = 1.0; // Linear term in x1
//     hcoeffs(2) = 1.0; // Linear in x2

//     Kokkos::View<double*,DeviceSpace> dcoeffs = ToDevice<DeviceSpace>(hcoeffs);

//     unsigned int cacheSize = dexpansion.CacheSize();
//     CHECK(cacheSize == (maxDegree+1)*(2*dim+1));

//     // Allocate some memory for the cache
//     Kokkos::View<double*, DeviceSpace> dcache("device cache", cacheSize);

//     Kokkos::View<double*,HostSpace> hpt("host point", dim);
//     hpt(0) = 0.5;
//     hpt(1) = 0.5;
//     Kokkos::View<double*,DeviceSpace> dpt = ToDevice<DeviceSpace>(hpt);

//     unsigned int maxSub = 20;
//     double relTol = 1e-5;
//     double absTol = 1e-5;
//     AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

//     Kokkos::View<double*,DeviceSpace> workspace("quadrature workspace", quad.WorkspaceSize());

//     Kokkos::View<double*, DeviceSpace> dres("Device Evaluation", 1);
//     // Run the fill cache funciton, using a parallel_for loop to ensure it's run on the device
//     Kokkos::RangePolicy<typename MemoryToExecution<DeviceSpace>::Space> policy(0,1);
//     Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i){
//         dexpansion.FillCache1(dcache.data(), dpt, DerivativeFlags::None);
//         dres(0) = MonotoneComponent<decltype(dexpansion),Exp, decltype(quad), DeviceSpace>::EvaluateSingle(dcache.data(), workspace.data(), dpt, dpt(dim-1), dcoeffs, quad, dexpansion);
//     });

//     Kokkos::fence();

//     CHECK(ToHost(dres)(0) == Approx(hcoeffs(0) + hcoeffs(1)*hpt(0) + hpt(1)*exp(hcoeffs(2))).epsilon(1e-4));

// }


// TEST_CASE( "Testing 1d monotone component evaluation on device", "[MonotoneComponent1d_Device]" ) {

//     typedef Kokkos::DefaultExecutionSpace::memory_space DeviceSpace;

//     const double testTol = 1e-7;
//     unsigned int dim = 1;

//     // Create points evently space on [lb,ub]
//     unsigned int numPts = 3;
//     double lb = -2.0;
//     double ub = 2.0;

//     Kokkos::View<double**,DeviceSpace>::HostMirror hevalPts("Evaluation Points", dim, numPts);
//     for(unsigned int i=0; i<numPts; ++i)
//         hevalPts(0,i) = (i/double(numPts-1))*(ub-lb) - lb;

//     Kokkos::View<double**,DeviceSpace> devalPts = ToDevice<DeviceSpace>(hevalPts);


//     /* Create and evaluate an affine map
//        - Set coefficients so that f(x) = 1.0 + x
//        - f(0) + int_0^x exp( d f(t) ) dt =  1.0 + int_0^x exp(1) dt = 1 + |x| * exp(1)
//     */
//     SECTION("Affine Map"){
//         unsigned int maxDegree = 1;
//         FixedMultiIndexSet<HostSpace> hset(dim, maxDegree);
//         FixedMultiIndexSet<DeviceSpace> mset = hset.ToDevice<DeviceSpace>();

//         MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,DeviceSpace> expansion(mset);

//         Kokkos::View<double*,HostSpace> hcoeffs("Expansion coefficients", mset.Size());
//         hcoeffs(0) = 1.0; // Constant term
//         hcoeffs(1) = 1.0; // Linear term

//         Kokkos::View<double*,DeviceSpace> dcoeffs = ToDevice<DeviceSpace>(hcoeffs);

//         unsigned int maxSub = 10;
//         double relTol = 1e-6;
//         double absTol = 1e-6;
//         AdaptiveSimpson<DeviceSpace> quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

//         MonotoneComponent<decltype(expansion),Exp, decltype(quad),DeviceSpace> comp(expansion, quad);

//         Kokkos::View<double*,DeviceSpace> doutput("dout", numPts);
//         comp.EvaluateImpl(devalPts, dcoeffs, doutput);
//         auto houtput = ToHost(doutput);

//         for(unsigned int i=0; i<numPts; ++i){
//             CHECK(houtput(i) == Approx(1+exp(1)*std::abs(hevalPts(0,i))).epsilon(testTol));
//         }
//     }


//     /* Create and evaluate a quadratic map
//        - Set coefficients so that f(x) = 1.0 + x + 0.5*x^2
//        - df/dt = 1.0 + t
//        - f(0) + int_0^x exp( df/dt ) dt =  1.0 + int_0^x exp(1+t) dt = 1+exp(1+x)
//     */
//     SECTION("Quadratic Map"){
//         unsigned int maxDegree = 2;

//         FixedMultiIndexSet<HostSpace> hset(dim, maxDegree);
//         FixedMultiIndexSet<DeviceSpace> mset = hset.ToDevice<DeviceSpace>();

//         MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,DeviceSpace> expansion(mset);

//         Kokkos::View<double*,HostSpace> hcoeffs("Expansion coefficients", mset.Size());
//         hcoeffs(1) = 1.0; // Linear term = x ^1
//         hcoeffs(2) = 0.5; // Quadratic term = x^2 - 1.0
//         hcoeffs(0) = 1.0 + hcoeffs(2); // Constant term = x^0

//         Kokkos::View<double*,DeviceSpace> dcoeffs = ToDevice<DeviceSpace>(hcoeffs);

//         unsigned int maxSub = 10;
//         double relTol = 1e-6;
//         double absTol = 1e-6;
//         AdaptiveSimpson<DeviceSpace> quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

//         MonotoneComponent<decltype(expansion), Exp, decltype(quad), DeviceSpace> comp(expansion, quad);

//         Kokkos::View<double*,DeviceSpace> doutput("dout",numPts);
//         comp.EvaluateImpl(devalPts, dcoeffs, doutput);
//         auto houtput = ToHost(doutput);

//         for(unsigned int i=0; i<numPts; ++i){
//             CHECK(houtput(i) == Approx(1+exp(1)*(exp(hevalPts(0,i))-1)).epsilon(testTol));
//         }
//     }

// }

// #endif
