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
    double lb = 0.;
    double ub = 1.- fdStep;

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

    SECTION("Coefficient Jacobian"){

        Kokkos::View<double*, HostSpace> evals2("FD Evals", numPts);
        Kokkos::View<double**, HostSpace> jac("Jacobian", numTerms, numPts);

        comp.CoeffJacobian(evalPts, coeffs, evals2, jac);

        for(unsigned int i=0; i<numPts; ++i)
            CHECK_THAT(evals2(i), WithinAbs(evals(i), 1e-12));

        const double fdStep = 1e-4;

        for(unsigned j=0; j<numTerms; ++j){
            coeffs(j) += fdStep;
            comp.EvaluateImpl(evalPts, coeffs, evals2);

            for(unsigned int i=0; i<numPts; ++i)
                CHECK_THAT(jac(j,i), WithinRel((evals2(i)-evals(i))/fdStep, 1e-4));

            coeffs(j) -= fdStep;
        }
    }

    SECTION("Mixed Continuous Jacobian"){

        const double fdStep = 1e-5;

        Kokkos::View<double*, HostSpace> derivs("Derivatives", numPts);
        Kokkos::View<double*, HostSpace> derivs2("Derivatives2", numPts);

        Kokkos::View<double**, HostSpace> jac("Jacobian", numTerms,numPts);

        comp.ContinuousMixedJacobian(evalPts, coeffs, jac);
        derivs = comp.ContinuousDerivative(evalPts, coeffs);

        // Perturb the coefficients and recompute
        Kokkos::View<double*, HostSpace> coeffs2("Coefficients2", numTerms);
        Kokkos::deep_copy(coeffs2, coeffs);

        for(unsigned int j=0; j<coeffs.extent(0); ++j){
            coeffs2(j) += fdStep;
            derivs2 = comp.ContinuousDerivative(evalPts, coeffs2);

            for(unsigned int i=0; i<derivs2.extent(0); ++i){
                CHECK_THAT(jac(j,i), WithinRel((derivs2(i) - derivs(i))/fdStep, 25*fdStep));
            }

            coeffs2(j) = coeffs(j);
        }

    }

    SECTION("Input Jacobian"){

        const double fdStep = 1e-4;

        Kokkos::View<double*, HostSpace> evals("Evaluations", numPts);
        Kokkos::View<double*, HostSpace> evals2("Evaluations 2", numPts);

        Kokkos::View<double**, HostSpace> jac("Jacobian", dim, numPts);

        comp.InputJacobian(evalPts, coeffs, evals, jac);

        Kokkos::View<double**, HostSpace> evalPts2("Points2", evalPts.extent(0), evalPts.extent(1));
        Kokkos::deep_copy(evalPts2, evalPts);

        for(unsigned int j=0; j<dim; ++j){
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                evalPts2(j,ptInd) += fdStep;

            comp.EvaluateImpl(evalPts2, coeffs, evals2);

            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd) {
                CHECK_THAT(jac(j,ptInd), WithinRel((evals2(ptInd) - evals(ptInd))/fdStep, 20*fdStep));
            }

            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                evalPts2(j,ptInd) = evalPts(j,ptInd);
        }

    }

    SECTION("DiscreteDerivative") {
            REQUIRE_THROWS_AS(comp.DiscreteDerivative(evalPts, coeffs), std::invalid_argument);
    }
    SECTION("GradientImpl") {

            Kokkos::View<double**, HostSpace> evals("Evaluations", 1, numPts);

            Kokkos::View<double**, HostSpace> sens("Jacobian", dim+1, numPts);
            REQUIRE_THROWS_AS(comp.GradientImpl(evalPts, sens, evals), std::invalid_argument);

    }
}


TEST_CASE( "Compact least squares test", "[CompactMonotoneComponentRegression]" ) {

    unsigned int numPts = 100;
    Kokkos::View<double**,HostSpace> pts("Training Points", 1,numPts);
    for(unsigned int i=0; i<numPts; ++i)
        pts(0,i) = i/(numPts-1.0);


    Kokkos::View<double*,HostSpace> fvals("Training Values", numPts);
    for(unsigned int i=0; i<numPts; ++i)
        fvals(i) = (pts(0,i)*pts(0,i) + pts(0,i))/2; // Bijection on [0,1]


    // Don't need limiter for compact in one dimension
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(1, 6);
    MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

    unsigned int maxSub = 30;
    double relTol = 1e-3;
    double absTol = 1e-3;
    AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

    MonotoneComponent<decltype(expansion), SoftPlus, AdaptiveSimpson<HostSpace>, HostSpace, true> comp(expansion, quad);

    unsigned int numTerms = mset.Size();
    Kokkos::View<double*,HostSpace> coeffs("Coefficients", numTerms);
    Kokkos::View<double**,HostSpace> jac("Gradient", numTerms,numPts);
    Kokkos::View<double*,HostSpace> preds("Predictions", numPts);


    double objective;

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic, Eigen::RowMajor>> jacMat(&jac(0,0), numTerms,numPts);

    Eigen::Map<Eigen::VectorXd> predVec(&preds(0), numPts);
    Eigen::Map<Eigen::VectorXd> obsVec(&fvals(0), numPts);
    Eigen::Map<Eigen::VectorXd> coeffVec(&coeffs(0), numTerms);

    Eigen::VectorXd objGrad;

    for(unsigned int optIt=0; optIt<5; ++optIt){

        comp.CoeffJacobian(pts, coeffs, preds, jac);

        objGrad = predVec-obsVec;

        objective = 0.5*objGrad.squaredNorm();
        coeffVec -= jacMat.transpose().colPivHouseholderQr().solve(objGrad);
    }

    CHECK(objective<1e-3);

}


TEST_CASE("Testing CompactMonotoneComponent CoeffGrad and LogDeterminantCoeffGrad", "[CompactMonotoneComponent_CoeffGrad]")
{
    //const double testTol = 1e-4;
    unsigned int dim = 2;

    // Create points evently spaced on [lb,ub]
    unsigned int numPts = 20;
    //double lb = -0.5;
    //double ub = 0.5;

    Kokkos::View<double**, HostSpace> evalPts("Evaluate Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i){
        evalPts(0,i) = 0.3;
        evalPts(1,i) = 0.5;
    }

    unsigned int maxDegree = 3;
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree, [](MultiIndex m){return m.HasNonzeroEnd() || m.Max() == 0;});
    MultivariateExpansionWorker<BasisEvaluator<BasisHomogeneity::Homogeneous,ProbabilistHermite>,HostSpace> expansion(mset);

    unsigned int maxSub = 20;
    double relTol = 1e-7;
    double absTol = 1e-7;
    AdaptiveSimpson quad(maxSub, 1, nullptr, absTol, relTol, QuadError::First);

    MonotoneComponent<decltype(expansion), Exp, AdaptiveSimpson<HostSpace>, HostSpace, true> comp(expansion, quad);

    Kokkos::View<double*, HostSpace> coeffs("Expansion coefficients", mset.Size());
    for(unsigned int i=0; i<coeffs.extent(0); ++i)
        coeffs(i) = 0.1*std::cos( 0.01*i );

    comp.SetCoeffs(coeffs);

    SECTION("CoeffGrad"){

        Kokkos::View<double**, HostSpace> sens("Sensitivity", 1, numPts);
        for(unsigned int i=0; i<numPts; ++i)
            sens(0,i) = 0.25*(i+1);

        Kokkos::View<double**, HostSpace> grads = comp.CoeffGrad(evalPts, sens);
        REQUIRE(grads.extent(0)==comp.numCoeffs);
        REQUIRE(grads.extent(1)==numPts);

        for(unsigned int j=1; j<numPts; ++j){
            for(unsigned int i=0; i<comp.numCoeffs; ++i){
                CHECK(grads(i,j) == (j+1.0)*grads(i,0));
            }
        }
    }

    SECTION("LogDeterminantCoeffGrad"){

        for(unsigned int i=0; i<numPts; ++i){
            evalPts(0,i) = 0.03*i;
            evalPts(1,i) = -0.05*i;
        }

        Kokkos::View<double*, HostSpace> logDets = comp.LogDeterminant(evalPts);
        Kokkos::View<double*, HostSpace> logDets2;
        Kokkos::View<double**, HostSpace> grads = comp.LogDeterminantCoeffGrad(evalPts);
        REQUIRE(grads.extent(0)==comp.numCoeffs);
        REQUIRE(grads.extent(1)==numPts);

        // Compare with finite difference derivatives
        const double fdstep = 1e-5;

        for(unsigned int i=0; i<coeffs.extent(0); ++i){
            coeffs(i) += fdstep;

            comp.SetCoeffs(coeffs);
            logDets2 = comp.LogDeterminant(evalPts);
            for(unsigned int ptInd=0; ptInd<numPts; ++ptInd)
                CHECK_THAT( grads(i,ptInd), WithinRel((logDets2(ptInd)-logDets(ptInd))/fdstep, 1e-3));

            coeffs(i) -= fdstep;
        }

    }
}