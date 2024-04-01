#include <catch2/catch_all.hpp>

#include "MParT/ConditionalMapBase.h"

using namespace mpart;
using namespace Catch;
using MemorySpace = Kokkos::HostSpace;

class MyIdentityMap : public ConditionalMapBase<MemorySpace>{
public:
    MyIdentityMap(unsigned int dim, unsigned int numParams) : ConditionalMapBase(dim,dim,numParams){};

    virtual ~MyIdentityMap() = default;

    virtual void EvaluateImpl(StridedMatrix<const double, MemorySpace> const& pts,
                              StridedMatrix<double, MemorySpace>              output) override{Kokkos::deep_copy(output,pts);};

    virtual void GradientImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                               StridedMatrix<const double, MemorySpace> const& sens,
                               StridedMatrix<double, MemorySpace>              output) override
    {
        assert(false);  
    }

    virtual void LogDeterminantImpl(StridedMatrix<const double, MemorySpace> const&,
                                    StridedVector<double, MemorySpace>        output) override{
        for(unsigned int i=0; i<output.size(); ++i)
            output(i)=0.0;
    }

    virtual void InverseImpl(StridedMatrix<const double, MemorySpace> const&,
                             StridedMatrix<const double, MemorySpace> const& r,
                             StridedMatrix<double, MemorySpace>              output) override{Kokkos::deep_copy(output,r);};

    virtual void ParamGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
                               StridedMatrix<const double, MemorySpace> const& sens,
                               StridedMatrix<double, MemorySpace>              output) override
    {
        assert(false);  
    }


    virtual void LogDeterminantParamGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                             StridedMatrix<double, MemorySpace>              output) override
    {   
        assert(false);
    }

    virtual void LogDeterminantInputGradImpl(StridedMatrix<const double, MemorySpace> const& pts, 
                                             StridedMatrix<double, MemorySpace>              output) override
    {   
        assert(false);
    }
};


TEST_CASE( "Testing coefficient functions of conditional map base class", "[ConditionalMapBaseCoeffs]" ) {

    unsigned int numParams = 10;
    MyIdentityMap map(4,numParams);

    CHECK(map.inputDim == 4);
    CHECK(map.outputDim == 4);

    SECTION("Using Kokkos"){

        Kokkos::View<double*, Kokkos::HostSpace> coeffs("New Coeffs", numParams);
        for(unsigned int i=0; i<numParams; ++i)
            coeffs(i) = i;

        map.SetParams(coeffs);
        CHECK(map.Coeffs().extent(0) == numParams);

        for(unsigned int i=0; i<numParams; ++i)
            CHECK(map.Coeffs()(i) == coeffs(i));

        coeffs(0) = 100;
        CHECK(map.Coeffs()(0) != coeffs(0));

        // Now check using a slice of the coefficients
        unsigned int start = 2;
        unsigned int end = 4;
        auto coeffSlice = Kokkos::subview(coeffs, std::make_pair(start, end));

        map.Coeffs() = coeffSlice;
        CHECK(coeffs.extent(0) == numParams);
        CHECK(map.Coeffs().extent(0)==(end-start));

        for(unsigned int i=0; i<end-start; ++i)
            CHECK(map.Coeffs()(i)==coeffs(i+start));

        coeffs(start) = 1024;
        for(unsigned int i=0; i<end-start; ++i)
            CHECK(map.Coeffs()(i)==coeffs(i+start));

    }

    SECTION("Using Eigen"){

        Eigen::VectorXd coeffs(numParams);
        for(unsigned int i=0; i<numParams; ++i)
            coeffs(i) = i;

        Kokkos::resize(map.Coeffs(), numParams);
        map.CoeffMap() = coeffs;
        CHECK(map.Coeffs().extent(0) == numParams);

        for(unsigned int i=0; i<numParams; ++i){
            CHECK(map.Coeffs()(i) == coeffs(i));
            coeffs(i)++;
            CHECK(map.Coeffs()(i) != coeffs(i));
        }

        map.SetParams(coeffs);
        for(unsigned int i=0; i<numParams; ++i){
            CHECK(map.Coeffs()(i) == coeffs(i));
            coeffs(i)++;
            CHECK(map.Coeffs()(i) != coeffs(i));
        }

        map.SetParams(coeffs);
        for(unsigned int i=0; i<numParams; ++i){
            CHECK(map.Coeffs()(i) == coeffs(i));
            coeffs(i)++;
            map.CoeffMap()(i)++;
            CHECK(map.Coeffs()(i) == coeffs(i));
        }
    }

}

TEST_CASE( "Testing evaluation of an identity conditional map", "[ConditionalMapBaseEvaluation]" ) {

    unsigned int dim = 4;
    unsigned int numPts = 100;
    MyIdentityMap map(dim,0);
    CHECK(map.inputDim == dim);
    CHECK(map.outputDim == dim);



    SECTION("Using Kokkos"){

        Kokkos::View<double**, Kokkos::HostSpace> pts("pts", dim, numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        StridedMatrix<const double, Kokkos::HostSpace> ptsConst = pts;

        Kokkos::View<double**, Kokkos::HostSpace> output = map.Evaluate(ptsConst);

        REQUIRE(output.extent(0)==dim);
        REQUIRE(output.extent(1)==numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }
    }

    SECTION("Using Eigen"){

        Eigen::RowMatrixXd pts(dim,numPts);
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        Eigen::RowMatrixXd output;
        output = map.Evaluate(pts);

        REQUIRE(output.rows()==dim);
        REQUIRE(output.cols()==numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }
    }

}


TEST_CASE( "Testing inverse evaluation of an identity conditional map", "[ConditionalMapBaseInverse]" ) {

    unsigned int dim = 4;
    unsigned int numPts = 100;
    MyIdentityMap map(dim,0);
    CHECK(map.inputDim == dim);
    CHECK(map.outputDim == dim);



    SECTION("Using Kokkos"){

        Kokkos::View<double**, Kokkos::HostSpace> pts("pts", dim, numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        Kokkos::View<double**, Kokkos::HostSpace> output = map.Inverse(pts, pts);

        REQUIRE(output.extent(0)==dim);
        REQUIRE(output.extent(1)==numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }
    }

    SECTION("Using Eigen"){

        Eigen::RowMatrixXd pts(dim,numPts);
        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                pts(i,j) = j;
            }
        }

        Eigen::RowMatrixXd output;
        output = map.Inverse(pts,pts);

        REQUIRE(output.rows()==dim);
        REQUIRE(output.cols()==numPts);

        for(unsigned int i=0; i<dim; ++i){
            for(unsigned int j=0; j<numPts; ++j){
                CHECK(output(i,j) == j);
            }
        }
    }

}