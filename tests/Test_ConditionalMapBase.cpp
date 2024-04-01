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

    virtual void InputGradImpl(StridedMatrix<const double, MemorySpace> const& pts,  
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


TEST_CASE( "Testing coefficient functions of conditional map base class", "[ConditionalMapBaseParams]" ) {

    unsigned int numParams = 10;
    MyIdentityMap map(4,numParams);

    CHECK(map.inputDim == 4);
    CHECK(map.outputDim == 4);

    SECTION("Using Kokkos"){

        Kokkos::View<double*, Kokkos::HostSpace> params("New Params", numParams);
        for(unsigned int i=0; i<numParams; ++i)
            params(i) = i;

        map.SetParams(params);
        CHECK(map.Params().extent(0) == numParams);

        for(unsigned int i=0; i<numParams; ++i)
            CHECK(map.Params()(i) == params(i));

        params(0) = 100;
        CHECK(map.Params()(0) != params(0));

        // Now check using a slice of the coefficients
        unsigned int start = 2;
        unsigned int end = 4;
        auto coeffSlice = Kokkos::subview(params, std::make_pair(start, end));

        map.Params() = coeffSlice;
        CHECK(params.extent(0) == numParams);
        CHECK(map.Params().extent(0)==(end-start));

        for(unsigned int i=0; i<end-start; ++i)
            CHECK(map.Params()(i)==params(i+start));

        params(start) = 1024;
        for(unsigned int i=0; i<end-start; ++i)
            CHECK(map.Params()(i)==params(i+start));

    }

    SECTION("Using Eigen"){

        Eigen::VectorXd params(numParams);
        for(unsigned int i=0; i<numParams; ++i)
            params(i) = i;

        Kokkos::resize(map.Params(), numParams);
        map.ParamMap() = params;
        CHECK(map.Params().extent(0) == numParams);

        for(unsigned int i=0; i<numParams; ++i){
            CHECK(map.Params()(i) == params(i));
            params(i)++;
            CHECK(map.Params()(i) != params(i));
        }

        map.SetParams(params);
        for(unsigned int i=0; i<numParams; ++i){
            CHECK(map.Params()(i) == params(i));
            params(i)++;
            CHECK(map.Params()(i) != params(i));
        }

        map.SetParams(params);
        for(unsigned int i=0; i<numParams; ++i){
            CHECK(map.Params()(i) == params(i));
            params(i)++;
            map.ParamMap()(i)++;
            CHECK(map.Params()(i) == params(i));
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