#include <catch2/catch_test_macros.hpp>
#include<Kokkos_Sort.hpp>

#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/Utilities/GPUtils.h"

using namespace mpart;

TEST_CASE( "Testing the FixedMultiIndexSet class", "[FixedMultiIndexSet]" ) {

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;

    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim,maxOrder);

    CHECK( mset.Size()==((maxOrder+1)*(maxOrder+2)/2));

    Kokkos::View<const unsigned int*, Kokkos::HostSpace> maxDegrees = mset.MaxDegrees();
    REQUIRE(maxDegrees.extent(0)==2);
    CHECK(maxDegrees(0)==maxOrder);
    CHECK(maxDegrees(1)==maxOrder);

    MultiIndexSet multiSet_original = MultiIndexSet::CreateTotalOrder(dim, maxOrder);
    FixedMultiIndexSet<Kokkos::HostSpace> multiSet_fixed = multiSet_original.Fix(true);
    MultiIndexSet multiSet_reconstructed = multiSet_fixed.Unfix();
    REQUIRE(multiSet_original.Size() == multiSet_reconstructed.Size());
    bool same_max_orders = multiSet_original.MaxOrders() == multiSet_reconstructed.MaxOrders();
    REQUIRE(same_max_orders);
    std::vector<unsigned int> diagonal_idxs_ref = multiSet_reconstructed.NonzeroDiagonalEntries();
    std::vector<unsigned int> diagonal_idxs = multiSet_fixed.NonzeroDiagonalEntries();
    bool diag_idxs_pass = diagonal_idxs_ref == diagonal_idxs;
    REQUIRE(diag_idxs_pass);
}

TEST_CASE( "Testing dimension sorting in the FixedMultiIndexSet class", "[FixedMultiIndexSetSorting]" ) {

    const unsigned int dim = 2;

    // Manually construct the multiindex with a proper sorted ordering of the dimensions
    // The set is given by the multiindices [[1,0], [0,1], [1,2]]
    Kokkos::View<unsigned int*, Kokkos::HostSpace> nzStarts("nzStarts", 4);
    Kokkos::View<unsigned int*, Kokkos::HostSpace> nzDims("nzDims", 4);
    Kokkos::View<unsigned int*, Kokkos::HostSpace> nzOrders("nzOrders", 4);
    nzStarts(0) = 0;
    nzStarts(1) = 1;
    nzStarts(2) = 2;
    nzStarts(3) = 4;

    nzDims(0) = 0; nzOrders(0)=1; // [1,0]
    nzDims(1) = 1; nzOrders(1)=1; // [0,1]
    nzDims(2) = 0; nzOrders(2)=1; // The 1 in [1,2]
    nzDims(3) = 1; nzOrders(3)=2; // The 2 in [1,2]

    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim, nzStarts, nzDims, nzOrders);
    
    CHECK(mset.nzDims(3)>mset.nzDims(2));
    CHECK(mset.nzOrders(3)==2);
    CHECK(mset.nzOrders(2)==1);

    // Manually construct the multiindex with an IMPROPER ordering of the dimensions

    nzDims(0) = 0; nzOrders(0)=1; // [1,0]
    nzDims(1) = 1; nzOrders(1)=1; // [0,1]
    nzDims(2) = 1; nzOrders(2)=2; // The 2 in [1,2]
    nzDims(3) = 0; nzOrders(3)=1; // The 1 in [1,2].  Internal to the FixedMultiIndexSet, this should come before 2.

    FixedMultiIndexSet<Kokkos::HostSpace> mset2(dim, nzStarts, nzDims, nzOrders);

    CHECK(mset2.nzDims(3)>mset2.nzDims(2));
    CHECK(mset2.nzOrders(3)==2);
    CHECK(mset2.nzOrders(2)==1);
    
}

TEST_CASE( "Testing the FixedMultiIndexSet Cartesian function", "[FixedMultiIndexSet_Cartesian]" ) {

    const unsigned int dim1 = 2;
    const unsigned int maxOrder1 = 5;

    const unsigned int dim2 = 2;
    const unsigned int maxOrder2 = 3;

    FixedMultiIndexSet<Kokkos::HostSpace> mset1(dim1, maxOrder1);
    FixedMultiIndexSet<Kokkos::HostSpace> mset2(dim2, maxOrder2);
    FixedMultiIndexSet<Kokkos::HostSpace> mset = mset1.Cartesian(mset2);

    CHECK(mset.Size() == (mset1.Size()*mset2.Size()));
    CHECK(mset.Unfix() == mset1.Unfix().Cartesian(mset2.Unfix()));
}

TEST_CASE( "Testing the FixedMultiIndexSet Concatenate function", "[FixedMultiIndexSet_Concatenate]" ) {

    const unsigned int dim1 = 1;
    const unsigned int maxOrder1 = 3;

    const unsigned int dim2 = 1;
    const unsigned int maxOrder2 = 5;

    FixedMultiIndexSet<Kokkos::HostSpace> mset1(dim1, maxOrder1);
    FixedMultiIndexSet<Kokkos::HostSpace> mset2(dim2, maxOrder2, maxOrder1+1);
    
    MultiIndexSet msetTrue = FixedMultiIndexSet<Kokkos::HostSpace>(dim2, maxOrder2).Unfix();
    FixedMultiIndexSet<Kokkos::HostSpace> mset = mset1.Concatenate(mset2);
    
    CHECK(mset.Unfix() == msetTrue);
}

TEST_CASE( "Testing the MultiIndexSet Cartesian function", "[MultiIndexSet_Cartesian]" ) {

    const unsigned int dim1 = 2;
    const unsigned int dim2 = 3;
    const unsigned int maxOrder = 3;

    MultiIndexSet mset1 = MultiIndexSet::CreateTensorProduct(dim1, maxOrder);
    MultiIndexSet mset2 = MultiIndexSet::CreateTensorProduct(dim2, maxOrder);
    MultiIndexSet msetTrue = MultiIndexSet::CreateTensorProduct(dim1+dim2, maxOrder);
    
    MultiIndexSet mset = mset1.Cartesian(mset2);
    REQUIRE(mset==msetTrue);
}


TEST_CASE("MultiIndexSet from Eigen", "[MultiIndexSetFromEigen]")
{
    Eigen::MatrixXi multis(3,2);
    multis << 0,0,
              1,2,
              3,4;

    MultiIndexSet mset(multis);
    CHECK(mset.at(0).Get(0) == 0);
    CHECK(mset.at(0).Get(1) == 0);
    CHECK(mset.at(1).Get(0) == 1);
    CHECK(mset.at(1).Get(1) == 2);
    CHECK(mset.at(2).Get(0) == 3);
    CHECK(mset.at(2).Get(1) == 4);
}

TEST_CASE( "Testing the FixedMultiIndexSet class with anisotropic degrees", "[AnisotropicFixedMultiIndexSet]" ) {

    const unsigned int dim = 2;

    // Set multiindices to [0,1], [5,2], [4,3]
    Kokkos::View<unsigned int*, Kokkos::HostSpace> degrees("Degrees", 3*dim);
    degrees(0) = 0;
    degrees(1) = 1;
    degrees(2) = 5;
    degrees(3) = 2;
    degrees(4) = 4;
    degrees(5) = 3;

    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim,degrees);

    CHECK( mset.Size()==3);

    Kokkos::View<const unsigned int*, Kokkos::HostSpace> maxDegrees = mset.MaxDegrees();
    REQUIRE(maxDegrees.extent(0)==2);
    CHECK(maxDegrees(0)==5);
    CHECK(maxDegrees(1)==3);
}


#if defined( MPART_ENABLE_GPU)

TEST_CASE( "Testing the FixedMultiIndexSet class copy to device", "[FixedMultiIndexSet]" ) {

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;

    FixedMultiIndexSet<Kokkos::HostSpace> mset(dim,maxOrder);

    FixedMultiIndexSet<DeviceSpace> deviceSet = mset.ToDevice<DeviceSpace>();

}
#endif



TEST_CASE("Conversions between MultiIndexSet types", "[MultiIndexSet Conversions]" ) {

    unsigned int dim = 10;
    unsigned int maxDegree = 3;
    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxDegree);

    FixedMultiIndexSet<Kokkos::HostSpace> fixedSet = mset.Fix();

    REQUIRE(mset.Size() == fixedSet.Size() );

    // Make sure the active multiindices are exactly the same
    for(unsigned int i=0; i<mset.Size(); ++i){
        std::vector<unsigned int> fixedVec = fixedSet.IndexToMulti(i);
        std::vector<unsigned int> vec = mset.IndexToMulti(i).Vector();

        for(unsigned int d=0; d<dim; ++d)
            CHECK(fixedVec.at(d)==vec.at(d));
    }

    Kokkos::View<const unsigned int*, Kokkos::HostSpace> maxDegrees = fixedSet.MaxDegrees();
    REQUIRE(maxDegrees.extent(0)==dim);
    for(unsigned int i=0; i<dim; ++i)
        CHECK(maxDegrees(i)==maxDegree);
}

TEST_CASE("Testing the MultiIndexSet class", "[MultiIndexSet]" ) {

    const unsigned int dim = 2;
    const unsigned int maxOrder = 5;

    MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, maxOrder);

    REQUIRE( mset.Size()==((maxOrder+1)*(maxOrder+2)/2));

    MultiIndexSet mset_sep = MultiIndexSet::CreateTotalOrder(dim+1, maxOrder, MultiIndexLimiter::Separable());
    REQUIRE( mset_sep.Size()== ((maxOrder+1)*(maxOrder+2)/2) + maxOrder);
    REQUIRE( mset_sep.NonzeroDiagonalEntries().size() == maxOrder);


    /*
    AdmissableNeighbor.ValidNeighbor
    --------------------------------

    Purpose:
    Make sure the MultiIndex class will return valid admissable neighbors.

    Test:
    Create MultiIndexSet with dimension of 2 and max order of 1. Add an admissable
    neighbor to index (2,0), and check to make sure that index is an admissable
    neighbor to the MultiIndexSet.

    3 |                         3 |
    2 |                         2 |
    1 | x   x                   1 | x   x
    0 | x   x   o               0 | x   x   x
        ----------------            ----------------
        0   1   2   3               0   1   2   3
    */
    SECTION("ValidNeighbor")
    {
        // MultiIndexSet - the "square".
        MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 1);

        // MultiIndex for testing.
        MultiIndex multi{2,0};

        bool isAdmiss = indexFamily.IsAdmissible(multi);
        // Check the result of IsAdmissable().
        REQUIRE( isAdmiss );
    }


    /*
    AdmissableNeighbor.UndefinedNeighbor
    ------------------------------------

    Purpose:
    Make sure that MultiIndex class will not return a valid neighbor for indices
    that do not have defined admissable neighbors.

    Test:
    Create MultiIndexSet with dimension of 2 and max order of 1. Add an admissable
    neighbor to index (2,1) and, check to make sure that index is not returned as
    an admissable neighbor to the MultiIndexSet - missing a required neighboring
    index (2,0).

    3 |                          3 |
    2 |                          2 |
    1 | x   x   o                1 | x   x   o
    0 | x   x                    0 | x   x
        -----------------            -----------------
        0   1   2   3                0   1   2   3
    */
    SECTION("UndefinedNeighbor")
    {
        // MultiIndexSet - the "square".
        MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 1);

        // Create MultiIndex for testing against the square.
        MultiIndex multi{2,1};

        // Check the result of IsAdmissable().
        REQUIRE( !indexFamily.IsAdmissible(multi));
    }

    /*
    AdmissableNeighbor.OutsideMaxOrder
    ----------------------------------

    Purpose:
    Make sure that MultiIndex class will not return a valid neighbor for indices
    that are outside the max order limit.

    Test:
    Create MultiIndexSet with dimension of 2 and order of 3, and create a max
    index limiter of 3. Add an admissable neighbor to index (4,0), and check to
    make sure that index is not returned as an admissable neighbor to the
    MultiIndexSet.

    4 |                           4 |
      -----------------             -----------------
    3 | x   x   x   x |           3 | x   x   x   x |
    2 | x   x   x   x |           2 | x   x   x   x |
    1 | x   x   x   x |           1 | x   x   x   x |
    0 | x   x   x   x | o         0 | x   x   x   x | o
      --------------------          --------------------
        0   1   2   3   4             0   1   2   3   4

    */
    SECTION("OutsideMaxOrder")
    {
        // Max order limit of 3.
        MultiIndexLimiter::MaxDegree limiter(std::vector<unsigned int>{3,3});

        // MultiIndexSet - the "square".
        MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 3, limiter);

        // Create MultiIndex for testing against the square.
        MultiIndex multi{4,0};

        // Check the result of IsAdmissable().
        REQUIRE( !indexFamily.IsAdmissible(multi) );
    }

    /*
    AdmissableNeighbor.OutsideTotalOrder
    ------------------------------------

    Purpose:
    Make sure that MultiIndex class will not return a valid neighbor for indices
    that are outside the total order.

    Test:
    Create MultiIndexSet with dimension of 2 and max order of 3, and create a
    total index limiter of 3. Add two admissable neighbors to indices (2,2) and
    (4,0), and check to make sure that both indices are not returned as
    admissable neighbors to the MultiIndexSet.

    4 |                           4 |
    3 | x\                        3 | x\
    2 | x   x\  o                 2 | x   x\  o
    1 | x   x   x\                1 | x   x   x\
    0 | x   x   x   x\  o         0 | x   x   x   x\  o
        --------------------          --------------------
        0   1   2   3   4             0   1   2   3   4
    */
    SECTION("OutsideTotalOrder")
    {
        // Total order limit of 3.
        MultiIndexLimiter::TotalOrder limiter(3);

        // MultiIndexSet - the "square".
        MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 3, limiter);

        // Create MultiIndex for testing against the square.
        MultiIndex multi1{4,0};
        MultiIndex multi2{2,2};

        // Check the result of IsAdmissable().
        REQUIRE( !indexFamily.IsAdmissible(multi1));

        REQUIRE( !indexFamily.IsAdmissible(multi2));
    }

    /*
    AdmissableNeighbor.AddAdmissibleNeighbor
    ----------------------------------------

    Purpose:
    Make sure that MultiIndex class will return a valid neighbor for admissible
    neighbors that have been added to the MultiIndexSet.

    Test:
    Create MultiIndexSet with dimension of 2 and max order of 1, add an admissible
    neighbor at (2,0), then add an admissable neighbor to index (2,1). Then check
    to make sure that new index is returned as an admissable neighbor to the
    MultiIndexSet.

    3 |                          3 |
    2 |                          2 |
    1 | x   x   o                1 | x   x   x
    0 | x   x   x                0 | x   x   x
        ----------------             ----------------
        0   1   2   3                0   1   2   3
    */
    SECTION("AddAdmissibleNeighbor")
    {
        // MultiIndexSet - the "square".
        MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 1);

        // Add forward admissible neighbor to index (1,0).
        MultiIndex newIndex{2,0};
        indexFamily.AddActive(newIndex);

        // Create MultiIndex for testing against the square.
        MultiIndex multi{2,1};

        // Check the result of IsAdmissable().
        REQUIRE(indexFamily.IsAdmissible(multi));
    }


    SECTION("BackwardsNeighbors")
    {

        MultiIndexSet indexSet = MultiIndexSet::CreateTensorProduct(2, 3);
        MultiIndex newMulti{2,1};

        std::vector<unsigned int> neighbors = indexSet.BackwardNeighbors(indexSet.MultiToIndex(newMulti));

        REQUIRE(neighbors.size() == 2);

        std::vector<unsigned int> neigh1 = indexSet.IndexToMulti(neighbors.at(0)).Vector();
        std::vector<unsigned int> neigh2 = indexSet.IndexToMulti(neighbors.at(1)).Vector();

        REQUIRE(1 == neigh1.at(0));
        REQUIRE(1 == neigh1.at(1));
        REQUIRE(2 == neigh2.at(0));
        REQUIRE(0 == neigh2.at(1));
    }

    /*
    AdmissableNeighbor.ForciblyExpandAdmissibleNeighbors
    ----------------------------------------------------

    Purpose:
    Make sure that MultiIndex class will return a valid neighbor for admissible
    neighbors that have been added to the MultiIndexSet.

    Test:
    Create MultiIndexSet with dimension of 2 and max order of 1, add admissible
    neighbors at (1,1) using the ForciblyExpand function. Add an admissable
    neighbor to index (3,0), and check to make sure that index is returned as an
    admissable neighbor to the MultiIndexSet.

    3 |                         3 |
    2 | o   o   o               2 | o   o   o
    1 | x   x   o               1 | x   x   o   o
    0 | x   x   o               0 | x   x   x   o
        -----------------           -------------------
        0   1   2   3               0   1   2   3   4
    */
    SECTION("Expand")
    {
        // MultiIndexSet - the "square".
        MultiIndexSet indexFamily = MultiIndexSet::CreateTensorProduct(2, 1);

        // Add forward admissible neighbor to index (1,1) using ForciblyExpand.
        MultiIndex newIndex{1,0};

        unsigned int oldSize = indexFamily.Size();
        REQUIRE(oldSize==4);

        int activeIndex = indexFamily.MultiToIndex(newIndex);
        REQUIRE(activeIndex>=0);

        indexFamily.Expand(activeIndex);
        REQUIRE( (oldSize+1) == indexFamily.Size());

        // Create MultiIndex for testing against the square.
        MultiIndex multi{3,0};

        // Check the result of IsAdmissable().
        REQUIRE( indexFamily.IsAdmissible(multi));

        // Check to make sure https://github.com/MeasureTransport/MParT/issues/308 is resolved
        indexFamily = MultiIndexSet(2);
        indexFamily += MultiIndex{0,0};
        indexFamily += MultiIndex{1,0};
        indexFamily.Expand();
        
        REQUIRE( indexFamily.IsActive(MultiIndex{2,0}) );
        REQUIRE( indexFamily.IsActive(MultiIndex{0,1}) );
        REQUIRE(! indexFamily.IsActive(MultiIndex{1,1}) );   
    }

    /*
        We start with a multiindex set that looks like
        4 | 0
        3 | x
        2 | x
        1 | x   x
        0 | x   x   x   x   0
            -----------------
            0   1   2   3   4

       The only admissible inactive extensions are [4,0] and [0,4] because we construct this set by
        "limiting" a total order multiindex so that the largest mixed term is [1,1].  Then we
        remove the limiter to end up with a set that looks like

        4 | o
        3 | x   o
        2 | x   o
        1 | x   x   o   o
        0 | x   x   x   x   o
           ------------------
            0   1   2   3   4

        The active indices are the same, but the "Frontier" of expandable multiindices is now much larger.
    */
    SECTION("Frontier")
    {
        auto limiter = [](MultiIndex const& multi){ return ( (multi.Get(0)==0)||(multi.Get(1)==0)||((multi.Get(0)<2)&&(multi.Get(1)<2)));};

        REQUIRE(limiter(MultiIndex{2,0}) );

        MultiIndexSet set = MultiIndexSet::CreateTotalOrder(2, 3, limiter);

        std::vector<unsigned int> inds = set.Frontier();
        REQUIRE( inds.size()==2);
        REQUIRE( set.at(inds.at(0)) == MultiIndex{0,3} );
        REQUIRE( set.at(inds.at(1)) == MultiIndex{3,0} );

        inds = set.StrictFrontier();
        REQUIRE( inds.size()==2);
        REQUIRE( set.at(inds.at(0)) == MultiIndex{0,3} );
        REQUIRE( set.at(inds.at(1)) == MultiIndex{3,0} );

        // Now remove the limiter, which should allow more multiindices in the
        set.SetLimiter( MultiIndexLimiter::None() );

        inds = set.Frontier();

        REQUIRE( inds.size() == 5);
        REQUIRE( set.at(inds.at(0)) == MultiIndex{0,2} );
        REQUIRE( set.at(inds.at(1)) == MultiIndex{0,3} );
        REQUIRE( set.at(inds.at(2)) == MultiIndex{1,1} );
        REQUIRE( set.at(inds.at(3)) == MultiIndex{2,0} );
        REQUIRE( set.at(inds.at(4)) == MultiIndex{3,0} );

        inds = set.StrictFrontier();
        REQUIRE( inds.size()==3);
        REQUIRE( set.at(inds.at(0)) == MultiIndex{0,3} );
        REQUIRE( set.at(inds.at(1)) == MultiIndex{1,1} );
        REQUIRE( set.at(inds.at(2)) == MultiIndex{3,0} );

    }

    SECTION("NonzeroDiagonalEntries") {
        unsigned int dim = 2, maxOrder = 3;
        MultiIndexSet set = MultiIndexSet::CreateTotalOrder(dim, maxOrder, MultiIndexLimiter::Separable());
        std::vector<unsigned int> inds = set.NonzeroDiagonalEntries();
        for(unsigned int ind: inds) {
            MultiIndex multi = set.at(ind);
            REQUIRE( multi.HasNonzeroEnd() );
        }
        unsigned int expected_size = 0;
        for(unsigned int i = 0; i < set.Size(); ++i) {
            MultiIndex multi = set.at(i);
            expected_size += multi.HasNonzeroEnd();
        }
        REQUIRE( expected_size == inds.size() );
        FixedMultiIndexSet<Kokkos::HostSpace> fixedSet = set.Fix(true);
        FixedMultiIndexSet<Kokkos::HostSpace> fixedSet2 = set.Fix(false);
        std::vector<unsigned int> inds_fixed = fixedSet.NonzeroDiagonalEntries();
        std::vector<unsigned int> inds_fixed2 = fixedSet2.NonzeroDiagonalEntries();
        std::sort(inds.begin(), inds.end());
        std::sort(inds_fixed.begin(), inds_fixed.end());
        std::sort(inds_fixed2.begin(), inds_fixed2.end());
        bool inds_same_fixed = inds == inds_fixed;
        REQUIRE( inds_same_fixed );
        bool inds_same_fixed2 = inds == inds_fixed2;
        REQUIRE( inds_same_fixed2 );
        MultiIndexSet full_set = MultiIndexSet::CreateTotalOrder(dim, maxOrder, MultiIndexLimiter::NonzeroDiag());
        inds = full_set.NonzeroDiagonalEntries();
        REQUIRE( inds.size() == full_set.Size() );
        for(int i = 0; i < full_set.Size(); ++i) {
            REQUIRE(inds[i] == i);
            REQUIRE( full_set.at(i).HasNonzeroEnd() );
        }
    }
}

TEST_CASE("MultiIndexSet Operator Tests", "[MultiIndexSet_Operators]")
{   
    Eigen::MatrixXi multis1(2,2);
    multis1 << 0,0,
               1,0;
    MultiIndexSet mset1(multis1); 

    Eigen::MatrixXi multis2(2,2);
    multis2 << 0,0,
               0,1;
    MultiIndexSet mset2(multis2); 

    MultiIndexSet mset3(multis1);
    mset3 += mset2;
    REQUIRE(mset3.Size() == 3);
    REQUIRE(mset3.at(0) == MultiIndex{0,0});
    REQUIRE(mset3.at(1) == MultiIndex{1,0});
    REQUIRE(mset3.at(2) == MultiIndex{0,1});

    mset3 = mset1 + mset2;
    REQUIRE(mset3.Size() == 3);
    REQUIRE(mset3.at(0) == MultiIndex{0,0});
    REQUIRE(mset3.at(1) == MultiIndex{1,0});
    REQUIRE(mset3.at(2) == MultiIndex{0,1});

    MultiIndexSet mset4(mset1 + mset2);
    REQUIRE(mset4.Size() == 3);
    REQUIRE(mset4.at(0) == MultiIndex{0,0});
    REQUIRE(mset4.at(1) == MultiIndex{1,0});
    REQUIRE(mset4.at(2) == MultiIndex{0,1});

}

TEST_CASE("MultiIndexSet ReducedMargin Test", "[MultiIndexSet_RM]") {
    unsigned int dim = 5;
    SECTION("OrderZero_RM") {
        // Simple tests for the reduced margin function
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, 0);
        std::vector<MultiIndex> rmset = mset.ReducedMargin();
        REQUIRE(rmset.size() == 5);
        for(auto& multi : rmset)
            REQUIRE(multi.Sum() == 1);
        unsigned int dim_idx = 2;
        rmset = mset.ReducedMarginDim(dim_idx);
        REQUIRE(rmset.size() == 1);
        MultiIndex multi = rmset.at(0);
        REQUIRE(multi.Sum() == 1);
        REQUIRE(multi.Get(dim_idx) == 1);
    }
    unsigned int P = 2;
    SECTION("OrderP_RM") {
        MultiIndexSet mset = MultiIndexSet::CreateTotalOrder(dim, P);
        // The RM of a TO set should be the same as the new midxs in the next TO set
        MultiIndexSet mset2 = MultiIndexSet::CreateTotalOrder(dim, P+1);
        std::vector<MultiIndex> rmset = mset.ReducedMargin();
        REQUIRE(rmset.size() == mset2.Size() - mset.Size());
        for(auto& multi : rmset) {
            REQUIRE(multi.Sum() == P+1);
        }
        unsigned int dim_idx = 0;
        rmset = mset.ReducedMarginDim(dim_idx);
        // I'm unsure how big the RM should be for a particular dimension
        // e.g. [0,2,1] has bw neighbors [0,1,1], [0,2,0], which means it gets included
        // this multiple bw neighbors business makes it hard to count exactly
        REQUIRE(rmset.size() < mset2.Size() - mset.Size());
        for(auto& multi : rmset) {
            REQUIRE(multi.Sum() == P+1);
        }
    }
}

TEST_CASE("MultiIndexSet Visualization Test", "[MultiIndexSet_Viz]")
{
    SECTION("Floating"){
        Eigen::MatrixXi multis(1,2);
        multis << 3,3;
        MultiIndexSet mset = MultiIndexSet(multis);

        std::stringstream sstream;
        mset.Visualize(sstream);

        std::stringstream expected;
        expected << " 4 |          m     \n"     
                << " 3 |       m  a  m  \n"
                << " 2 |          m     \n"
                << " 1 |                \n"
                << " 0 |                \n"
                << "    ----------------\n"
                << "     0  1  2  3  4  \n";

        CHECK(expected.str() == sstream.str());
    }

    SECTION("Fixed"){
        Eigen::MatrixXi multis(2,2);
        multis << 1,3,
                  3,3;
        MultiIndexSet mset = MultiIndexSet(multis);

        std::stringstream sstream;
        mset.Visualize(sstream);

        std::stringstream expected;
        expected <<" 4 |    m     m     \n"     
                << " 3 | m  a  m  a  m  \n"
                << " 2 |    m     m     \n"
                << " 1 |                \n"
                << " 0 |                \n"
                << "    ----------------\n"
                << "     0  1  2  3  4  \n";

        CHECK(expected.str() == sstream.str());
    }

    SECTION("TotalOrder")
    {
        auto limiter = [](MultiIndex const& multi){ return ( (multi.Get(0)==0)||(multi.Get(1)==0)||((multi.Get(0)<2)&&(multi.Get(1)<2)));};

        MultiIndexSet set = MultiIndexSet::CreateTotalOrder(2, 3, limiter);
        set.SetLimiter( MultiIndexLimiter::None() );

        std::stringstream truth, output;
        truth << " 4 | r              \n"
              << " 3 | a  m           \n"
              << " 2 | a  r           \n"
              << " 1 | a  a  r  m     \n"
              << " 0 | a  a  a  a  r  \n"
              << "    ----------------\n"
              << "     0  1  2  3  4  \n";

        set.Visualize(output);

        REQUIRE(output.str() == truth.str());
    }
}