#include <catch2/catch_all.hpp>

#include "MParT/MapFactory.h"
#include "MParT/ConditionalMapBase.h"
#include "MParT/MultiIndices/FixedMultiIndexSet.h"

#include <unordered_map>
#include <string> 

using namespace mpart;
using namespace Catch;


TEST_CASE( "Testing map component factory", "[MapFactoryComponent]" ) {

    MapOptions options;
    options.polyType = PolyTypes::ProbabilistHermite;

    unsigned int dim = 3;
    unsigned int maxDegree = 5;
    FixedMultiIndexSet mset(dim,maxDegree);

    std::shared_ptr<ConditionalMapBase> map = CreateComponent(mset, options);

    unsigned int numPts = 100;
    Kokkos::View<double**,Kokkos::HostSpace> pts("Points", dim, numPts);
    for(unsigned int i=0; i<numPts; ++i)
        pts(dim-1,i) = double(i)/double(numPts-1);

    Kokkos::View<double**, Kokkos::HostSpace> eval = map->Evaluate(pts);
}

