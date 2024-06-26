#include "MParT/MultiIndices/FixedMultiIndexSet.h"
#include "MParT/MultiIndices/MultiIndex.h"
#include "MParT/MultiIndices/MultiIndexSet.h"
#include "MParT/Utilities/ArrayConversions.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/GPUtils.h"


#include "MParT/Utilities/Miscellaneous.h"
#include <sstream>
#include <stdio.h>

using namespace mpart;


namespace mpart{

    template<typename MemorySpace>
    struct StartSetter {

        StartSetter(Kokkos::View<unsigned int*, MemorySpace> nzStarts,
                   unsigned int dim) : _nzStarts(nzStarts), _dim(dim){};

        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
            this->_nzStarts(i) = i*_dim;
        };

        Kokkos::View<unsigned int*, MemorySpace> _nzStarts;
        const unsigned int _dim;
    };

    template<typename MemorySpace>
    struct DimSetter {

        DimSetter(Kokkos::View<unsigned int*, MemorySpace> nzDims,
                  unsigned int dim) : _nzDims(nzDims), _dim(dim){};

        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
            this->_nzDims(i) = i%_dim;
        };

        Kokkos::View<unsigned int*, MemorySpace> _nzDims;
        const unsigned int _dim;
    };

    template<typename MemorySpace>
    struct MaxDegreeSetter {

        MaxDegreeSetter(Kokkos::View<unsigned int*, MemorySpace> maxDegrees,
                        Kokkos::View<unsigned int*, MemorySpace> nzDims,
                        Kokkos::View<unsigned int*, MemorySpace> nzOrders,
                        unsigned int dim) : maxDegrees_(maxDegrees), nzDims_(nzDims), nzOrders_(nzOrders), dim_(dim){};

        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
            Kokkos::atomic_max(&maxDegrees_(nzDims_(i)), nzOrders_(i));
        }

        Kokkos::View<unsigned int*, MemorySpace> maxDegrees_;
        Kokkos::View<unsigned int*, MemorySpace> nzDims_;
        Kokkos::View<unsigned int*, MemorySpace> nzOrders_;
        const unsigned int dim_;
    };
    
    /** Used to sort the dimensions within each multiindex so they are increasing. */
    template<typename MemorySpace>
    struct DimensionSorter {

        DimensionSorter(Kokkos::View<unsigned int*, MemorySpace> nzStarts,
                        Kokkos::View<unsigned int*, MemorySpace> nzDims,
                        Kokkos::View<unsigned int*, MemorySpace> nzOrders) : nzStarts_(nzStarts), 
                                                                             nzDims_(nzDims), 
                                                                             nzOrders_(nzOrders){}

        KOKKOS_INLINE_FUNCTION void operator()(const size_t mi) const{
            
            unsigned int start = nzStarts_(mi);
            unsigned int end = nzStarts_(mi+1);
            unsigned int key, orderKey, j;

            // insertion sort 
            for (unsigned int step = start+1; step < end; step++) {
                key = nzDims_(step);
                orderKey = nzOrders_(step);
                j = step - 1;

                while ((key < nzDims_(j)) && (j >= start)) {
                    nzDims_(j + 1) = nzDims_(j);
                    nzOrders_(j + 1) = nzOrders_(j);
                    --j;
                }
                nzDims_(j + 1) = key;
                nzOrders_(j + 1) = orderKey;
            }
        }

        Kokkos::View<unsigned int*, MemorySpace> nzStarts_;
        Kokkos::View<unsigned int*, MemorySpace> nzDims_;
        Kokkos::View<unsigned int*, MemorySpace> nzOrders_;
    };

    template<typename MemorySpace>
    struct MaxDegreeInitializer {

        MaxDegreeInitializer(Kokkos::View<unsigned int*, MemorySpace> maxDegrees) : maxDegrees_(maxDegrees){};

        KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const{
            maxDegrees_(i) = 0;
        };

        Kokkos::View<unsigned int*, MemorySpace> maxDegrees_;
    };
}




template<typename MemorySpace>
FixedMultiIndexSet<MemorySpace>::FixedMultiIndexSet(unsigned int                dimen,
                                       Kokkos::View<unsigned int*, MemorySpace> nonzeroOrders) : dim(dimen),
                                                                                            isCompressed(false),
                                                                                            nzDims("Nonzero dims", nonzeroOrders.extent(0)),
                                                                                            nzOrders(nonzeroOrders)
{
    SetupTerms();
    CalculateMaxDegrees();
}

template<typename MemorySpace>
void FixedMultiIndexSet<MemorySpace>::SetupTerms()
{

    unsigned int numTerms = nzOrders.extent(0) / dim;

    nzStarts = Kokkos::View<unsigned int*, MemorySpace>("Start of a Multiindex", numTerms+1);
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0, numTerms);
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policyDims(0, dim*numTerms);
    Kokkos::parallel_for(policy, StartSetter<MemorySpace>(nzStarts,dim));
    Kokkos::parallel_for(policyDims, DimSetter<MemorySpace>(nzDims,dim));
}
template<>
void FixedMultiIndexSet<Kokkos::HostSpace>::SetupTerms()
{

    unsigned int numTerms = nzOrders.extent(0) / dim;

    nzStarts = Kokkos::View<unsigned int*, Kokkos::HostSpace>("Start of a Multiindex", numTerms+1);
    {
    StartSetter<Kokkos::HostSpace> functor(nzStarts,dim);
    for(unsigned int i=0; i<numTerms; ++i)
        functor(i);
    }
    {
    DimSetter<Kokkos::HostSpace> functor(nzDims,dim);
    for(unsigned int i=0; i<dim*numTerms; ++i)
        functor(i);
    }
}


template<typename MemorySpace>
void FixedMultiIndexSet<MemorySpace>::CalculateMaxDegrees()
{
    maxDegrees = Kokkos::View<unsigned int*, MemorySpace>("Maximum degrees", dim);
    
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> DimPolicy(0, dim); 
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> NZPolicy(0, nzOrders.extent(0));
    Kokkos::parallel_for(DimPolicy, MaxDegreeInitializer<MemorySpace>(maxDegrees));
    Kokkos::parallel_for(NZPolicy, MaxDegreeSetter<MemorySpace>(maxDegrees, nzDims, nzOrders, dim));
}

template<>
void FixedMultiIndexSet<Kokkos::HostSpace>::CalculateMaxDegrees()
{
    maxDegrees = Kokkos::View<unsigned int*, Kokkos::HostSpace>("Maximum degrees", dim);

    {
    MaxDegreeInitializer<Kokkos::HostSpace> functor(maxDegrees);
    for(unsigned int i=0; i<dim; ++i)
        functor(i);
    }

    {
    MaxDegreeSetter<Kokkos::HostSpace> functor(maxDegrees, nzDims, nzOrders, dim);
    for(unsigned int i=0; i<nzOrders.extent(0); ++i)
        functor(i);
    }
}

template<typename MemorySpace>
FixedMultiIndexSet<MemorySpace>::FixedMultiIndexSet(unsigned int                dim,
                                       Kokkos::View<unsigned int*, MemorySpace> nzStarts,
                                       Kokkos::View<unsigned int*, MemorySpace> nzDims,
                                       Kokkos::View<unsigned int*, MemorySpace> nzOrders) : dim(dim),
                                                                                isCompressed(true),
                                                                                nzStarts(nzStarts),
                                                                                nzDims(nzDims),
                                                                                nzOrders(nzOrders)
{
    DimensionSorter<MemorySpace> dimSort {nzStarts, nzDims, nzOrders};
    // Sort so that nzDims increases for each multiindex
    unsigned int N_midxs = nzStarts.extent(0) - 1;
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy{0lu, N_midxs};
    Kokkos::parallel_for(policy, dimSort);

    CalculateMaxDegrees();
}

template<typename MemorySpace>
FixedMultiIndexSet<MemorySpace>::FixedMultiIndexSet(unsigned int dim,
                                                    unsigned int maxOrder,
                                                    unsigned int minOrder) : dim(dim), isCompressed(true)
{
    // Figure out the number of terms in the total order
    unsigned int numTerms, numNz;
    std::tie(numTerms,numNz) = TotalOrderSize(maxOrder, 0);

    // Allocate space for the multis in compressed form
    nzStarts = Kokkos::View<unsigned int*, MemorySpace>("nzStarts", numTerms+1);
    nzDims = Kokkos::View<unsigned int*, MemorySpace>("nzDims", numNz);
    nzOrders = Kokkos::View<unsigned int*, MemorySpace>("nzOrders", numNz);

    // Put in a parallel for loop of length 1 to make sure it's executed on device
    Kokkos::View<unsigned int*, MemorySpace> workspace("workspace", dim);
    unsigned int newNumTerms = 0;
    unsigned int currNz=0;
    FillTotalOrder(maxOrder, minOrder, workspace, 0, newNumTerms, currNz);
    
    Kokkos::resize(nzStarts, newNumTerms+1);
    Kokkos::resize(nzDims, nzStarts(newNumTerms));
    Kokkos::resize(nzOrders, nzStarts(newNumTerms));
   
    CalculateMaxDegrees();
}


template<typename MemorySpace>
Kokkos::View<const unsigned int*, MemorySpace> FixedMultiIndexSet<MemorySpace>::MaxDegrees() const
{
    return maxDegrees;
}

template<typename MemorySpace>
std::vector<unsigned int> FixedMultiIndexSet<MemorySpace>::IndexToMulti(unsigned int index) const
{
    assert(false);
    return std::vector<unsigned int>();
}

template<>
std::vector<unsigned int> FixedMultiIndexSet<Kokkos::HostSpace>::IndexToMulti(unsigned int index) const
{
    std::vector<unsigned int> output(dim,0);
    if(isCompressed){
        for(unsigned int i=nzStarts(index); i<nzStarts(index+1); ++i){
            output.at( nzDims(i) ) = nzOrders(i);
        }
    }else{
        for(unsigned int i=0; i<dim; ++i)
            output.at(i) = nzOrders(i + dim*index);
    }
    return output;
}

std::vector<unsigned int> CompressedNonzeroDiagonalEntries(
    const Kokkos::View<unsigned int*, Kokkos::HostSpace> &nzStarts,
    const Kokkos::View<unsigned int*, Kokkos::HostSpace> &nzDims,
    unsigned int dim) {
    std::vector<unsigned int> output;
    for(unsigned int midx = 0; midx < nzStarts.extent(0)-1; midx++){
        if(nzStarts(midx) == nzStarts(midx+1)) continue;
        if(nzDims(nzStarts(midx+1)-1) == dim-1) output.push_back(midx);
    }
    return output;
}

std::vector<unsigned int> UncompressedNonzeroDiagonalEntries(
    const Kokkos::View<unsigned int*, Kokkos::HostSpace> &orders,
    unsigned int dim) {
    std::vector<unsigned int> output;
    for(unsigned int midx = 0; midx < orders.extent(0)/dim; midx++){
        bool isDiagonal = orders((midx+1)*dim-1) > 0;
        if(isDiagonal) output.push_back(midx);
    }
    return output;
}

template<typename MemorySpace>
std::vector<unsigned int> FixedMultiIndexSet<MemorySpace>::NonzeroDiagonalEntries() const
{
    Kokkos::View<unsigned int*, Kokkos::HostSpace> h_nzStarts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nzStarts);
    Kokkos::View<unsigned int*, Kokkos::HostSpace> h_nzDims = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nzDims);
    Kokkos::View<unsigned int*, Kokkos::HostSpace> h_nzOrders = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nzOrders);
    if(isCompressed) {
        return CompressedNonzeroDiagonalEntries(h_nzStarts, h_nzDims, this->dim);
    } else {
        return UncompressedNonzeroDiagonalEntries(h_nzOrders, this->dim);
    }
}

template<typename MemorySpace>
int FixedMultiIndexSet<MemorySpace>::MultiToIndex(std::vector<unsigned int> const& multi) const
{
    if(isCompressed){

        // Figure out how many nonzeros are in this multiindex
        unsigned int nnz = 0;
        for(auto& val : multi)
            nnz += (val>0) ? 1:0;

        // Now search for the matching multi
        for(unsigned int i=0; i<nzStarts.extent(0); ++i){

            // First, check if the number of nonzeros matches
            if((nzStarts(i+1)-nzStarts(i))==nnz){

                // Now check the contents
                bool matches = true;
                for(unsigned int j=nzStarts(i); j<nzStarts(i+1); ++j){
                    if(nzOrders(j)!=multi.at(nzDims(j))){
                        matches=false;
                        break;
                    }
                }

                // We found it!  Return the current index
                if(matches)
                    return i;
            }
        }

        // We didn't find it, return a negative value
        return -1;

    }else{
        unsigned int numTerms = Size();
        for(unsigned int i=0; i<numTerms; ++i){

            bool isMatch = true;
            for(unsigned int d=0; d<dim; ++d){
                if(multi.at(d) != nzOrders(d + i*dim)){
                    isMatch = false;
                    break;
                }
            }

            if(isMatch)
                return i;
        }

        return -1;

    }
}

template<typename MemorySpace>
FixedMultiIndexSet<MemorySpace> FixedMultiIndexSet<MemorySpace>::Cartesian(FixedMultiIndexSet<MemorySpace> const& otherSet) const
{      
    unsigned int thisSize = Size();
    unsigned int otherSize = otherSet.Size();
    unsigned int newNumTerms = thisSize * otherSize;
    unsigned int thisLength = Length();

    Kokkos::View<unsigned int*, MemorySpace> newStarts("nzStarts", newNumTerms+1);
    Kokkos::View<unsigned int*, MemorySpace> newDims("nzDims", otherSize*nzDims.size() + thisSize*otherSet.nzDims.size());
    Kokkos::View<unsigned int*, MemorySpace> newOrders("nzOrders", otherSize*nzOrders.size() + thisSize*otherSet.nzOrders.size());

    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0, 1);
    Kokkos::parallel_for("Dummy Loop", policy, KOKKOS_CLASS_LAMBDA (const int blahind) {
        
        unsigned int currStart = 0;
        unsigned int thisNumNz, otherNumNz;

        for(unsigned int i=0; i<thisSize; ++i){
            thisNumNz = nzStarts(i+1)-nzStarts(i);

            for(unsigned int j=0; j<otherSize; ++j){
                newStarts(i*otherSize + j) = currStart;
                otherNumNz = otherSet.nzStarts(j+1)-otherSet.nzStarts(j);

                // Copy the multiindex from *this
                for(unsigned int k=0; k<thisNumNz; ++k){
                    newDims(newStarts(i*otherSize + j) + k) = nzDims(nzStarts(i)+k);
                    newOrders(newStarts(i*otherSize + j) + k) = nzOrders(nzStarts(i)+k);
                }
                // Copy the multiindex from otherSet
                for(unsigned int k=0; k<otherNumNz; ++k){
                    newDims(newStarts(i*otherSize + j) + thisNumNz + k) = thisLength + otherSet.nzDims(otherSet.nzStarts(j)+k);
                    newOrders(newStarts(i*otherSize + j) + thisNumNz + k) = otherSet.nzOrders(otherSet.nzStarts(j)+k);
                }

                currStart += thisNumNz + otherNumNz;
            }
        }
        newStarts(newNumTerms) = currStart;
    });
    return FixedMultiIndexSet<MemorySpace>(Length()+otherSet.Length(), newStarts, newDims, newOrders);
}

template<typename MemorySpace>
FixedMultiIndexSet<MemorySpace> FixedMultiIndexSet<MemorySpace>::Concatenate(FixedMultiIndexSet<MemorySpace> const& otherSet) const
{
    if(Length() != otherSet.Length()){
        std::stringstream ss;
        ss << "Length of multiindices must be the same but first set has length " << Length() << " multis and second set has length " << otherSet.Length() << " multis.";
        ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
    }
    
    unsigned int thisSize = Size();
    unsigned int thisNumNz = nzDims.size();
    unsigned int otherSize = otherSet.Size();
    Kokkos::View<unsigned int*, MemorySpace> newStarts("nzStarts", thisSize + otherSize + 1);
    Kokkos::View<unsigned int*, MemorySpace> newDims("nzDims", nzDims.size() + otherSet.nzDims.size());
    Kokkos::View<unsigned int*, MemorySpace> newOrders("nzOrders", nzOrders.size() + otherSet.nzOrders.size());
    
    Kokkos::deep_copy(Kokkos::subview(newDims,std::pair<int,int>(0,nzDims.size())), nzDims);
    Kokkos::deep_copy(Kokkos::subview(newDims,std::pair<int,int>(nzDims.size(),newDims.size())), otherSet.nzDims);
    Kokkos::deep_copy(Kokkos::subview(newOrders,std::pair<int,int>(0,nzOrders.size())), nzOrders);
    Kokkos::deep_copy(Kokkos::subview(newOrders,std::pair<int,int>(nzOrders.size(),newOrders.size())), otherSet.nzOrders);

    // Starts stemming from this set
    Kokkos::deep_copy(Kokkos::subview(newStarts,std::pair<int,int>(0,thisSize+1)), nzStarts);

    //  Starts stemming from other set
    Kokkos::View<unsigned int*, MemorySpace> otherStarts = otherSet.nzStarts;
    Kokkos::RangePolicy<typename MemoryToExecution<MemorySpace>::Space> policy(0, otherSize+1);
    Kokkos::parallel_for("Other Starts", policy, KOKKOS_LAMBDA (const int i) {
        newStarts(thisSize+i) = thisNumNz + otherStarts(i);
    });
    
    return FixedMultiIndexSet<MemorySpace>(Length(), newStarts, newDims, newOrders);
}


template<typename MemorySpace>
MultiIndexSet FixedMultiIndexSet<MemorySpace>::Unfix() const
{
    if(!isCompressed)
        throw std::runtime_error("MultiIndexSet must be compressed to unfix");
    Kokkos::View<unsigned int*, Kokkos::HostSpace> h_nzStarts = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nzStarts);
    Kokkos::View<unsigned int*, Kokkos::HostSpace> h_nzDims = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nzDims);
    Kokkos::View<unsigned int*, Kokkos::HostSpace> h_nzOrders = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nzOrders);
    MultiIndexSet output (this->dim);
    for(int term = 0; term < h_nzStarts.extent(0)-1; term++){
        unsigned int start = h_nzStarts(term);
        unsigned int end = h_nzStarts(term+1);
        auto nzIndTerm = Kokkos::subview(h_nzDims, Kokkos::pair<unsigned int, unsigned int>(start, end));
        auto nzValTerm = Kokkos::subview(h_nzOrders, Kokkos::pair<unsigned int, unsigned int>(start, end));
        MultiIndex midx_term {nzIndTerm, nzValTerm, this->dim};
        output.AddActive(midx_term);
    }
    return output;
}

template<typename MemorySpace>
void FixedMultiIndexSet<MemorySpace>::Print() const
{
    if(isCompressed){
        std::cout << "Starts:\n";
        for(unsigned int i=0; i<nzStarts.extent(0); ++i)
            std::cout << nzStarts(i) << "  ";
        std::cout << std::endl;

        std::cout << "\nDims:\n";
        for(unsigned int i=0; i<nzDims.extent(0); ++i)
            std::cout << nzDims(i) << "  ";
        std::cout << std::endl;

        std::cout << "\nOrders:\n";
        for(unsigned int i=0; i<nzOrders.extent(0); ++i)
            std::cout << nzOrders(i) << "  ";
        std::cout << std::endl;
    }

    std::cout << "\nMultis:\n";
    std::vector<unsigned int> multi;
    for(unsigned int term=0; term<nzStarts.extent(0)-1; ++term){
        multi = IndexToMulti(term);

        for(auto& m : multi)
            std::cout << m << "  ";

        std::cout << std::endl;
    }

}

template<typename MemorySpace>
std::pair<unsigned int, unsigned int> FixedMultiIndexSet<MemorySpace>::TotalOrderSize(unsigned int maxOrder, unsigned int currDim)
{
    unsigned int numTerms=0;
    unsigned int numNz=0;
    unsigned int localTerms, localNz;
    if(currDim<dim-1) {
        for(unsigned int pow=0; pow<=maxOrder; ++pow){
            std::tie(localTerms,localNz) = TotalOrderSize(maxOrder-pow,currDim+1);
            numTerms += localTerms;
            numNz += localNz + ((pow>0)?localTerms:0);
        }
    }else{
        numTerms = maxOrder+1;
        numNz = maxOrder;
    }

    return std::make_pair(numTerms, numNz);
}

template<typename MemorySpace>
void FixedMultiIndexSet<MemorySpace>::FillTotalOrder(unsigned int maxOrder,
                                                     unsigned int minOrder,
                                                     Kokkos::View<unsigned int*, MemorySpace> workspace,
                                                     unsigned int currDim,
                                                     unsigned int &currTerm,
                                                     unsigned int &currNz)
{

    if(currDim<dim-1) {
        for(unsigned int pow=0; pow<=maxOrder; ++pow){
            workspace[currDim] = pow;
            FillTotalOrder(maxOrder-pow, minOrder, workspace, currDim+1, currTerm, currNz);
        }
    }else{

        for(unsigned int pow=0; pow<=maxOrder; ++pow){
            workspace[currDim] = pow;

            // Figure out the total order of the workspace
            unsigned int currOrder = 0;
            for(unsigned int i=0; i<currDim+1; ++i)
                currOrder += workspace[i];

            if(currOrder>=minOrder){
                // Copy the multiindex into the compressed format
                nzStarts(currTerm) = currNz;
                for(unsigned int i=0; i<dim; ++i){
                    if(workspace[i]>0){
                        nzDims(currNz) = i;
                        nzOrders(currNz) = workspace[i];
                        currNz++;
                    }
                }
                currTerm++;
            }
        }

    }

    if(currDim==0)
        nzStarts(currTerm) = currNz;
}

template<>
template<>
FixedMultiIndexSet<Kokkos::HostSpace> FixedMultiIndexSet<Kokkos::HostSpace>::ToDevice<Kokkos::HostSpace>()
{
    return *this;
}

// If a device is being used, compile code to copy the FixedMultiIndexSet to device memory
#if defined(MPART_ENABLE_GPU)
    template<>
    template<>
    FixedMultiIndexSet<DeviceSpace> FixedMultiIndexSet<Kokkos::HostSpace>::ToDevice<DeviceSpace>()
    {
        auto deviceStarts = mpart::ToDevice<DeviceSpace>(nzStarts);
        auto deviceDims = mpart::ToDevice<DeviceSpace>(nzDims);
        auto deviceOrders =  mpart::ToDevice<DeviceSpace>(nzOrders);
        FixedMultiIndexSet<DeviceSpace> output(dim, deviceStarts, deviceDims, deviceOrders);
        return output;
    }

    template<>
    template<>
    FixedMultiIndexSet<Kokkos::HostSpace> FixedMultiIndexSet<DeviceSpace>::ToDevice<Kokkos::HostSpace>()
    {
        assert(false);
        return FixedMultiIndexSet<Kokkos::HostSpace>(0,0);
    }

    template<>
    template<>
    FixedMultiIndexSet<DeviceSpace> FixedMultiIndexSet<DeviceSpace>::ToDevice<DeviceSpace>()
    {
        return *this;
    }

#endif


// Explicit template instantiation
#if defined(MPART_ENABLE_GPU)
    template class mpart::FixedMultiIndexSet<Kokkos::HostSpace>;
    template class mpart::FixedMultiIndexSet<DeviceSpace>;
#else
    template class mpart::FixedMultiIndexSet<Kokkos::HostSpace>;
#endif

