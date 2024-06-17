#ifndef MPART_SIGMOID_H
#define MPART_SIGMOID_H

#include <Kokkos_Core.hpp>
#include "MParT/PositiveBijectors.h"
#include "MParT/Utilities/MathFunctions.h"
#include "MParT/Utilities/KokkosSpaceMappings.h"
#include "MParT/Utilities/Miscellaneous.h"
#include "MParT/PositiveBijectors.h"
#include <iostream>

namespace mpart {

/** If the array of starting indices \f$\sigma_k\f$ is not defined explicitly, we can 
 *  will interpret the length of the weight and center vectors in one of these ways.
*/
enum class SigmoidSumSizeType {
    Linear, // The number of terms in the sum satisfy k_1 = 1 and k_i = k_{i-1} + 1
    Constant, // k_i = 1 for all i
};

/**
 * @brief Class to represent univariate function space spanned by sigmoids.
 * Order should generally be >= 4.
 *
 * @details This class defines a family of basis functions $f_k(x)$ built from one-dimensional
 * sigmoid functions.  This family is index by an integer \f$k\f$, similar to the 
 * way the polynomial degree is used to index a class of orthogonal polynomials.  In particular,
 * the family of functions is defined by 
 * \f[
   \begin{aligned}
    f_0(x) & = 1 \\
    f_1(x) & = x \\
    f_2(x) & = -w_0 g(-b_0(x-c_0))\\
    f_3(x) & = w_1 g(b_1(x-c_1)) \\
    f_k(x) & = \sum_{j=1}^{N_k} w_{kj} s( b_{kj}(x-c_{kj}))
    \end{aligned}
    \f]
 *  where \f$c\f$ represents the "center" of a single sigmoid function $s$, \f$w_j,b_j\geq 0\f$ are positive constants, 
 * \f$s\f$ is a monotone increasing function, and \f$g\f$ is an "edge" function
 *  satisfying \f$\lim_{x\to-\infty}g(x)=0\f$ and \f$\lim_{x\to\infty}g(x)=\infty\f$.
 * Generally, the weights will be uniform, but the centers and widths of these sigmoids can
 * affect behavior dramatically. If performing approximation in \f$L^2(\mu)\f$,
 * a good heuristic is to place the weights at the quantiles of \f$\mu\f$ and
 * create widths that are half the distance to the nearest neighbor.
 *
 * The quantities \f$w_{kj}\f$, \f$b_{kj}\f$, and \f$c_{kj}\f$ are stored in unrolled vectors and indexed
 * by an array with values \f$\sigma_k\f$ that store the index of \f$c_{k0}\f$ in the unrolled vector,
 * much like the vector used to define sparse matrices stored in CSR format.   For example if `c` is the
 * unrolled vector of centers and `startInds` contains \f$\sigma_k\f$, then \f$c_{kj}\f$ is stored at `c[startInds[k]+j-1]`.
 * 
 * @tparam MemorySpace Where the (nonlinear) parameters are stored
 * @tparam SigmoidType Class defining eval, @ref SigmoidTypeSpace
 * @tparam EdgeType Class defining the edge function type, e.g., SoftPlus
 */
template <typename MemorySpace, typename SigmoidType = Logistic, typename EdgeType = SoftPlus>
class Sigmoid1d {
 public:

    /** 
     * @brief Constructs starting indices from the number of provided centers.
     * @param centerSize The number of centers provided to the constructor, i.e., centers.extent(0)
     * @param sumType The form of each term's expansion.  
    */
    static Kokkos::View<unsigned int*, MemorySpace> ConstructStartInds(unsigned int centerSize, 
                                                                       SigmoidSumSizeType sumType){

        unsigned int numTerms = Sigmoid1d<MemorySpace,SigmoidType,EdgeType>::ComputeNumSigmoids(centerSize, sumType);
        Kokkos::View<unsigned int*, Kokkos::HostSpace> houtput("starting inds",0);
        
        if(centerSize<=2)
            return Kokkos::View<unsigned int*, MemorySpace>("starting inds", 0);

        if(sumType == SigmoidSumSizeType::Linear){
            
            if((centerSize-2) != (numTerms*(numTerms+1)/2)){
                std::stringstream ss;
                ss << "Sigmoid: Could not construct starting indices.  The size of the centers array cannot be used with the Linear sum type.";
                ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
            }

            Kokkos::resize(houtput, numTerms);
            houtput(0)=2;
            for(int i=1; i<numTerms; ++i){
                houtput(i) = houtput(i-1) + i;
            }

        }else if(sumType == SigmoidSumSizeType::Constant){
            numTerms = centerSize - 2; // 2 is to account for the edge terms
            Kokkos::resize(houtput, numTerms);
            houtput(0) = 2;
            for(int i=1; i<numTerms; ++i){
                houtput(i) = houtput(i-1) + 1;
            }
        }

        return Kokkos::create_mirror_view_and_copy(MemorySpace(), houtput);
    }


	/**
	 * @brief Construct a new Sigmoid 1d object
	 *
	 * Each input should be of length \f$2 + n(n+1)/2\f$, where \f$n\f$ is the
	 * number of sigmoids.
	 *
	 * @param centers Where to center the sigmoids
	 * @param widths How "wide" the basis functions should be
	 * @param weights How much to weight the sigmoids linearly
     * @param startInds Contains indices into "centers", "widths", and "weights" indicating where each term starts.  This contains the \f$\sigma_k\f$ values mentioned above.
	 */
	Sigmoid1d(Kokkos::View<double*, MemorySpace> centers,
			  Kokkos::View<double*, MemorySpace> widths,
			  Kokkos::View<double*, MemorySpace> weights,
              Kokkos::View<unsigned int*, MemorySpace> startInds)
			: startInds_(startInds), centers_(centers), widths_(widths), weights_(weights) {
		Validate();
	}

	/**
	 * @brief Construct a new Sigmoid 1d object from centers and widths
	 *
	 * Each input should be of length \f$2 + n(n+1)/2\f$, where \f$n\f$ is the
	 * number of sigmoids.
	 *
	 * @param centers
	 * @param widths
	 */
	Sigmoid1d(Kokkos::View<double*, MemorySpace> centers,
			  Kokkos::View<double*, MemorySpace> widths,
              Kokkos::View<unsigned int*, MemorySpace> startInds)
			: startInds_(startInds), centers_(centers), widths_(widths) {
		Kokkos::View<double*, MemorySpace> weights ("Sigmoid weights", centers.extent(0));
		Kokkos::deep_copy(weights, 1.0);
		weights_ = weights;
		Validate();
	}

    /**
	 * @brief Construct a new Sigmoid 1d object
	 *
	 * Each input should either be of length \f$2 + n(n+1)/2\f$ if sumType==Linear
     * or of length \f$2+n\f$ if sumType==Constant, where \f$n\f$ is the
	 * number of sigmoids.
	 *
	 * @param centers Where to center the sigmoids
	 * @param widths How "wide" the basis functions should be
	 * @param weights How much to weight the sigmoids linearly
     * @param sumType Indication of how the centers and widths should be interpreted: Either as separate basis
     *  functions if sumType==Constant or if sumType==Linear, it will assume that basis function `k` includes the sum of `k-3` Sigmoid terms.
	 */
	Sigmoid1d(Kokkos::View<double*, MemorySpace> centers,
			  Kokkos::View<double*, MemorySpace> widths,
			  Kokkos::View<double*, MemorySpace> weights,
              SigmoidSumSizeType sumType = SigmoidSumSizeType::Linear)
			: Sigmoid1d(centers, widths, weights, ConstructStartInds(centers.extent(0), sumType)) {}

    /**
	 * @brief Construct a new Sigmoid 1d object from centers and widths
	 *
	 * Each input should either be of length \f$2 + n(n+1)/2\f$ if sumType==Linear
     * or of length \f$2+n\f$ if sumType==Constant, where \f$n\f$ is the
	 * number of sigmoids.
	 *
	 * @param centers Where to center the sigmoids
	 * @param widths How "wide" the basis functions should be
     * @param sumType Indication of how the centers and widths should be interpreted: Either as separate basis
     *  functions if sumType==Constant or if sumType==Linear, it will assume that basis function `k` includes the sum of `k-3` Sigmoid terms.
	 */
	Sigmoid1d(Kokkos::View<double*, MemorySpace> centers,
			  Kokkos::View<double*, MemorySpace> widths,
              SigmoidSumSizeType sumType = SigmoidSumSizeType::Linear)
			: Sigmoid1d(centers, widths, ConstructStartInds(centers.extent(0), sumType)) {}


	/**
	 * @brief Evaluate all sigmoids at one input
	 *
	 * @param output Place to put the output (size max_order+1)
	 * @param max_order Maximum order of basis function to evaluate
	 * @param input Point to evaluate function
	 */
	KOKKOS_FUNCTION void EvaluateAll(double* output, int max_order, double input) const {
		if (order_ < max_order) {
			ProcAgnosticError<std::invalid_argument>("Sigmoid basis evaluation order too large.");
		}

		output[0] = 1.;
		if (max_order == 0) return;

		output[1] = input;
		if (max_order == 1) return;

		output[2] = -weights_(0)*EdgeType::Evaluate(-widths_(0)*(input-centers_(0)));
		if (max_order == 2) return;

		output[3] =  weights_(1)*EdgeType::Evaluate( widths_(1)*(input-centers_(1)));
		if (max_order == 3) return;

		for (int curr_order = START_SIGMOIDS_ORDER; curr_order <= max_order; curr_order++) {
			output[curr_order] = 0.;

            // Figure out what centers correspond to this term
            int istart = startInds_(curr_order-START_SIGMOIDS_ORDER);
            int iend = centers_.extent(0);
            if((curr_order-START_SIGMOIDS_ORDER +1)<startInds_.extent(0)){
                iend = startInds_(curr_order-START_SIGMOIDS_ORDER +1);
            }
            
            // Evaluate the sum that defines this basis function
			for (int param_idx = istart; param_idx < iend; param_idx++) {
				output[curr_order] +=
					weights_(param_idx) *
					SigmoidType::Evaluate(widths_(param_idx) *
						(input - centers_(param_idx)));
			}
		}
	}

	/**
	 * @brief Evaluate all sigmoids up to given order and first derivatives
	 *
	 * @param output Storage for sigmoid evaluation, size max_order+1
	 * @param output_diff Storage for sigmoid derivative, size max_order+1
	 * @param max_order Number of sigmoids to evaluate
	 * @param input Where to evaluate sigmoids
	 */
	KOKKOS_FUNCTION void EvaluateDerivatives(double* output, double* output_diff, int max_order,
							 double input) const {
		if (order_ < max_order) {
			ProcAgnosticError<std::invalid_argument>("Sigmoid derivative evaluation order too large.");
		}

		output[0] = 1.;
		output_diff[0] = 0.;
		if (max_order == 0) return;

		output[1] = input;
		output_diff[1] = 1.;
		if (max_order == 1) return;

		output[2] = -weights_(0)*EdgeType::Evaluate(-widths_(0)*(input-centers_(0)));
		output_diff[2] = weights_(0)*widths_(0)*EdgeType::Derivative(-widths_(0)*(input-centers_(0)));
		if (max_order == 2) return;

		output[3] = weights_(1)*EdgeType::Evaluate( widths_(1)*(input-centers_(1)));
		output_diff[3] = weights_(1)*widths_(1)*EdgeType::Derivative( widths_(1)*(input-centers_(1)));
		if (max_order == 3) return;

		for (int curr_order = START_SIGMOIDS_ORDER; curr_order <= max_order; curr_order++) {
			output[curr_order] = 0.;
			output_diff[curr_order] = 0.;

            // Figure out what centers correspond to this term
            int istart = startInds_(curr_order-START_SIGMOIDS_ORDER);
            int iend = centers_.extent(0);
            if((curr_order-START_SIGMOIDS_ORDER +1)<startInds_.extent(0)){
                iend = startInds_(curr_order-START_SIGMOIDS_ORDER +1);
            }

			for (int param_idx = istart; param_idx < iend; param_idx++) {
				output[curr_order] +=
					weights_(param_idx) *
					SigmoidType::Evaluate(widths_(param_idx) *
						(input - centers_(param_idx)));
				output_diff[curr_order] +=
					weights_(param_idx) * widths_(param_idx) *
					SigmoidType::Derivative(widths_(param_idx) *
						(input - centers_(param_idx)));
			}
		}
	}

	/**
	 * @brief Evaluate sigmoids up to given order and first+second derivatives
	 *
	 * @param output Storage for sigmoid evaluation, size max_order+1
	 * @param output_diff Storage for sigmoid derivative, size max_order+1
	 * @param output_diff2 Storage for sigmoid 2nd deriv, size max_order+1
	 * @param max_order Maximum order of sigmoid to evaluate
	 * @param input Where to evaluate the sigmoids
	 */
	KOKKOS_FUNCTION void EvaluateSecondDerivatives(double* output, double* output_diff,
								   double* output_diff2, int max_order,
								   double input) const {
		if (order_ < max_order) {
			ProcAgnosticError<std::invalid_argument>("Sigmoid second derivative evaluation order too large");
		}

		output[0] = 1.;
		output_diff[0] = 0.;
		output_diff2[0] = 0.;
		if (max_order == 0) return;

		output[1] = input;
		output_diff[1] = 1.;
		output_diff2[1] = 0.;
		if (max_order == 1) return;

		output[2] = -weights_(0)*EdgeType::Evaluate(-widths_(0)*(input-centers_(0)));
		output_diff[2] = weights_(0)*widths_(0)*EdgeType::Derivative(-widths_(0)*(input-centers_(0)));
		output_diff2[2] = -weights_(0)*widths_(0)*widths_(0)*EdgeType::SecondDerivative(-widths_(0)*(input-centers_(0)));
		if (max_order == 2) return;

		output[3] = weights_(1)*EdgeType::Evaluate( widths_(1)*(input-centers_(1)));
		output_diff[3] = weights_(1)*widths_(1)*EdgeType::Derivative( widths_(1)*(input-centers_(1)));
		output_diff2[3] = weights_(1)*widths_(1)*widths_(1)*EdgeType::SecondDerivative( widths_(1)*(input-centers_(1)));
		if (max_order == 3) return;

		for (int curr_order = START_SIGMOIDS_ORDER; curr_order <= max_order; curr_order++) {
			output[curr_order] = 0.;
			output_diff[curr_order] = 0.;
			output_diff2[curr_order] = 0.;

            // Figure out what centers correspond to this term
            int istart = startInds_(curr_order-START_SIGMOIDS_ORDER);
            int iend = centers_.extent(0);
            if((curr_order-START_SIGMOIDS_ORDER +1)<startInds_.extent(0)){
                iend = startInds_(curr_order-START_SIGMOIDS_ORDER +1);
            }

            // Inner loop to evaluate this term
			for (int param_idx = istart; param_idx < iend; param_idx++) {
				output[curr_order] +=
					weights_(param_idx) *
					SigmoidType::Evaluate(widths_(param_idx) *
						(input - centers_(param_idx)));
				output_diff[curr_order] +=
					weights_(param_idx) * widths_(param_idx) *
					SigmoidType::Derivative(widths_(param_idx) *
						(input - centers_(param_idx)));
				output_diff2[curr_order] +=
					weights_(param_idx) * widths_(param_idx) * widths_(param_idx) *
					SigmoidType::SecondDerivative(widths_(param_idx) *
						(input - centers_(param_idx)));
			}
		}
	}

	/**
	 * @brief Get the maximum order of this function class
	 *
	 * @return int
	 */
	KOKKOS_INLINE_FUNCTION int GetOrder() const { return order_; }

	/// @brief Given the length of centers, return the number of sigmoid levels
	static int ComputeNumSigmoids(int numCenters, SigmoidSumSizeType sumType){

        if(sumType == SigmoidSumSizeType::Constant){
            return numCenters - 2;
        }else{
            return (MathSpace::sqrt(1+8*(numCenters - 2)) - 1)/2;
        }
	}

 private:
	void Validate() {
		if (centers_.extent(0) != widths_.extent(0) ||
			centers_.extent(0) != weights_.extent(0) ||
            centers_.extent(0)<startInds_.extent(0)) {
			std::stringstream ss;
			ss << "Sigmoid: incompatible dims of centers and widths.\n";
			ss << "centers: " << centers_.extent(0) << ", \n";
			ss << "widths: " << widths_.extent(0) << ",\n";
			ss << "weights: " << weights_.extent(0) << "\n";
			ProcAgnosticError< std::invalid_argument>(
				ss.str().c_str());
		}
		if (centers_.extent(0) < 2) {
			std::stringstream ss;
			ss << "Sigmoid: centers/widths/weights too short.\n";
			ss << "Length should be of form 2+(1+2+3+...+n) for some order n>=0";
			ProcAgnosticError< std::invalid_argument>(
				ss.str().c_str());
		}
        
        if(startInds_.extent(0)>0){
            Kokkos::View<unsigned int*, Kokkos::HostSpace> hStartInds("device inds", startInds_.extent(0));
            Kokkos::deep_copy(hStartInds, startInds_); 
            if(hStartInds(0)!=2){
                std::stringstream ss;
                ss << "Sigmoid: First term must start at index 2, but has startInds(0)=" << startInds_(0);
                ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
            }
        
            // Make sure start inds are increasing
            for(int i=1; i<hStartInds.extent(0); ++i){
                if(hStartInds(i)<=hStartInds(i-1)){
                    std::stringstream ss;
                    ss << "Sigmoid: Starting indices must be strictly increasing.";
                    ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
                }
                if(hStartInds(i)>=centers_.extent(0)){
                    std::stringstream ss;
                    ss << "Sigmoid: All starting indices must be smaller than the number of sigmoid centers but ";
                    ss << "startInds(" << i << ")=" << startInds_(i) << " and centers.extent(0)=" << centers_.extent(0) << ".";
                    ProcAgnosticError<std::invalid_argument>(ss.str().c_str());
                }
            }
        }

		// one added for affine part of this, two added for edge terms
		order_ = startInds_.extent(0) + 1 + 2;
	}

	int order_;
	static int constexpr START_SIGMOIDS_ORDER = 4;
	static int constexpr START_SIGMOIDS_IDX = 2;
    Kokkos::View<const unsigned int*, MemorySpace> startInds_;
	Kokkos::View<const double*, MemorySpace> centers_, widths_, weights_;
    
};

}  // namespace mpart

#endif  // MPART_SIGMOID_H