from mpart import *
import numpy as np

# note: if you don't assign KokkosInit to a variable, mset.fix below will error. 
ki = KokkosInit({})  

# print(ki)

# Create multi-index set:
multis = np.array([[0, 1], [2, 0]])
mset = MultiIndexSet(multis)

fixed_mset = mset.fix(True)

# Set MapOptions
opts = MapOptions()

# opts.basisType   = BasisTypes.ProbabilistHermite
# opts.posFuncType = PosFuncTypes.SoftPlus
# opts.quadType    = QuadTypes.AdaptiveSimpson
# opts.quadAbsTol  = 1e-6
# opts.quadRelTol  = 1e-6

# map = CreateComponent(fixed_mset, opts)