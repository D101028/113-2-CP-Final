import numpy as np
from numpy.linalg import svd 

A = np.array([
    [1,2],
    [8,7]
])

a, b, c  = svd(A)

print(a, b, c)

print()
