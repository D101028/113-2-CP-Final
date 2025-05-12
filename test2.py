import numpy as np
epsilon = 0.1
delta = 0.001
rank = 500 # k
m = int((rank * np.log(42/epsilon) + np.log(2/delta))/(100000*epsilon**2/2))  # Compression dimension (must be > k)
print(m)