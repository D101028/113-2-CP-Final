import numpy as np
epsilon = 0.9
delta = 0.01
rank = 500 # k
m = int((rank * np.log(42/epsilon) + np.log(2/delta))/(epsilon**2/2))  # Compression dimension (must be > k)
print(m)