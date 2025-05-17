import numpy as np
epsilon = 0.1
delta = 0.001
k = 10          # rank
m = int((k * np.log(42/epsilon) + np.log(2/delta))/(1000*epsilon**2/2))  # Compression dimension (must be > k)
print(m)