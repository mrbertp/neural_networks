import numpy as np

# 4 inputs
# 1 batch
inputs = [5, 3, 2, 4]

# 3 neurons
weights = [[0.02, 0.40, 0.10, 0.05],
           [0.20, 0.10, 0.04, 0.23],
           [0.34, 0.56, 0.02, 0.55]]

outputs = np.dot(weights, inputs)

print(outputs)
