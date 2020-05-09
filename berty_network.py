import numpy as np

# 4 inputs
# 3 batches
inputs = [[5, 3, 2, 4],
          [6, 2, 4, 3],
          [7, 3, 2, 5]]

# 1st layer of 3 neurons
weights1 = [[0.02, 0.40, 0.10, 0.05],
            [0.20, 0.10, 0.04, 0.23],
            [0.34, 0.56, 0.02, 0.55]]

# 2nd layer of 2 neurons
# notice that the number of inputs has changed to 3, conditioned by dimensons of output1
weights2 = [[0.34, 0.56, 0.11],
            [0.22, 0.10, 0.12]]

# the output of neurons computed via multiplication of matrixes 'inputs' and 'weights'
# transposing weights for the dimensions to match

# uno
# dos
# tres
# cuatro
# cinco
# pipo

# calculating output of layer 1
outputs1 = np.dot(inputs, np.array(weights1).T)

# feeding output of layer 1 to layer 2 and calculating output of layer 2
# uhg
outputs2 = np.dot(outputs1, np.array(weights2).T)

print(outputs2)
