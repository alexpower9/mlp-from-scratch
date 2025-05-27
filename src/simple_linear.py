import numpy as np
import matplotlib.pyplot as plt
from mlp import Dense, ReLu, MSELoss, SGDOptimizer

layer_1 = Dense(1, 1) #so just a sole neuron, taking 1 input
loss = MSELoss()
optimizer = SGDOptimizer(learning_rate=0.01)

inputs = np.array([[x] for x in [1, 2, 3, 4, 5, 6, 7]])
outputs = np.array([[y] for y in [3, 5, 7, 9, 11, 13, 15]])  

for epoch in range(0,1000):
    layer_1.forward(inputs)
    loss.forward(outputs, layer_1.output)

    #now do backward passes
    dvalues = loss.backward(outputs, layer_1.output)    
    layer_1.backward(dvalues)

    optimizer.update_params(layer_1)

    #now just check progress
    if epoch % 100 == 0:
        print(f"Loss:{loss.loss}")

#now just print the weights and biases we expect. The weight should be 2 and the bias should be 1
print("Weights:",layer_1.weights)
print("Bias:", layer_1.biases)






