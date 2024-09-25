import matplotlib.pyplot as plt
import numpy as np

def signum(x):
    # If x is a numpy array, apply element-wise
    if isinstance(x, np.ndarray):
        result = np.empty_like(x)
        for i in range(x.size):
            if x.flat[i] > 0:
                result.flat[i] = 1
            elif x.flat[i] < 0:
                result.flat[i] = -1
            else:
                result.flat[i] = 0
        return result
    # If x is a scalar
    else:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.array([[1, 1,  1, 1, 1, 1],
                            [2, -2,  2, -2, 0, 0]])
        self.b1 = [[-16, 16, -14, 18, -10, -9]] 
        self.W2 = np.array([[1],
                            [1],
                            [-1],
                            [-1],
                            [1],
                            [-1]])
        self.b2 = [[0]] 

    def forward(self, X):
        # Input to hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = signum(self.z1)  

        # Hidden to output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = signum(self.z2)  

        return self.a2

# Example usage
if __name__ == "__main__":
    # Define the network parameters
    input_size = 2  # Number of input features
    hidden_size = 3  # Number of hidden neurons
    output_size = 1  # Number of output classes

    # Create the MLP
    mlp = MLP(input_size, hidden_size, output_size)

    # Perform forward pass
    matrix = []
    for i in range(16):
        matrix.append([])
        for j in range(16):
            output = mlp.forward(np.array([i, j]))
            matrix[i].append(output[0][0])
            print(output[0][0], end=' ')
        print()

# Displaying the matrix
plt.imshow(matrix, cmap='binary')  # 'binary' colormap: 0 = black, 1 = white
plt.colorbar()  # Optional: show a color bar
plt.title('16x16 Matrix (1 = White, 0 = Black)')
plt.axis('off')  # Hide axes
plt.show()
