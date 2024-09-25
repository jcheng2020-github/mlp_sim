import matplotlib.pyplot as plt
import numpy as np

# Original Data points
x1 = [1, 0, -1, 0, 0.5, -0.5, 0.5, -0.5]
x2 = [0, 1, 0, -1, 0.5, 0.5, -0.5, -0.5]
d = [1, 1, 1, 1, 0, 0, 0, 0]

# Augment the dataset by adding slight noise, rotations, or scaling
def augment_data(x1, x2, d, num_augmentations=5):
    x1_aug, x2_aug, d_aug = [], [], []
    for i in range(len(x1)):
        for _ in range(num_augmentations):
            noise_x1 = x1[i] + np.random.uniform(-0.1, 0.1)
            noise_x2 = x2[i] + np.random.uniform(-0.1, 0.1)
            x1_aug.append(noise_x1)
            x2_aug.append(noise_x2)
            d_aug.append(d[i])
    return np.array(x1_aug), np.array(x2_aug), np.array(d_aug)

x1_aug, x2_aug, d_aug = augment_data(x1, x2, d)

plt.figure(figsize=(6,6))
for i in range(len(d)):
    if d[i] == 1:
        plt.scatter(x1[i], x2[i], color='blue', label='d=1' if i == 0 else "")
    else:
        plt.scatter(x1[i], x2[i], color='red', label='d=0' if i == 4 else "")

plt.scatter(x1_aug, x2_aug, color='green', alpha=0.1, label='Augmented')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Scatter plot of Original and Augmented Data')
plt.legend()
plt.grid(True)
plt.show()

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# MSE loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# MLP class with augmented data
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, x, y_true, y_pred):
        output_error = y_true - y_pred
        d_output = output_error * sigmoid_derivative(y_pred)

        hidden_error = np.dot(d_output, self.W2.T)
        d_hidden = hidden_error * sigmoid_derivative(self.a1)

        self.W2 += self.learning_rate * np.dot(self.a1.T, d_output)
        self.b2 += self.learning_rate * np.sum(d_output, axis=0, keepdims=True)
        self.W1 += self.learning_rate * np.dot(x.T, d_hidden)
        self.b1 += self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

    def train(self, x, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(x)
            self.backward(x, y, y_pred)
            loss = mse_loss(y, y_pred)
            losses.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        
        plt.figure()
        plt.plot(losses, label='Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Reduction During Training')
        plt.grid(True)
        plt.show()

# Original dataset
x_input = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])
y_output = np.array([[1], [1], [1], [1], [0], [0], [0], [0]])

# Augmented dataset
x_input_aug = np.vstack((x_input, np.column_stack((x1_aug, x2_aug))))
y_output_aug = np.vstack((y_output, np.array(d_aug).reshape(-1, 1)))

# Initialize and train the MLP on original data
print("Training on Original Data:")
mlp_original = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)

mlp_original.train(x_input, y_output, epochs=1000)
# Show decision boundary for the original data
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

grid_input = np.c_[xx.ravel(), yy.ravel()]
grid_output = mlp_original.forward(grid_input)
grid_output = grid_output.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, grid_output, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'blue'])
plt.scatter(x1, x2, c=d, cmap='bwr', edgecolor='k', label="Original Data")

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary after Training with Original Data')
plt.grid(True)
plt.show()

# Initialize and train the MLP on augmented data
print("Training on Augmented Data:")
mlp_augmented = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
mlp_augmented.train(x_input_aug, y_output_aug, epochs=1000)

# Show decision boundary for the augmented data
x_min, x_max = -1.5, 1.5
y_min, y_max = -1.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

grid_input = np.c_[xx.ravel(), yy.ravel()]
grid_output = mlp_augmented.forward(grid_input)
grid_output = grid_output.reshape(xx.shape)

plt.figure(figsize=(6,6))
plt.contourf(xx, yy, grid_output, levels=[0, 0.5, 1], alpha=0.3, colors=['red', 'blue'])
plt.scatter(x1, x2, c=d, cmap='bwr', edgecolor='k', label="Original Data")
plt.scatter(x1_aug, x2_aug, c=d_aug, cmap='bwr', alpha=0.1, label="Augmented Data")

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundary after Training with Augmented Data')
plt.grid(True)
plt.show()
