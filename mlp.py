from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class operator(ABC):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self, w):
        pass
    
class add(operator):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def backward(self, w):
        return self.x.backward(w) + self.y.backward(w)

    def forward(self):
        return self.x.forward() + self.y.forward()

class mul(operator):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def backward(self, w):
        return self.x.backward(w) * self.y.forward() + self.x.forward() * self.y.backward(w)

    def forward(self):
        return self.x.forward() * self.y.forward()

class linear(operator):
    def __init__(self):
        self.x = 0

    def input(self, x):
        self.x = x

    def backward(self, w):
        if w is self:
            return 1
        else:
            return 0

    def forward(self):
        return self.x

class signum(operator):
    def __init__(self, x):
        self.x = x 

    def forward(self):
        if self.x.forward() == 0:
            return 0
        if self.x.forward() > 0:
            return 1
        if self.x.forward() < 0:
            return -1

    def backward(self, w):
        return 0
    

x01 = linear()
w01_11 = linear()

p01_11 = mul(x01, w01_11)

x02 = linear()
w02_11 = linear()

p02_11 = mul(x02, w02_11)

u11 = add(p01_11, p02_11)
x11 = signum(u11)

image = []
for i in range(16):
    image.append([])
    for j in range(16):
        x01.input(i)
        w01_11.input(0)
        x02.input(j)
        w02_11.input(1)
        image[i].append(x11.forward())

# Displaying the matrix
plt.imshow(image, cmap='gray', interpolation='nearest')
plt.colorbar()  # Optional: show a color bar
plt.title('16x16 Matrix (1 = White, 0 = Black)')
plt.axis('off')  # Hide axes
plt.show()
