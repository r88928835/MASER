import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a class for LeNet-5 with modified fully connected layers
class lenet(nn.Module):
    def __init__(self):
        super(lenet, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)  # Grayscale input (like MNIST)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Output with 16 channels
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 pooling with stride 2
        
        # Modified Fully Connected Layers
        self.fc1 = nn.Linear(16 * 4 * 4, 300)  # From convolution to fully connected
        self.fc2 = nn.Linear(300, 100)  # 300 input features to 100
        self.fc3 = nn.Linear(100, 10)  # 100 to 10 output features for classification

    def forward(self, x):
        # Convolutional layers with pooling
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        #print("Shape before flattening:", x.shape)
        # Flattening and Fully Connected Layers
        #x = x.view(-1, 16 * 5 * 5)  # Flatten for fully connected layers
        x = x.view(x.size(0), 16 * 4 * 4)

        x = F.relu(self.fc1(x))  # First fully connected with ReLU
        x = F.relu(self.fc2(x))  # Second fully connected with ReLU
        
        # Output layer
        x = self.fc3(x)  # Final output, optionally apply softmax
        return x
