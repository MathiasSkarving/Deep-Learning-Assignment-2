import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets

class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):

        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        super(CNN, self).__init__()

        # 1st convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2) 
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1, count_include_pad=False)
        
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 

        self.batchnorm = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.batchnorm3 = nn.BatchNorm2d(128)
        
        # Booze 
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layer   
        self.fc1 = nn.Linear(7 * 7 * 128, 256)
        
        self.fc2 = nn.Linear(256, 512)
        
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        x = self.avgpool(x)
        x = F.relu(self.batchnorm(self.conv1(x)))   # 32 x 28 x 28
        x = self.pool(F.relu(self.batchnorm2(self.conv2(x))))   # 64 x 14 x 14
        x = self.pool(F.relu(self.batchnorm3(self.conv3(x))))   # 128 x 7 x 7
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class CNNLeNet5(nn.Module):
    def __init__(self, in_channels, num_classes):
        """
        Building blocks of convolutional neural network.

        Parameters:
            * in_channels: Number of channels in the input image (for grayscale images, 1)
            * num_classes: Number of classes to predict. In our problem, 10 (i.e digits from  0 to 9).
        """
        # Call the parent class's init method
        super(CNNLeNet5, self).__init__()
                
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels=6, kernel_size=5, stride=1)
        
        # Max Pooling Layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # First Fully Connected Layer
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        
        # Second Fully Connected Layer
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        
        # Output Layer
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        """
        Define the forward pass of the neural network.

        Parameters:
            x: Input tensor.

        Returns:
            torch.Tensor
                The output tensor after passing through the network.
        """
        
        # Pass the input through the first convolutional layer and activation function
        x = self.pool(F.relu(self.conv1(x)))
        
        # Pass the output of the first layer through 
        # the second convolutional layer and activation function
        x = self.pool(F.relu(self.conv2(x)))
        
        # Reshape the output to be passed through the fully connected layers
        x = x.view(-1, 16 * 4 * 4)
        
        # Pass the output through the first fully connected layer and activation function
        x = F.relu(self.fc1(x))
        
        # Pass the output of the first fully connected layer through 
        # the second fully connected layer and activation function
        x = F.relu(self.fc2(x))
        
        # Pass the output of the second fully connected layer through the output layer
        x = self.fc3(x)
        
        # Return the final output
        return x