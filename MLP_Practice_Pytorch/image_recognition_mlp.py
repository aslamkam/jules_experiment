import os
import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class MLP(nn.Module):
    '''
    Multilayer Perceptron.
    '''
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

if __name__ == '__main__':
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Prepare CIFAR-10 dataset
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    
    # Initialize the MLP
    mlp = MLP()
    
    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    
    # Store loss values for plotting
    loss_values = []
    iterations = []
    
    # Run the training loop
    iteration = 0
    for epoch in range(0, 5):  # 5 epochs at maximum
        print(f'Starting epoch {epoch+1}')
        
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Perform forward pass
            outputs = mlp(inputs)
            
            # Compute loss
            loss = loss_function(outputs, targets)
            
            # Perform backward pass
            loss.backward()
            
            # Perform optimization
            optimizer.step()
            
            # Store loss for plotting
            loss_values.append(loss.item())
            iterations.append(iteration)
            iteration += 1
    
    # Plot the loss curve
    plt.plot(iterations, loss_values, label='Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs Iteration')
    plt.legend()
    plt.show()
    
    print('Training process has finished.')
