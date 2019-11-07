import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# ================================================================== #
#                        0. Define Hyper-parameters
# ================================================================== #
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# ================================================================== #
#                        1. Load Data
# ================================================================== #
train_dataset = torchvision.datasets.MNIST(root='./datasets',
                                           train=True, 
                                           transform=transforms.ToTensor(),  
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./datasets',
                                          train=False, 
                                          transform=transforms.ToTensor())


# ================================================================== #
#                        2. Define Dataloader
# ================================================================== #
data_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


# ================================================================== #
#                        3. Define Model
# ================================================================== #
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = Net(input_size, hidden_size, num_classes).to(device)


# ================================================================== #
#                        4. Set Loss & Optimizer
# ================================================================== #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  


# ================================================================== #
#                        5. Train / Test
# ================================================================== #
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):  
        # Assign Tensors to Configured Device
        images = images.reshape(-1, 28*28).to(device) # reshape dimensions of the input images to fit model
        labels = labels.to(device)
        
        # Forward Propagation
        outputs = model(images)
        
        # Get Loss, Compute Gradient, Update Parameters
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print Loss for Tracking Training
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(data_loader), loss.item()))

# Test after Training is done
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device) # reshape dimensions of the input images to fit model
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


# ================================================================== #
#                        6. Save & Visualization
# ================================================================== #
torch.save(model.state_dict(), 'model.pth')
