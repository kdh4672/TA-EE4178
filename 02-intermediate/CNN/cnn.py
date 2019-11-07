import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


# ================================================================== #
#                        0. Define Hyper-parameters
# ================================================================== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
in_channel = 1

batch_size = 1
shuffle = False
max_pool_kernel = 2
learning_rate = 0.01
num_epochs = 5


# ================================================================== #
#                        1. Load Data
# ================================================================== #
train_data = torchvision.datasets.MNIST(root='./datasets',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

test_data = torchvision.datasets.MNIST(root='./datasets',
                                       train=False,
                                       transform=transforms.ToTensor())


# ================================================================== #
#                        2. Define Dataloader
# ================================================================== #
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=shuffle)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)


# ================================================================== #
#                        3. Define Model
# ================================================================== #
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(in_channel, 16, 5, stride=1, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(max_pool_kernel))
        self.fc1 = nn.Linear(7*7*32, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = ConvNet(num_classes).to(device)


# ================================================================== #
#                        4. Set Loss & Optimizer
# ================================================================== #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# ================================================================== #
#                        5. Train / Test
# ================================================================== #
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Assign Tensors to Configured Device
        images = images.to(device)
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
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test after Training is done
model.eval() # Set model to Evaluation Mode (Batchnorm uses moving mean/var instead of mini-batch mean/var)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader)*batch_size, 100 * correct / total))

