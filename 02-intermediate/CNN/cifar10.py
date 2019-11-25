import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


# ================================================================== #
#                        0. Define Hyper-parameters
# ================================================================== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_classes = 10
in_channel = 3

batch_size = 100
shuffle = True
learning_rate = 1e-3
num_epochs = 50


# ================================================================== #
#                        1. Load Data
# ================================================================== #
train_data = torchvision.datasets.CIFAR10(root='./datasets',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

test_data = torchvision.datasets.CIFAR10(root='./datasets',
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
                nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, stride=1, padding=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2))
        self.fc1 = nn.Linear(8*8*32, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))
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
if __name__ == '__main__':
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
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                test_image, test_label = next(iter(test_loader))
                _, test_predicted = torch.max(model(test_image.to(device)).data, 1)
                print('Testing data: [Predicted: {} / Real: {}]'.format(test_predicted, test_label))

        if epoch+1 == num_epochs:
            torch.save(model.state_dict(), 'model.pth')
#        else:
#            torch.save(model.state_dict(), 'model-{:02d}_epochs.pth'.format(epoch+1))

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


