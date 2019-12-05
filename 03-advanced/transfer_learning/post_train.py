from cnn import ConvNet
from font_dataset import FontDataset
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import os


lr = 0.001
num_epochs = 1
batch_size = 100

### Config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Load Data
train_dir = '~/datasets/font/npy_train'.replace('~', os.path.expanduser('~'))
train_data = FontDataset(train_dir)

test_dir = '~/datasets/font/npy_test'.replace('~', os.path.expanduser('~'))
test_data = FontDataset(test_dir)

### Define Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                           batch_size=batch_size)

### Define Model and Load Params
model = ConvNet().to(device)
print("========================== Original Model =============================", "\n", model)
model.load_state_dict(torch.load('./pths/cifar10_pre_model.pth', map_location=device))

### User pre-trained model and Only change last layer
for param in model.parameters():
    param.requires_grad = False

model.fc2 = nn.Linear(120, 50)
modle = model.to(device)

print("========================== Modified Model =============================", "\n", model)

### Define Loss and Optim
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

### Train
if __name__ == '__main__':
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).to(device)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print Loss for Tracking Training
            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
                test_image, test_label = next(iter(test_loader))
                _, test_predicted = torch.max(model(test_image.to(device)).data, 1)

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
