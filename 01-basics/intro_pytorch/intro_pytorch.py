import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


# ================================================================== #
#                        1. Load Data
# ================================================================== #
train_data = torchvision.datasets.CIFAR10(root='./datasets',
        train=True,
        transform=transforms.ToTensor(),
        download=True)

# ================================================================== #
#                        2. Define Dataloader
# ================================================================== #
data_loader = torch.utils.data.DataLoader(dataset=train_data,
        batch_size=64,
        shuffle=True)

# ================================================================== #
#                        3. Define Model
# ================================================================== #
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 6, 5),
                nn.BatchNorm2d(6),
                nn.ReLU(),
                nn.MaxPool2d(2, 2)
                )
        self.fc = nn.Linear(6*14*14, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

model = Net()

# ================================================================== #
#                        4. Set Loss & Optimizer
# ================================================================== #
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# ================================================================== #
#                        5. Train / Test
# ================================================================== #
epochs = 1
for epoch in range(epochs):
    for i, (images, labels) in enumerate(data_loader):
        outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%50 == 0 or i+1 == len(data_loader):
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, i+1, len(data_loader), loss.item()))

# ================================================================== #
#                        6. Save & Visualization
# ================================================================== #
torch.save(model.state_dict(), 'model.ckpt')

