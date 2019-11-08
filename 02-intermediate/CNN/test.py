import torch
import torchvision
import torchvision.transforms as transforms
from cnn import ConvNet


# Set Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare Test Data
test_data = torchvision.datasets.MNIST(root='./datasets',
                                       train=False,
                                       transform=transforms.ToTensor())

# Define Test Dataloader
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=100,
                                          shuffle=True)

# Load Model and Trained Parameters
model_test = ConvNet(10).to(device)
model_test.load_state_dict(torch.load('./model.pth', map_location=device))
model_test.eval()

# Evaluate
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_test(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {} test images: {} %'.format(len(test_loader)*100, 100 * correct / total))
