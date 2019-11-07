import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ================================================================== #
#                        1. Load Data
# ================================================================== #
train_data = torch.tensor([[0.,0.], [0.,1.], [1.,0.], [1.,1.]], requires_grad=True)
targets = torch.tensor([0.,1.,1.,0.]).view(-1, 1)

# ================================================================== #
#                        2. Define Dataloader
# ================================================================== #
# - Doesn't Need PyTorch Built-in Dataloader for Small Data
# - Doesn't Need Custom Dataloader for No Preprocessing Required Data

# ================================================================== #
#                        3. Define Model
# ================================================================== #
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
        
model = Net()

# ================================================================== #
#                        4. Set Loss & Optimizer
# ================================================================== #
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# ================================================================== #
#                        5. Train / Test
# ================================================================== #
epochs = 15000
for idx in range(epochs):
    for i, (input, target) in enumerate(zip(train_data, targets)):
        # Forward Propagation
        output = model(input)
        
        # Get Loss, Compute Gradient, Update Parameters
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print Loss for Tracking Training
        if idx % 1000 == 0 and (idx//1000)%4 == i:
            print("Epoch: {: >8} | Loss: {:8f} | Output: {:4f} | Target: {}".format(idx, loss, output.data[0], target.data[0]))
            
# Test after Training is done
with torch.no_grad():
    print("-----------------------------------------------------------------")
    print("Trained Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.size(), '\n', np.round_(param.data.numpy(),2))

    print("-----------------------------------------------------------------")
    print("Final results:")
    for input, target in zip(train_data, targets):
        output = model(input)
        print("Input: {} | Output: {:4f} | Target: {}".format(input.data, output.data[0], target.data[0]))
    

# ================================================================== #
#                        6. Save & Visualization
# ================================================================== #
torch.save(model.state_dict(), 'model.pth')

