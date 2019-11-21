import torch
import os
import torchvision # To Download MNIST Datasets from Torch 
import torchvision.transforms as transforms # To Transform MNIST "Images" to "Tensor"
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image # Load 'save_image' Function


# ================================================================== #
#                        0. Define Hyper-parameters
# ================================================================== #
# Device Configuration for Where the Tensors Be Operated
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define OS Configuration
sample_dir = './results'

# Hyper-parameters
image_size = 784
h_dim = 400
z_dim = 20

num_epochs = 20
batch_size = 128
learning_rate = 1e-3

# ================================================================== #
#                        1. Load Data
# ================================================================== #
train_data = torchvision.datasets.MNIST(root='./datasets',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)

# Doesn't Need Test Data (Going to be Sampled from z~N(0,1))

# ================================================================== #
#                        2. Define Dataloader
# ================================================================== #
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=batch_size,
                                           shuffle=True)

# Doesn't Need Test Loader As Well

# ================================================================== #
#                        3. Define Model
# ================================================================== #
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=h_dim, z_dim=z_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim) # from 784 Nodes(28x28 MNIST Image) to 400 Nodes (h_dim) 
        self.fc2 = nn.Linear(h_dim, z_dim) # from 400 Nodes (h_dim) to 20 Nodes (Dims of mean of z)
        self.fc3 = nn.Linear(h_dim, z_dim) # from 400 Nodes (h_dim) to 20 Nodes (Dims of std of z)
        self.fc4 = nn.Linear(z_dim, h_dim) # from 20 Nodes (reparameterized z=mean+eps*std) to 400 Nodes (h_dim)
        self.fc5 = nn.Linear(h_dim, image_size) # from 400 Nodes (h_dim) to 784 Nodes (Reconstructed 28x28 Image)
        
    # Encoder: Encode Image to Latent Vector z
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    # Reparameterize z=mean+std to z=mean+esp*std
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Decoder: Decode Reparameterized Latent Vector z to Reconstructed Image
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    # Feed Forward the Process and Outputs Estimated (Mean, Std, Reconstructed_Image) at the same time
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

model = VAE().to(device)

# ================================================================== #
#                        4. Set Loss & Optimizer
# ================================================================== #
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Total Loss is going to be defined in Training Part as it is a combination of Reconstruction Loss and Regularization Loss

# ================================================================== #
#                        5. Train / Test
# ================================================================== #
for epoch in range(num_epochs):
    for i, (x, _) in enumerate(train_loader): # '_' as we don't need label of the input Image
        # Feed Forward
        x = x.to(device).view(-1, image_size) # Flatten 2D Image into 1D Nodes
        x_reconst, mu, log_var = model(x)
        
        # Compute the Total Loss
        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False) # See the Description below
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        
        # Get Loss, Compute Gradient, Update Parameters
        loss = reconst_loss + kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print Loss for Tracking Training
        if (i+1) % 50 == 0:
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(train_loader), reconst_loss.item(), kl_div.item()))
            
    # Save Model on Last epoch
    if epoch+1 == num_epochs:
        torch.save(model.state_dict(), './model.pth')
    
    # Save Generated Image and Reconstructed Image at every Epoch
    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(batch_size, z_dim).to(device) # Randomly Sample z (Only Contains Mean)
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))
