# Imports
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets
from torchvision.transforms import ToTensor

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# --Hyperparameters----------------------------------------------------------------

# Training parameters
learning_rate = 0.0002
batch_size = 128
epochs = 215

# Network parameters
image_dim = 28 * 28
gen_hidd_dim = 256
disc_hidd_dim = 256
z_noise_dim = 100  # Noise data points


# --Networks-----------------------------------------------------------------------

# Generator Network
class Generator(nn.Module):

    def __init__(
        self,
        noise_dim,
        hidden_dim,
        image_dim,
    ):
        super().__init__()

        self.ff1 = nn.Linear(noise_dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, image_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = F.leaky_relu(self.ff1(x), 0.2)
        x = F.tanh(self.ff2(x))

        return x


# Discriminator Network
class Discriminator(nn.Module):

    def __init__(
        self,
        image_dim,
        hidden_dim,
    ):
        super().__init__()

        self.ff1 = nn.Linear(image_dim, hidden_dim)
        self.ff2 = nn.Linear(hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

    def forward(self, x):
        x = F.leaky_relu(self.ff1(x), 0.2)
        x = self.ff2(x)

        return x


# --Loss functions-------------------------------------------------------------

def loss_discriminator(img, model_d, model_g, opt_d):

    # real loss
    real_targets = torch.ones(img.size(0), 1)
    real_preds = model_d(img)
    real_loss = F.binary_cross_entropy_with_logits(real_preds, real_targets)

    # fake images
    noise = torch.tensor(
        np.random.uniform(-1, 1, size=(img.size(0), z_noise_dim))
        ).float()
    fake_imgs = model_g(noise)

    # fake loss
    fake_targets = torch.zeros(img.size(0), 1)
    fake_preds = model_d(fake_imgs)
    fake_loss = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)

    # total loss
    loss = real_loss + fake_loss

    # update discriminator
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()

    return loss


def loss_generator(img, model_d, model_g, opt_g):

    # fake images
    noise = torch.tensor(
        np.random.uniform(-1, 1, size=(img.size(0), z_noise_dim))
        ).float()
    fake_imgs = model_g(noise)

    # fake loss
    fake_targets = torch.ones(img.size(0), 1)
    fake_preds = model_d(fake_imgs)
    fake_loss = F.binary_cross_entropy_with_logits(fake_preds, fake_targets)

    # update generator
    opt_g.zero_grad()
    fake_loss.backward()
    opt_g.step()

    return fake_loss


# --Train----------------------------------------------------------------------

# Setup training data
train_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
    )

# Initialize models
generator = Generator(z_noise_dim, gen_hidd_dim, image_dim)
discriminator = Discriminator(image_dim, disc_hidd_dim)

# Optimizers
opt_d = torch.optim.Adam(
    discriminator.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
    )
opt_g = torch.optim.Adam(
    generator.parameters(),
    lr=learning_rate,
    betas=(0.5, 0.999)
    )

# Loader
data_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True,
    )

# Training
for epoch in range(epochs):
    for batch_idx, (img, _) in enumerate(data_loader):

        img = img.view(img.shape[0], -1)  # flatten
        img = 2 * img - 1  # normalize to [-1, 1] from [0, 1]

        # loss discriminator
        loss_d = loss_discriminator(img, discriminator, generator, opt_d)

        # loss generator
        loss_g = loss_generator(img, discriminator, generator, opt_g)

    # print loss
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss D: {loss_d}, Loss G: {loss_g}')

# --Generate-------------------------------------------------------------------

# Generate images
noise = torch.tensor(np.random.uniform(-1, 1, size=(25, z_noise_dim))).float()
fake_imgs = generator(noise)
fake_imgs = fake_imgs.view(fake_imgs.size(0), 28, 28).data

# Normalize to [0, 1]
fake_imgs = (fake_imgs + 1) / 2

# Plot
fig, ax = plt.subplots(5, 5)
k = 0
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(fake_imgs[k], cmap='gray')
        ax[i, j].axis('off')
        k += 1
plt.show()
