import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

NUM_EPOCHS = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0002
LATENT_DIM = 100
SAVE_INTERVAL = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # Construct generator. You should experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 784
        #   Output non-linearity

        self.layers = nn.Sequential(
            nn.Linear(LATENT_DIM, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            # non-linearity
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        out = self.layers(z)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You should experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        img_flat = img.view(img.size(0), -1)
        validity = self.layers(img_flat)
        return validity


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D):
    # Binary loss function
    bce_loss = nn.BCELoss()
    bce_loss = bce_loss.to(device)

    for epoch in range(NUM_EPOCHS):
        for i, (imgs, _) in enumerate(dataloader):
            # imgs: 64*1*28*28(64*1*784)
            imgs = imgs.to(device)

            # set ground truth for obtaining loss
            real = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)

            # ------- Train generator -------
            optimizer_G.zero_grad()

            # noise inputs
            noise = torch.randn(imgs.size(0), LATENT_DIM).to(device)
            fake_img = generator(noise)

            # evaluate discriminator
            g_loss = bce_loss(discriminator(fake_img), real)

            g_loss.backward()
            optimizer_G.step()

            # ------- Train discriminator -------
            optimizer_D.zero_grad()

            real_loss = bce_loss(discriminator(imgs.detach()), real)
            fake_loss = bce_loss(discriminator(fake_img.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # ------- Save Images -------
            batches_done = epoch * len(dataloader) + i
            if batches_done % SAVE_INTERVAL == 0:
                save_img = fake_img.view(BATCH_SIZE, 1, 28, 28)
                save_image(save_img.data[:25], f'images/{batches_done}.png', nrow=5, normalize=True)

            print(f"[Epoch {epoch}/{NUM_EPOCHS}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5),
                                                (0.5))])),
        batch_size=BATCH_SIZE, shuffle=True)

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D)

    # You can save your generator here to re-use it to generate images for your report
    torch.save(generator.state_dict(), "mnist_generator.pth")

print(f'Device: {device}')
main()
