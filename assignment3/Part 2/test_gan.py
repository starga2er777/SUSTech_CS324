import torch
import my_gan
import numpy as np
from torchvision.utils import save_image, make_grid


LATENT_DIM = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'


seed = 1
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

noises = []
noise_begin = torch.randn(1, LATENT_DIM).to(device)
noise_end = torch.randn(1, LATENT_DIM).to(device)
num_interpolations = 7
alphas = np.linspace(0, 1, num_interpolations)
noises.append(noise_begin)
noises.extend([(1 - alpha) * noise_begin + alpha * noise_end for alpha in alphas])
noises.append(noise_end)

# Load Model
mnist_generator = my_gan.Generator().to(device)
state_dict = torch.load('mnist_generator.pth', map_location=device)
mnist_generator.load_state_dict(state_dict)
mnist_generator.eval()

generated_images = [mnist_generator(noise).view(1, 28, 28).detach().cpu() for noise in noises]

grid = make_grid(generated_images, nrow=len(generated_images), normalize=True, range=(-1, 1))

save_image(grid, 'images/interpolation.png')