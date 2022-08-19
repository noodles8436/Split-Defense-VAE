import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from VAE_MNIST import VAE_MNIST_Dataset

# 출처 : https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

# Model Hyperparameters

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

batch_size = 512
img_size = (12, 12)  # (width, height)
print_step = 100

x_dim = 144 #784
hidden_dim = 400
latent_dim = 200

lr = 1e-3

epochs = 50


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True

    def forward(self, x):
        h_ = self.LeakyReLU(self.FC_input(x))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x):
        h = self.LeakyReLU(self.FC_hidden(x))
        h = self.LeakyReLU(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

BCE_loss = nn.BCELoss()


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

optimizer = Adam(model.parameters(), lr=lr)


def getModel():
    _encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    _decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=x_dim)
    _model = Model(Encoder=_encoder, Decoder=_decoder).to(DEVICE)

    _BCE_loss = nn.BCELoss()
    _optimizer = Adam(model.parameters(), lr=lr)

    return _model


def train_VAE():
    _train = VAE_MNIST_Dataset(split_size=img_size[0], isTrain=True)
    train_loader = DataLoader(_train, batch_size=batch_size, shuffle=True)

    print("Start training VQ-VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.view(batch_size, x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))

    print("Finish!!")

    torch.save(model.state_dict(), "./VQ-VAE.pth")


def load_model():
    model.load_state_dict(torch.load("./VAE.pth", map_location='cpu'))


def draw_sample_image(x, postfix):
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Visualization of {}".format(postfix))
    plt.imshow(np.transpose(make_grid(x.detach().cpu(), padding=2, normalize=True), (1, 2, 0)))
    plt.savefig(postfix)


def draw_samples():
    _val = VAE_MNIST_Dataset(split_size=img_size[0], isTrain=False)
    val_loader = DataLoader(_val, batch_size=batch_size, shuffle=False)

    model.eval()

    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(tqdm(val_loader)):
            x = x.to(DEVICE)
            x_hat, mean, log_var = model(x)
            break

    draw_sample_image(x[:batch_size // 2], "Ground-truth images")
    draw_sample_image(x_hat[:batch_size // 2], "Reconstructed images")


if __name__ == "__main__":
    train_VAE()
    load_model()
    draw_samples()
