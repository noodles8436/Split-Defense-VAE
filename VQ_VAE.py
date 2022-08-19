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

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    return total_params

# 출처 : https://github.com/Jackson-Kang/Pytorch-VAE-tutorial

# Model Hyperparameters

cuda = True
DEVICE = torch.device("cuda" if cuda else "cpu")

batch_size = 512
img_size = (12, 12)  # (width, height)
print_step = 100
lr = 2e-4
epochs = 100

input_dim = 1
hidden_dim = 128
n_embeddings = 768
output_dim = 1


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(4, 4, 3, 1), stride=2):
        super(Encoder, self).__init__()

        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_size

        self.strided_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, stride, padding=1)
        self.strided_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, stride, padding=1)

        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_3, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, output_dim, kernel_4, padding=0)

    def forward(self, x):
        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)

        x = F.relu(x)
        y = self.residual_conv_1(x)
        y = y + x

        x = F.relu(y)
        y = self.residual_conv_2(x)
        y = y + x

        return y


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))

    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)

        return quantized

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        codebook_loss = F.mse_loss(x.detach(), quantized)
        e_latent_loss = F.mse_loss(x, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, commitment_loss, codebook_loss, perplexity


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
        super(Decoder, self).__init__()

        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes

        self.residual_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, padding=0)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)

        self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_3, stride, padding=0)
        self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_4, stride, padding=0)

    def forward(self, x):
        y = self.residual_conv_1(x)
        y = y + x
        x = F.relu(y)

        y = self.residual_conv_2(x)
        y = y + x
        y = F.relu(y)
        y = self.strided_t_conv_1(y)
        y = self.strided_t_conv_2(y)

        return y


class Model(nn.Module):
    def __init__(self, Encoder, Codebook, Decoder):
        super(Model, self).__init__()
        self.encoder = Encoder
        self.codebook = Codebook
        self.decoder = Decoder

    def forward(self, x):
        z = self.encoder(x)
        z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
        x_hat = self.decoder(z_quantized)

        return x_hat, commitment_loss, codebook_loss, perplexity


encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=hidden_dim)
decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)

model = Model(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(DEVICE)

mse_loss = nn.MSELoss()

optimizer = Adam(model.parameters(), lr=lr)

count_parameters(model)


def getModel():
    _encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=hidden_dim)
    _codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=hidden_dim)
    _decoder = Decoder(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    _model = Model(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(DEVICE)

    _mse_loss = nn.MSELoss()

    _optimizer = Adam(_model.parameters(), lr=lr)

    return _model


def train_VAE():
    _train = VAE_MNIST_Dataset(split_size=img_size[0], isTrain=True)
    train_loader = DataLoader(_train, batch_size=batch_size, shuffle=True)

    print("Start training VQ-VAE...")
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, commitment_loss, codebook_loss, perplexity = model(x)

            recon_loss = mse_loss(x_hat, x)

            loss = recon_loss + commitment_loss + codebook_loss

            loss.backward()
            optimizer.step()

            if batch_idx % print_step == 0:
                print("epoch:", epoch + 1, "  step:", batch_idx + 1, "  recon_loss:", recon_loss.item(),
                      "  perplexity: ",
                      perplexity.item(),
                      "\n\t\tcommit_loss: ", commitment_loss.item(), "  codebook loss: ", codebook_loss.item(),
                      "  total_loss: ", loss.item())

    print("Finish!!")

    torch.save(model.state_dict(), "./VQ-VAE.pth")


def load_model():
    model.load_state_dict(torch.load("./VQ-VAE.pth", map_location='cpu'))


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
            x_hat, commitment_loss, codebook_loss, perplexity = model(x)

            print("perplexity: ", perplexity.item(), "commit_loss: ", commitment_loss.item(), "  codebook loss: ",
                  codebook_loss.item())
            break

    draw_sample_image(x[:batch_size // 2], "Ground-truth images")
    draw_sample_image(x_hat[:batch_size // 2], "Reconstructed images")


if __name__ == "__main__":
    train_VAE()
    load_model()
    draw_samples()
