import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Définition du VAE
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x est de taille (batch, 784)
        h = F.relu(self.fc1(x))
        mu_z = self.fc_mu(h)
        logvar_z = self.fc_logvar(h)
        return mu_z, logvar_z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        # On met un sigmoid car MNIST est dans [0,1]
        mu_x = torch.sigmoid(self.fc_out(h))
        return mu_x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def encode(self, x):
        mu_z, logvar_z = self.encoder(x)
        return mu_z, logvar_z

    def reparameterize(self, mu_z, logvar_z):
        std = torch.exp(0.5 * logvar_z)
        eps = torch.randn_like(std)
        return mu_z + std * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        recon_x = self.decode(z)
        return recon_x, mu_z, logvar_z


def vae_loss(recon_x, x, mu_z, logvar_z):
    """
    Perte = reconstruction + KL
    Reconstruction: MSE sur les pixels
    KL: analytique pour gaussiennes
    """

    # reconstruction: on somme sur toutes les dimensions puis sur le batch
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # KL pour q(z|x) = N(mu_z, diag(sigma^2)) et p(z) = N(0, I)
    kl_div = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    # Normalisation par la taille du batch
    return (recon_loss + kl_div) / x.size(0)


# 2. Chargement de MNIST
def get_mnist_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),                 # [0,1]
        transforms.Lambda(lambda x: x.view(-1))  # on aplatit en vecteur de taille 784
    ])

    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# 3. Entrainement
def train_vae(model, train_loader, optimizer, num_epochs=10, log_interval=100):
    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        for batch_idx, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon_x, mu_z, logvar_z = model(batch_x)
            loss = vae_loss(recon_x, batch_x, mu_z, logvar_z)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

            if batch_idx % log_interval == 0:
                print(
                    f"Epoch {epoch} "
                    f"[{batch_idx * len(batch_x)}/{len(train_loader.dataset)}] "
                    f"Loss batch: {loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"==> Epoch {epoch} - Loss moyenne: {avg_loss:.4f}")


# 4. Reconstructions et génération
def save_reconstructions(model, test_loader, epoch, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        batch_x, _ = next(iter(test_loader))
        batch_x = batch_x.to(device)
        recon_x, _, _ = model(batch_x)

        # On prend les 8 premières images
        n = 8
        originals = batch_x[:n].view(n, 1, 28, 28)
        recons = recon_x[:n].view(n, 1, 28, 28)

        # On concatène pour avoir une grille original en haut, reconstructions en bas
        comparison = torch.cat([originals, recons])
        save_image(comparison.cpu(), os.path.join(output_dir, f"reconstruction_epoch_{epoch}.png"), nrow=n)
        print(f"Images de reconstruction sauvegardées dans {output_dir}")


def save_samples(model, latent_dim, epoch, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        # z ~ N(0, I)
        z = torch.randn(64, latent_dim).to(device)
        samples = model.decode(z).view(64, 1, 28, 28)
        save_image(samples.cpu(), os.path.join(output_dir, f"samples_epoch_{epoch}.png"), nrow=8)
        print(f"Echantillons générés sauvegardés dans {output_dir}")


# 5. Main
if __name__ == "__main__":
    batch_size = 128
    input_dim = 28 * 28
    hidden_dim = 400
    latent_dim = 20
    num_epochs = 10
    lr = 1e-3
    print(device)

    train_loader, test_loader = get_mnist_dataloaders(batch_size=batch_size)

    model = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        train_vae(model, train_loader, optimizer, num_epochs=1)
        save_reconstructions(model, test_loader, epoch)
        save_samples(model, latent_dim, epoch)
