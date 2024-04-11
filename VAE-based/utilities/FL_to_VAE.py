import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam

torch.cuda.empty_cache()
cuda = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_dim = 600 # 512 # 64, 400, 600
latent_dim = 300 # 256 # 32, 200, 300
lr = 1e-3
epochs = 100
"""
   A simple implementation of Gaussian MLP Encoder and Decoder
"""
class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
                super(Encoder, self).__init__()

                self.FC_input = nn.Linear(input_dim, hidden_dim)
                self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
                self.FC_mean = nn.Linear(hidden_dim, latent_dim)
                self.FC_var = nn.Linear(hidden_dim, latent_dim)
                self.LeakyReLU = nn.LeakyReLU(0.2) # 0.2
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
                self.LeakyReLU = nn.LeakyReLU(0.2) #0.2

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
                #z = self.reparameterization(mean,torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
                z = self.reparameterization(mean, torch.exp(log_var))
                x_hat = self.Decoder(z)
                return x_hat, mean, log_var

def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD   = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    return reproduction_loss + KLD

def fl_to_vae(x):
  x = torch.from_numpy(x).float()
  x_dim = x.size()[1]  # 784, columns
  encoder = Encoder(input_dim=x_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
  decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = x_dim)
  model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
  #BCE_loss = nn.BCELoss()
  optimizer = Adam(model.parameters(), lr=lr)
  print("Start training VAE...")
  model.train()
  for epoch in range(epochs):
        overall_loss = 0
        x = x.to(DEVICE)
        optimizer.zero_grad()
        x_hat, mean, log_var = model(x)
        loss = loss_function(x, x_hat, mean, log_var)
        overall_loss += loss.item()
        loss.backward()
        optimizer.step()
        #print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss)
  print("VAE Training Finish!!")
  model.eval()
  with torch.no_grad():
      x_hat, mean, log_var = model(x)
      reconstruction_errors = torch.mean((x_hat - x) ** 2, dim=1)
  # choose the reconstruction one with max reconstruction_errors as attacker
  #outlier_indices = torch.topk(reconstruction_errors, k=1).indices

  print(reconstruction_errors)
  sort_idx = torch.sort(reconstruction_errors)[1]
  outlier_indices = sort_idx[-2]


  # choose the reconstruction one with min reconstruction_errors as attacker
  #outlier_indices = torch.argmin(reconstruction_errors)

  w_attack = x_hat[outlier_indices].squeeze()
  #w_attack = x[outlier_indices].squeeze()
  #w_attack =  w_attack * np.random.uniform(0.01, 0.1, len(w_attack))
  #w_attack = w_attack
  w_attack = w_attack.detach().cpu().numpy()
  return w_attack