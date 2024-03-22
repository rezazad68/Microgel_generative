import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import idx2onehot



class AE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.view(-1, 784)
        z = self.encoder(x)
        recon_x = self.decoder(z)
        return recon_x
    


class VAE_base(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super(VAE_base, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = x.view(-1, 784)
        mu_logvar = self.encoder(x)
        mu = mu_logvar[:, :latent_dim]
        log_var = mu_logvar[:, latent_dim:]
        z = self.reparameterize(mu, log_var)
        recon_x = self.decoder(z)
        return recon_x, mu, log_var
        
CLASSES = 10
class VAE(nn.Module):

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, reap=1,
                 conditional=False, num_labels=0):

        super().__init__()
        global CLASSES
        CLASSES = num_labels

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size
        self.reap = reap

        self.encoder = Encoder(
            encoder_layer_sizes, latent_size, reap, conditional, num_labels)
        self.decoder = Decoder(
            decoder_layer_sizes, latent_size, reap, conditional, num_labels)

    def forward(self, x, c=None):

        if x.dim() > 2:
            x = x.view(-1, 28*28)

        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, reap, conditional, num_labels):

        super().__init__()
        self.reap = reap
        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += (self.reap*num_labels)

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            c = idx2onehot(c, n=CLASSES)
            for i in range(self.reap):
                x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, reap, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()
        self.reap = reap

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + (self.reap*num_labels)
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size]+layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i+1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            else:
                self.MLP.add_module(name="sigmoid", module=nn.Sigmoid())

    def forward(self, z, c):

        if self.conditional:
            c = idx2onehot(c, n=CLASSES)
            for i in range(self.reap):
                z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
    
if __name__ == "__main__":
    vae = VAE(
            encoder_layer_sizes=[7500, 256],
            latent_size=2,
            decoder_layer_sizes=[256, 7500],
            conditional=True,
            reap=5,
            num_labels=7)
    
    inp = torch.rand(7, 7500)
    c = torch.arange(0, 7).long().unsqueeze(1)
    out,_,_,_ = vae(inp, c)
    print(out.shape)

