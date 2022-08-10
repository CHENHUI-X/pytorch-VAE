import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from torch import distributions

class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, latent_size):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.enc_mu = torch.nn.Linear(H, latent_size)
        self.enc_log_sigma = torch.nn.Linear(H, latent_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = self.enc_mu(x)

        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        return torch.distributions.Normal(loc=mu, scale=sigma)


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        

    def forward(self, x):
        x = F.relu(self.linear1(x))
        mu = torch.tanh(self.linear2(x))
        # shape is equals to original input  image
        return torch.distributions.Normal(mu, torch.ones_like(mu))
        # the  decoder  return  a   distribution

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, state):
        self.q_z = self.encoder(state)
        # print(self.q_z.mean.shape) # ( 128 , 8 )

        z = self.q_z.rsample() # sample a code from latenspace
        # print(z.shape) # ( 128 , 8 ) ,  最后一个batch由于只有96个,所以返回(96,8)

        return self.decoder(z) , self.q_z
        # that's need decoder z as close as x ,
        # meanwhile ,the q_z should be as close as N(0,1)


transform = transforms.Compose(
    [transforms.ToTensor(),
     # Normalize the images to be -0.5, 0.5
     transforms.Normalize(0.5, 1)]
    )
mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

input_dim = 28 * 28
batch_size = 128
num_epochs = 100
learning_rate = 0.01
hidden_size = 512
latent_size = 64

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

dataloader = torch.utils.data.DataLoader(
    mnist, batch_size=batch_size,
    shuffle=True, 
    pin_memory=torch.cuda.is_available())

print('Number of samples: ', len(mnist))

encoder = Encoder(input_dim, hidden_size, latent_size)
decoder = Decoder(latent_size, hidden_size, input_dim)

vae = VAE(encoder, decoder).to(device).train()

optimizer = optim.Adam(vae.parameters(), lr=learning_rate)
fig, axs = plt.subplots(3, 3)
for epoch in range(num_epochs):
    vae.train()
    batch_num = 0
    for data in dataloader:
        batch_num += 1
        inputs, _ = data
        inputs = inputs.view(-1, input_dim).to(device)
        optimizer.zero_grad()
        p_x, q_z = vae(inputs) # p(x|z) and q(z)
        # 这里的decoder 返回的是一个分布,一个基于z 的 x 的分布(即x是变量)
        # 他的想法是,这个分布,如果把原输入x放进去,出来的probability 应该是最大的
        # 返回的分布是一个高斯分布,考虑到稳定性,含有e的指数,因此直接使用log来比大小
        log_likelihood = p_x.log_prob(inputs).sum(-1).mean()

        kl = torch.distributions.kl_divergence(
            q_z, 
            torch.distributions.Normal(0, 1.)
        ).sum(-1).mean()
        loss = -(log_likelihood - kl)
        loss.backward()
        optimizer.step()
        l = loss.item()
        print(
            f'[epoch : {epoch+1}]/[{num_epochs}] - batch_num : [{batch_num}] - loss : {l:.3f} - loglike : {log_likelihood.item():.3f} - kl : {kl.item():.3f}'
        )
        with torch.no_grad():

            z = vae.q_z.rsample()
            out_image = vae.decoder(z).mean[:9, :].reshape(9, 28, 28)
            for i in range(3):
                for j in range(3):
                    axs[i, j].imshow(out_image[i*3+j,:,:])
            fig.tight_layout()  # 调整间距
            plt.pause(0.05)  # 暂停0.01秒
            plt.ioff()  # 关闭画图的窗口

            del z, out_image # clear memory








