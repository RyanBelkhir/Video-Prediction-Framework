
import torch 
import torch.optim as optim
from models.ddpm import EMA
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def noise_estimation_loss(model, x_0, ddpm, cond=None, mode="L2"):
    batch_size = x_0.shape[0]
    #if cond is not None:
    #    x_0_cond = x_0[:,:5,:]
    #    x_0 = x_0[:,5:10,:]
    # Select a random step for each example
    t = torch.randint(0, ddpm.n_steps, size=(batch_size // 2 + 1,)).cuda()
    t = torch.cat([t, ddpm.n_steps - t - 1], dim=0)[:batch_size].long().cuda()
    # x0 multiplier
    a = ddpm.extract(ddpm.alphas_bar_sqrt, t, x_0)
    # eps multiplier
    am1 = ddpm.extract(ddpm.one_minus_alphas_bar_sqrt, t, x_0)
    e = torch.randn_like(x_0).cuda()
    # model input
    x = x_0 * a + e * am1
    output = model(x, t, cond)
    if mode == "L1":
        def pow_(x):
            return x.abs()
    else:
        def pow_(x):
            return 1 / 2. * x.square()

    #loss2 = ((e - output).reshape(len(x), -1)).abs().sum(dim=-1).mean(dim=0) 
    loss = pow_((e - output).reshape(len(x), -1)).sum(dim=-1).mean(dim=0) 
    return loss


class Trainer(object):
    
    def __init__(self, model, ddpm, model_name="model"):
        self.model = model
        self.ddpm = ddpm
        self.n_epochs = model.config.training.n_epochs
        self.learning_rate = model.config.training.learning_rate
        self.model_name = model_name
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.ema = EMA(model.config)
        self.ema.register(self.model)


    def train(self, train_loader):
        list_loss = []
        num_frames = self.model.num_frames
        num_frames_cond = self.model.num_frames_cond
        t = 0
        for n in range(self.n_epochs):
            for seq in train_loader:
                seq = seq.float().to(device).squeeze()
                seq = 2 * seq - 1
                cond, data = seq[:, :num_frames_cond, :], seq[:, num_frames_cond:num_frames_cond + num_frames, :]
                # Compute the loss.
                loss = noise_estimation_loss(self.model, data, self.ddpm, cond=cond)
                # Before the backward pass, zero all of the network gradients
                self.optimizer.zero_grad()
                # Backward pass: compute gradient of the loss with respect to parameters
                loss.backward()
                # Perform gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                # Calling the step function to update the parameters
                self.optimizer.step()
                # Update the exponential moving average
                self.ema.update(self.model)
                # Print loss
                list_loss.append(loss.cpu().detach())
                if t % 50 == 0:
                    print(f"Epoch : {n} loss : {np.mean(list_loss)} step : {t}")
                    list_loss = [] 
                t += 1              
            if n % self.model.config.training.save_freq == 0:
                directory = "checkpoints/" + self.model.config.data.dataset + "/" + self.model.config.model.sigma_dist + "/" + self.model_name + "_" + str(n)
                print(directory)
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(self.model.state_dict(), directory + "/model_ckt")
                print("Model saved.")

