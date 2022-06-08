import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_steps = 1000

def make_beta_schedule(schedule='linear', n_steps=1000, start=0.02, end=0.0001):
    """
    Create a schedule of betas
    """
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_steps)
    elif schedule == 'geometric':
        betas = torch.logspace(np.log10(start), np.log10(end), n_steps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_steps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_steps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def get_alphas(schedule="geometric", n_steps=1000, start=0.02, end=0.0001):
    betas = make_beta_schedule(schedule, n_steps, start, end).to(device)
    alphas = 1 - betas
    alphas_prod = torch.cumprod(1 - betas.flip(0), 0).flip(0).to(device)
    alphas_prod_p = torch.cat(([alphas_prod[1:], torch.tensor([1.0]).to(device)])).to(device)
    alphas_bar_sqrt = alphas_prod_p.sqrt().to(device)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod).to(device)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod).to(device)

    return betas, alphas, alphas_prod, alphas_prod_p, alphas_bar_sqrt, one_minus_alphas_bar_log, one_minus_alphas_bar_sqrt

def get_posteriors(betas, alphas, alphas_prod, alphas_prod_p):
    posterior_mean_coef_1 = (betas * torch.sqrt(alphas_prod_p) / (1 - alphas_prod))
    posterior_mean_coef_2 = ((1 - alphas_prod_p) * torch.sqrt(alphas) / (1 - alphas_prod))
    posterior_variance = betas * (1 - alphas_prod_p) / (1 - alphas_prod)
    posterior_log_variance_clipped = torch.log(torch.cat((posterior_variance[1].view(1, 1), posterior_variance[1:].view(-1, 1)), 0)).view(-1)
    return posterior_mean_coef_1, posterior_mean_coef_2, posterior_variance, posterior_log_variance_clipped

class DDPM(object):

    def __init__(self, config):
        self.betas, self.alphas, self.alphas_prod, self.alphas_prod_p, self.alphas_bar_sqrt, self.one_minus_alphas_bar_log, self.one_minus_alphas_bar_sqrt = get_alphas(config.model.sigma_dist, config.model.num_classes, config.model.sigma_begin, config.model.sigma_end)
        self.posterior_mean_coef_1, self.posterior_mean_coef_2, self.posterior_variance, self.posterior_log_variance_clipped = get_posteriors(self.betas, self.alphas, self.alphas_prod, self.alphas_prod_p)
        self.n_steps = config.model.num_classes
    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0).cuda()
        alphas_t = self.extract(self.alphas_bar_sqrt, t, x_0)
        alphas_1_m_t = self.extract(self.one_minus_alphas_bar_sqrt, t, x_0)
        return (alphas_t * x_0 + alphas_1_m_t * noise)

    def q_posterior_mean_variance(self, x_0, x_t, t):
        coef_1 = self.extract(self.posterior_mean_coef_1, t, x_0)
        coef_2 = self.extract(self.posterior_mean_coef_2, t, x_0)
        mean = coef_1 * x_0 + coef_2 * x_t
        var = self.extract(self.posterior_log_variance_clipped, t, x_0)
        return mean, var

class EMA(object):
    def __init__(self, config):
        self.mu = config.model.ema_rate
        self.shadow = {}

    def register(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = type(module)(module.config).to(module.config.device)
        module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict