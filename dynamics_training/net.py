import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 200,
                 seed=0,
                 l2_reg_multiplier=1.,
                 num=0):

        super(Net, self).__init__()
        torch.manual_seed(seed)
        self.net = BayesianNeuralNetwork(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim,
                                         l2_reg_multiplier=l2_reg_multiplier, seed=num)
        self.weights = self.net.weights

    @staticmethod
    def filter_inputs(state, action, state_filter, action_filter):
        state_f = state_filter.filter_torch(state)
        action_f = action_filter.filter_torch(action)
        state_action_f = torch.cat((state_f, action_f), dim=1)
        return state_action_f

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def _train_model_forward(self, x_batch):
        self.net.train()
        self.net.zero_grad()
        x_batch = x_batch.to(device, non_blocking=True)
        y_pred = self.forward(x_batch)
        return y_pred

    def train_model_forward(self, x_batch, delta_batch, r_batch):
        delta_batch, r_batch = delta_batch.to(device, non_blocking=True), r_batch.to(device, non_blocking=True)
        y_pred = self._train_model_forward(x_batch)
        y_batch = torch.cat([delta_batch, r_batch], dim=1)
        loss = self.net.loss(y_pred, y_batch)
        return loss

    def get_predictions_from_loader(self, data_loader, return_targets=False, return_sample=False):
        self.net.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x_batch_val, delta_batch_val, r_batch_val in data_loader:
                x_batch_val, delta_batch_val, r_batch_val = x_batch_val.to(device,
                                                                           non_blocking=True), delta_batch_val.to(
                    device, non_blocking=True), r_batch_val.to(device, non_blocking=True)
                y_pred_val = self.forward(x_batch_val)
                preds.append(y_pred_val)
                if return_targets:
                    y_batch_val = torch.cat([delta_batch_val, r_batch_val],
                                            dim=1)
                    targets.append(y_batch_val)

        preds = torch.vstack(preds)

        if return_sample:
            mu, logvar = preds.chunk(2, dim=1)
            dist = torch.distributions.Normal(mu, logvar.exp().sqrt())
            sample = dist.sample()
            preds = torch.cat((sample, preds), dim=1)

        if return_targets:
            targets = torch.vstack(targets)
            return preds, targets
        else:
            return preds

    def get_validation_loss(self, validation_loader):
        self.net.eval()
        preds, targets = self.get_predictions_from_loader(validation_loader, return_targets=True)
        return self.net.loss(preds, targets, logvar_loss=False).item()

    def get_mean_logvar(self, state: torch.Tensor, action: torch.Tensor, state_filter, action_filter):
        # Used during agent training in rollout generation
        state_action_f = self.filter_inputs(state, action, state_filter, action_filter)
        mean_logvar = self.forward(state_action_f)
        obs_dim = state.shape[1]
        mean_logvar[:, :obs_dim] += state  # as .forward() returned the mean of (nextstate - state) (and logvar)
        return mean_logvar


class BayesianNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int,
                 output_dim: int,
                 hidden_dim: int = 200,
                 l2_reg_multiplier=1.,
                 seed=0):
        super().__init__()

        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        reinitialize_fc_layer_(self.fc1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        reinitialize_fc_layer_(self.fc2)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        reinitialize_fc_layer_(self.fc3)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        reinitialize_fc_layer_(self.fc4)
        self.use_blr = False
        self.delta = nn.Linear(hidden_dim, output_dim)
        reinitialize_fc_layer_(self.delta)
        self.logvar = nn.Linear(hidden_dim, output_dim)
        reinitialize_fc_layer_(self.logvar)
        self.loss = GaussianMSELoss()
        self.activation = nn.SiLU()
        self.lambda_prec = 1.0
        self.max_logvar = None
        self.min_logvar = None
        params = []
        self.layers = [self.fc1, self.fc2, self.fc3, self.fc4, self.delta, self.logvar]
        self.decays = np.array([0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]) * l2_reg_multiplier
        for layer, decay in zip(self.layers, self.decays):
            params.extend(get_weight_bias_parameters_with_decays(layer, decay))
        self.weights = params
        self.to(device)

    def get_l2_reg_loss(self):
        l2_loss = 0
        for layer, decay in zip(self.layers, self.decays):
            for name, parameter in layer.named_parameters():
                if 'weight' in name:
                    l2_loss += parameter.pow(2).sum() / 2 * decay
        return l2_loss

    def update_logvar_limits(self, max_logvar, min_logvar):
        self.max_logvar, self.min_logvar = max_logvar, min_logvar

    def forward(self, x: torch.Tensor):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        delta = self.delta(x)
        logvar = self.logvar(x)
        # Taken from the PETS code to stabilise training
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return torch.cat((delta, logvar), dim=1)


def reinitialize_fc_layer_(fc_layer):
    # Initialize a fc layer to have a truncated normal over the weights, and zero over the biases 
    input_dim = fc_layer.weight.shape[1]
    std = get_trunc_normal_std(input_dim)
    torch.nn.init.trunc_normal_(fc_layer.weight, std=std, a=-2 * std, b=2 * std)
    torch.nn.init.zeros_(fc_layer.bias)


def get_trunc_normal_std(input_dim):
    # Return truncated normal standard deviation required for weight initialization
    return 1 / (2 * np.sqrt(input_dim))


def get_weight_bias_parameters_with_decays(fc_layer, decay):
    # For the fc_layer, extract only the weight from the .parameters() method so we don't regularize the bias terms
    decay_params = []
    non_decay_params = []
    for name, parameter in fc_layer.named_parameters():
        if 'weight' in name:
            decay_params.append(parameter)
        elif 'bias' in name:
            non_decay_params.append(parameter)
    decay_dicts = [{'params': decay_params, 'weight_decay': decay}, {'params': non_decay_params, 'weight_decay': 0.}]
    return decay_dicts


class GaussianMSELoss(nn.Module):
    def __init__(self):
        super(GaussianMSELoss, self).__init__()

    def forward(self, mu_logvar, target, logvar_loss=True):
        mu, logvar = mu_logvar.chunk(2, dim=1)
        inv_var = (-logvar).exp()
        if logvar_loss:
            return (logvar + (target - mu) ** 2 * inv_var).mean()
        else:
            return ((target - mu) ** 2).mean()
