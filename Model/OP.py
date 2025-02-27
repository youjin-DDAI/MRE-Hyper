import torch
import torch.nn as nn
from deepxde.nn import activations
class MLP(nn.Module):
    def __init__(self, layers, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
class FCLayer_batch(nn.Module):
    """
    Fully Connected Layer with dynamically generated weights.
    Supports batched inputs: (batch, pred_num, input_dim)
    """
    def __init__(self, num_in, num_out):
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.weight_size = torch.tensor([num_out, num_in])  # Shape of weight matrix
        self.bias_size = torch.tensor([num_out])  # Shape of bias vector

    def forward(self, x, param):
        """
        Forward pass with dynamic weight and bias.

        Args:
            x: Tensor of shape (batch, pred_num, input_dim)
            param: Tensor of shape (batch, param_size) containing weights and biases.

        Returns:
            Output Tensor of shape (batch, pred_num, output_dim)
        """
        B, P, _ = x.shape  # (batch, pred_num, input_dim)

        # Extract weights and biases for each batch
        w = param[:, :torch.prod(self.weight_size)]  # (batch, weight_size)
        b = param[:, torch.prod(self.weight_size):]  # (batch, bias_size)

        # Reshape parameters
        w = w.reshape(B, self.weight_size[0], self.weight_size[1])  # (batch, output_dim, input_dim)
        b = b.reshape(B, self.bias_size[0])  # (batch, output_dim)

        # Apply affine transformation (batch matrix multiplication)
        return torch.einsum('bpi, bio->bpo', x, w.transpose(1, 2)) + b.unsqueeze(1)  # (batch, pred_num, output_dim)

    def get_param_size(self):
        return torch.prod(self.weight_size) + self.bias_size
class MREHyper(nn.Module):
    def __init__(self, depth_target=3, width_target=128, act_target='sin',
                 depth_hyper=3, width_hyper=128, act_hyper='relu',
                 observation_num=4000, input_dim=3, num_basis=512, final=128, freq=100):
        super(MREHyper, self).__init__()

        self.depth_target = depth_target

        ### Target Network for u
        self.activation_target_u = self.get_activation(act_target)
        self.target_list_u = nn.ModuleList()
        self.param_sizes_u = []

        self.target_list_u.append(FCLayer_batch(width_target, width_target))
        self.param_sizes_u.append(FCLayer_batch(width_target, width_target).get_param_size())

        for _ in range(depth_target - 1):
            self.target_list_u.append(FCLayer_batch(width_target, width_target))
            self.param_sizes_u.append(FCLayer_batch(width_target, width_target).get_param_size())

        self.target_list_u.append(FCLayer_batch(width_target, num_basis))
        self.param_sizes_u.append(FCLayer_batch(width_target, num_basis).get_param_size())

        self.target_list_u.append(FCLayer_batch(num_basis, 1))
        self.param_sizes_u.append(FCLayer_batch(num_basis, 1).get_param_size())

        self.param_size_u = int(sum(self.param_sizes_u))

        ### Target Network for mu
        self.activation_target_mu = self.get_activation(act_target)
        self.target_list_mu = nn.ModuleList()
        self.param_sizes_mu = []

        self.target_list_mu.append(FCLayer_batch(width_target, width_target))
        self.param_sizes_mu.append(FCLayer_batch(width_target, width_target).get_param_size())

        for _ in range(depth_target - 1):
            self.target_list_mu.append(FCLayer_batch(width_target, width_target))
            self.param_sizes_mu.append(FCLayer_batch(width_target, width_target).get_param_size())

        self.target_list_mu.append(FCLayer_batch(width_target, num_basis))
        self.param_sizes_mu.append(FCLayer_batch(width_target, num_basis).get_param_size())

        self.target_list_mu.append(FCLayer_batch(num_basis, final))
        self.param_sizes_mu.append(FCLayer_batch(num_basis, final).get_param_size())

        self.param_size_mu = int(sum(self.param_sizes_mu))

        ### Hypernetwork for u
        self.activation_hyper_u = self.get_activation(act_hyper)

        self.hyper_list_u = nn.Sequential(
            nn.Linear(observation_num, width_hyper),
            self.activation_hyper_u,
            *[layer for _ in range(depth_hyper - 1) for layer in
              [nn.Linear(width_hyper, width_hyper), self.activation_hyper_u]],
            nn.Linear(width_hyper, self.param_size_u)
        )

        ### Hypernetwork for mu
        self.activation_hyper_mu = self.get_activation(act_hyper)
        self.hyper_list_mu = nn.Sequential(
            nn.Linear(observation_num, width_hyper),
            self.activation_hyper_mu,
            *[layer for _ in range(depth_hyper - 1) for layer in
              [nn.Linear(width_hyper, width_hyper), self.activation_hyper_mu]],
            nn.Linear(width_hyper, self.param_size_mu)
        )

        self.FF = nn.Parameter(torch.randn(input_dim, width_target // 2) * freq, requires_grad=False)

        self.mu_output = MLP([final * 3, 64, 1], activation=nn.ReLU())

    def input_encoding(self, x):
        x_proj = torch.matmul(x, self.FF)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, inputs):
        u, y = inputs # (observation_num, 3), (pred_num, 3)
        pred_num = y.size(0)

        y = self.input_encoding(y)
        y_u = y_mu = y.unsqueeze(0).repeat(3, 1, 1)

        u_mean = torch.mean(u.squeeze(0), dim=0)
        u_std = torch.std(u.squeeze(0), dim=0)
        u = ((u.squeeze(0) - u_mean) / u_std).transpose(0, 1)

        # Generate weight parameters from hypernetworks
        weight_u = self.get_param_u(u)
        weight_mu = self.get_param_mu(u)

        # Apply params generated hyper_u to y
        cut = 0
        for i in range(self.depth_target + 1):
            y_u = self.target_list_u[i](y_u, weight_u[:, cut:cut + self.param_sizes_u[i]])
            y_u = self.activation_target_u(y_u)
            cut += self.param_sizes_u[i]

        output_u = self.target_list_u[self.depth_target + 1](y_u, weight_u[:,
                                                                cut:cut + self.param_sizes_u[self.depth_target + 1]])
        output_u = output_u.squeeze(-1).transpose(-1, -2)
        output_u = output_u * u_std.unsqueeze(0) + u_mean.unsqueeze(0)

        # Apply params generated hyper_mu to y
        cut = 0
        for i in range(self.depth_target + 1):
            y_mu = self.target_list_mu[i](y_mu, weight_mu[:, cut:cut + self.param_sizes_mu[i]])
            y_mu = self.activation_target_mu(y_mu)
            cut += self.param_sizes_mu[i]
        output_mu = self.target_list_mu[self.depth_target + 1](y_mu, weight_mu[:,
                                                                   cut:cut + self.param_sizes_mu[self.depth_target + 1]])
        output_mu = self.mu_output(output_mu.permute(1, 0, 2).reshape(pred_num, -1))

        return output_u, output_mu

    def get_param_u(self, data):

        return self.hyper_list_u(data)

    def get_param_mu(self, data):

        return self.hyper_list_mu(data)

    def get_activation(self, act):

        if act == 'tanh':
            return nn.Tanh()
        elif act == 'prelu':
            return nn.PReLU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'sin':
            return activations.get('sin')
        else:
            raise ValueError("Invalid activation function")