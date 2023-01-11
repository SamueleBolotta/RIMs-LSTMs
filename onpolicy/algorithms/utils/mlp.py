import torch.nn as nn
import torch
from .util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), active_func)
        self.fc_h = nn.Sequential(init_(
            nn.Linear(hidden_size, hidden_size)), active_func)
        self.fc2 = get_clones(self.fc_h, self._layer_N)

    def forward(self, x):
        print("forward mlp layer: x before fc1", x)
        x = self.fc1(x)
        print("forward mlp layer: x after fc1", x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
            print("forward mlp layer: x after fc2", x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size

        obs_dim = obs_shape[0]

        self.mlp = MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        print("forward mlp: x before mlp", x)
        obs_isnan_mask = torch.isnan(x)
        print("forward mlp: nan mask x", obs_isnan_mask)
        obs_num_nans = torch.sum(obs_isnan_mask)
        print("forward mlp: number of nans", obs_num_nans)
        x = self.mlp(x)
        print("forward mlp: x after mlp", x)
        return x
