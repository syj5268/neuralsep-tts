import torch
import torch.nn as nn

from torch_geometric.nn import MetaLayer
from src.nn.gn import GNLayer, NodeModel, EdgeModel
from src.nn.mlp import MLP


class CoarseningNet(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 latent_dim: int,
                 n_gn_layer: int,
                 gn_mlp_params: dict = {},
                 policy_mlp_params: dict = {}):
        super(CoarseningNet, self).__init__()

        if policy_mlp_params == {}:
            policy_mlp_params = {'out_act': 'Sigmoid'}

        self.node_encoder = nn.Linear(node_dim, latent_dim)
        self.edge_encoder = nn.Linear(edge_dim, latent_dim)

        self.gn_layers = nn.ModuleList([MetaLayer(EdgeModel(edge_indim=latent_dim, node_indim=latent_dim,
                                                            edge_outdim=latent_dim),
                                                  NodeModel(edge_indim=latent_dim, node_indim=latent_dim,
                                                            node_outdim=latent_dim)) for _ in range(n_gn_layer)])

        self.policy = MLP(input_dim=latent_dim,
                          output_dim=1, **policy_mlp_params)

    def forward(self, nf, ef, edge_index):
        unf, uef = self.node_encoder(nf), self.edge_encoder(ef)
        nf_residual, ef_residual = unf, uef
        for layer in self.gn_layers:
            unf, uef, _ = layer(x=unf, edge_index=edge_index, edge_attr=uef)
            unf, uef = unf + nf_residual, uef + ef_residual

        unf = self.policy(unf)
        return torch.clamp(unf, 0., 1.)
