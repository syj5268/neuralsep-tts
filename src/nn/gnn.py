import torch
import torch.nn as nn
from torch_geometric.nn import Sequential

from src.nn.gn import GNLayer


class GNN(nn.Module):
    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 latent_dim: int,
                 n_layer: int,
                 gn_params: dict = None):
        super(GNN, self).__init__()

        if gn_params is None:
            gn_params = {}

        self.node_encoder = nn.Linear(node_dim, latent_dim)
        self.edge_encoder = nn.Linear(edge_dim, latent_dim)

        gn = GNLayer  # AttnGN if use_attn else HeteroGN
        self.gn_layers = Sequential(
            *tuple([gn(dim=latent_dim, **gn_params) for _ in range(n_layer)])
        )

        # self.policy = MLP(input_dim=latent_dim, output_dim=1, **policy_params)

    def forward(self, g: dgl.graph) -> torch.tensor:
        unf = self.node_encoder(g.ndata['nf'].float())
        uef = self.edge_encoder(g.edata['ef'].float())
        unf, uef = self.gn_layers(g, unf, uef)

        return unf, uef