import torch
from torch import Tensor
from torch_geometric.nn import MessagePassing, MetaLayer
from torch_sparse import SparseTensor
from torch_scatter import scatter

from src.nn.mlp import MLP


class GNLayer(MessagePassing):
    def __init__(self,
                 edge_indim: int,
                 edge_outdim: int,
                 node_indim: int,
                 node_outdim: int,
                 node_aggregator: str = 'add',
                 residual: bool = True,
                 **mlp_params):
        super(GNLayer, self).__init__(aggr=node_aggregator)

        self.edge_model = MLP(input_dim=edge_indim + 2 * node_indim,
                              output_dim=edge_outdim,
                              **mlp_params)
        self.node_model = MLP(input_dim=edge_outdim + node_indim,
                              output_dim=node_outdim,
                              **mlp_params)

        self.residual = residual

    def forward(self,
                nf: torch.tensor,
                ef: torch.tensor,
                edge_idx: torch.tensor):
        nf_residual, ef_residual = nf, ef
        uef = self.edge_update(nf, ef, edge_idx)
        unf = self.propagate(edge_index=edge_idx, x=nf, edge_features=uef)
        if self.residual:
            unf, uef = unf + nf_residual, uef + ef_residual
        return unf, uef

    def edge_update(self, nf, ef, edge_index):
        row, col = edge_index
        x_i, x_j = nf[row], nf[col]
        uef = self.edge_model(torch.cat([x_i, x_j, ef], dim=-1))
        return uef

    def message(self, edge_features: torch.tensor):
        return edge_features

    def message_and_aggregate(self, adj_t: SparseTensor) -> Tensor:
        pass

    def update(self,
               aggr_msg: torch.tensor,
               x: torch.tensor):
        if aggr_msg.shape[0] != x.shape[0]:
            print("d")
        unf = self.node_model(torch.cat([x, aggr_msg], dim=-1))
        return unf


# class MetaGNLayer(MetaLayer):
#     def __init__(self,
#                  edge_indim: int,
#                  edge_outdim: int,
#                  node_indim: int,
#                  node_outdim: int,
#                  node_aggregator: str = 'add',
#                  residual: bool = True,
#                  **mlp_params):
#         super(MetaGNLayer, self).__init__(aggr=node_aggregator)
#
#         self.gn = MetaLayer(EdgeModel(), NodeModel())
#         self.residual = residual
#
#     def forward(self, x, edge_index, edge_attr):
#         nf_residual, ef_residual = x, edge_attr
#         uef, unf = self.propagate(edge_index=edge_idx, x=nf, edge_features=uef)
#         if self.residual:
#             unf, uef = unf + nf_residual, uef + ef_residual
#         return unf, uef
#


class EdgeModel(torch.nn.Module):
    def __init__(self,
                 edge_indim,
                 node_indim,
                 edge_outdim):
        super().__init__()

        self.edge_mlp = MLP(input_dim=edge_indim + 2 * node_indim, output_dim=edge_outdim)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self,
                 edge_indim,
                 node_indim,
                 node_outdim):
        super().__init__()

        self.node_mlp = MLP(input_dim=edge_indim + node_indim, output_dim=node_outdim)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        aggr_msg = torch.zeros(x.shape, device=x.device)
        aggr_msg = scatter(edge_attr, row.view(-1, 1), dim=0, out=aggr_msg)
        out = self.node_mlp(torch.cat([x, aggr_msg], dim=-1))
        # out = torch.cat([x[row], edge_attr], dim=1)
        return out
