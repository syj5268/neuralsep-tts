import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch


class GraphProbDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.collate_fn
        super().__init__(*args, **kwargs)

    def collate_fn(self, batch):
        gs = []
        for g_idx, g in enumerate(batch):
            for i in range(1, g.num_nodes):
                copied = g.clone()
                initial_prob = 0.5 * torch.ones(g.num_nodes)
                initial_prob[0] = 0
                initial_prob[i] = 1
                copied.x = torch.cat([g.x, initial_prob.view(-1, 1)], dim=-1)
                copied.g_idx = g_idx
                gs.append(copied)
        batched_gs = Batch.from_data_list(gs)
        return batched_gs