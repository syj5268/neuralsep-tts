import os
import pickle
import torch
import torch_geometric.utils as geo_utils
from torch_geometric.data import Data
from tqdm import tqdm

import numpy as np
import networkx as nx


def get_pyg_from_julia(n, edges, x, s_hat, demand, capacity):
    gs = []

    src, dest, ef = [], [], []
    for idx, (u, v) in enumerate(edges):
        src.extend([u - 1, v - 1])
        dest.extend([v - 1, u - 1])
        ef.extend([x[idx], x[idx]])

    for j in range(2, n + 2):
        if [1, j] not in edges:  # and [j, 1] not in edges:  # julia index
            src.extend([0, j-1])
            dest.extend([j - 1, 0])
            ef.extend([0., 0.])

    edge_idx = torch.tensor([src, dest], dtype=torch.long)
    edge_attr = torch.tensor(ef, dtype=torch.float).view(-1, 1)

    num_vehicles = torch.ceil(demand.sum() / capacity)
    num_vehicles = num_vehicles.int().item()


    for m in range(num_vehicles):
        x = torch.stack([demand / capacity, m / num_vehicles * torch.ones_like(demand)], dim=-1)
        y = torch.zeros(demand.shape)
        g = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y, demand=demand,
                    m=m * torch.ones(demand.shape),
                    k=num_vehicles * torch.ones(demand.shape),
                    Q=capacity * torch.ones(demand.shape))
        g.edge_index, g.edge_attr = geo_utils.sort_edge_index(g.edge_index, g.edge_attr)
        gs.append(g)

    return gs


def get_nx_graph(edges, ):
    edge_from, edge_to, edge_weight = [], [], []

    g = nx.Graph()

    for idx, (u, v) in enumerate(edges):
        g.add_edge(u-1, v-1)
    return g


def get_pyg_graph(n, edges, x, s_hat, demand, capacity, vehicles, max_only=False, add_reverse=False):
    gs = []

    src, dest, ef = [], [], []
    for idx, (u, v) in enumerate(edges):
        if add_reverse:
            src.extend([u - 1, v - 1])
            dest.extend([v - 1, u - 1])
            ef.extend([x[idx], x[idx]])
        else:
            src.append(u - 1)
            dest.append(v - 1)
            ef.append(x[idx])

    for j in range(2, n + 2):
        if [1, j] not in edges: # and [j, 1] not in edges:  # julia index
            if add_reverse:
                src.extend([0, j - 1])
                dest.extend([j - 1, 0])
                ef.extend([0., 0.])
            else:
                src.append(0)
                dest.append(j - 1)
                ef.append(0.)

    edge_idx = torch.tensor([src, dest], dtype=torch.long)
    edge_attr = torch.tensor(ef, dtype=torch.float).view(-1, 1)

    num_repeat = 1

    if vehicles == 0:  # unlabeled data (from julia call)
        num_vehicles = torch.ceil(demand.sum() / capacity)
        num_vehicles = num_vehicles.int().item()

        if max_only:
            x = torch.stack([demand / capacity], dim=-1)
            y = torch.zeros(demand.shape)
            g = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y, demand=demand,
                        m=m * torch.ones(demand.shape),
                        k=num_vehicles * torch.ones(demand.shape),
                        Q=capacity * torch.ones(demand.shape))
            g.edge_index, g.edge_attr = geo_utils.sort_edge_index(g.edge_index, g.edge_attr)
            gs.append(g)
        else:
            for m in range(num_vehicles):
                x = torch.stack([demand / capacity, m / num_vehicles * torch.ones_like(demand)], dim=-1)
                y = torch.zeros(demand.shape)
                g = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y, demand=demand,
                            m=m * torch.ones(demand.shape),
                            k=num_vehicles * torch.ones(demand.shape),
                            Q=capacity * torch.ones(demand.shape))
                g.edge_index, g.edge_attr = geo_utils.sort_edge_index(g.edge_index, g.edge_attr)
                gs.append(g)

    else:  # labeled data
        for m, label in enumerate(s_hat):
            if max_only:
                x = torch.stack([demand / capacity], dim=-1)
            else:
                x = torch.stack([demand / capacity, m / vehicles * torch.ones_like(demand)], dim=-1)
            y = torch.tensor([int(i+1 in label) for i in range(n+1)])

            g = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y, demand=demand,
                        m=m * torch.ones(demand.shape),
                        k=vehicles * torch.ones(demand.shape),
                        Q=capacity * torch.ones(demand.shape))
            g.edge_index, g.edge_attr = geo_utils.sort_edge_index(g.edge_index, g.edge_attr)
            gs.append(g)

    return gs


def get_pyg_data(data_dir, save_dir=None, max_only=False):
    with open(data_dir, 'rb') as f:
        data_list = pickle.load(f)

    graph_list = []
    for data in tqdm(data_list):
        n = data['customers']
        edges = data['edges']
        x = data['x_bar']
        s_hat = data['s_hat']
        demand = torch.tensor(data['demand'])#[:,None]
        capacity = data['capacity']
        vehicles = data['vehicles']
        if max_only:
            max_idx = torch.argmax(torch.tensor(data['violation']))
            s_hat = [s_hat[max_idx]]

        gs = get_pyg_graph(n, edges, x, s_hat, demand, capacity, vehicles, max_only)
        graph_list.extend(gs)

    torch.save(graph_list, save_dir)


def save_rci_graphs(data_dir, save_dir=None):
    instances = os.listdir(data_dir)
    instances = [file for file in instances if file.endswith('n200.pickle')]

    save_dir = save_dir if save_dir is not None else os.path.join(data_dir, "processed.pt")

    gs = []
    for instance in tqdm(instances):
        with open(os.path.join(data_dir, instance), 'rb') as f:
            data = pickle.load(f)

        num_vehicles = data['k']
        capacity = data['capacity']
        n = len(data['demand']) - 1
        demand = data['demand'][:n]

        if sum(demand == 0) > 1:
            # print(instance, "has 0 demand.")
            demand[demand == 0] = 1
            demand[0] = 0

        demand = torch.tensor(demand, dtype=torch.float)

        for step, e in enumerate(data['list_e']):
            x = data['list_x'][step]
            s_hat = data['list_s'][step]

            src, dest, ef = [], [], []
            for idx, (u, v) in enumerate(e):
                src.extend([u - 1, v - 1])
                dest.extend([v - 1, u - 1])
                ef.extend([x[idx], x[idx]])

            for j in range(2, n + 1):
                if (1, j) not in e:  # julia index
                    src.extend([0, j - 1])
                    dest.extend([j - 1, 0])
                    ef.extend([0.0, 0.0])

            edge_idx = torch.tensor([src, dest], dtype=torch.long)
            edge_attr = torch.tensor(ef, dtype=torch.float).view(-1, 1)

            for m, label in enumerate(s_hat):
                x = torch.stack([demand / capacity, m / num_vehicles * torch.ones_like(demand)], dim=-1)

                y = torch.tensor([int(i in label) for i in range(n)])

                g = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=y, demand=demand,
                         m=m * torch.ones(demand.shape),
                         k=num_vehicles * torch.ones(demand.shape),
                         Q=capacity * torch.ones(demand.shape))
                g.edge_index, g.edge_attr = geo_utils.sort_edge_index(g.edge_index, g.edge_attr)
                gs.append(g)

        torch.save(gs, save_dir)


def dgl_to_pyg_data(data_dir, save_dir):
    from dgl.data.utils import load_graphs

    sizes = [50, 75, 100, 200]
    for size in sizes:
        dgl_gs, _ = load_graphs(os.path.join(data_dir, "data_test_{}.bin".format(size)))
        gs = []
        for dgl_g in tqdm(dgl_gs):
            edge_idx = dgl_g.adj()._indices()
            edge_attr = dgl_g.edata['x_val'].float().view(-1, 1)
            demand = dgl_g.ndata['demand'].float().view(-1)
            capacity = dgl_g.ndata['capacity'].float().view(-1)

            s_hat = dgl_g.ndata['s_hat'].T
            num_vehicles = s_hat.shape[0]

            for m, label in enumerate(s_hat):
                x = torch.stack([demand / capacity, m / num_vehicles * torch.ones_like(demand)], dim=-1)

                g = Data(x=x, edge_index=edge_idx, edge_attr=edge_attr, y=label, demand=demand,
                         m=m * torch.ones(demand.shape),
                         k=num_vehicles * torch.ones(demand.shape),
                         Q=capacity * torch.ones(demand.shape))
                g.edge_index, g.edge_attr = geo_utils.sort_edge_index(g.edge_index, g.edge_attr)
                gs.append(g)
        torch.save(gs, save_dir.format(size))

