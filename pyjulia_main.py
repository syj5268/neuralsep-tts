import os
from box import Box
from time import perf_counter

import numpy as np

import random
random.seed(1234)
from math import ceil

import pickle
import torch
from torch_scatter import scatter
from torch_geometric.data import Batch

from src.model.coarsening_net import CoarseningNet
from src.utils.train_utils import set_seed
from src.utils.eval_utils import evaluate_batch_with_search
from src.utils.graph_utils import calculate_cut_value
from src.utils.data_utils import get_pyg_graph, get_pyg_from_julia
from src.utils.graphchip import check_fci, check_rci_more

EPS = 1e-6


def get_rci_with_coarsening(edge, edge_x, demand, capacity, use_pi_greedy=True, use_graphchip_rci=True, return_fci=False):
    device = 'cpu' # 'cuda:0'
    set_seed(seed=1234, use_cuda='cuda' in device)

    model = CoarseningNet(node_dim = 2,
                          edge_dim = 1,
                          latent_dim = 128,
                          n_gn_layer = 5)
    
    model.load_state_dict(torch.load("../../bestmodels/coarsening_best.pt", map_location=device))

    model = model.to(device)
    model.eval()

    demand = torch.tensor(demand[:-1], dtype=torch.float)
    n = len(demand) - 1  # number of customers

    g_list = get_pyg_from_julia(n, edge, edge_x, s_hat=None, demand=demand, capacity=capacity) 

    connected_prob = None
    gs = Batch.from_data_list(g_list).to(device)

    ##########################################################################
    ### Predict with coarsening net 
    # Use pi-greedy search to improve prediction (option: use_pi_greedy)
    ###########################################################################
    original_gs, pred_y, _, mapping_hist, gs_hist = evaluate_batch_with_search(gs, model, connected_prob, return_history=True, use_search=use_pi_greedy)
    pred_cut = calculate_cut_value(original_gs, pred_y) 

    rhs = scatter(original_gs.demand * pred_y / original_gs.Q, original_gs.batch).ceil()
    vs = torch.clamp(rhs * 2 - pred_cut, 0)
    pred_y = pred_y.reshape(original_gs.num_graphs, -1)
    duplicated_cuts = 0
    list_s, list_rhs, list_z = [], [], []
    list_string = []
    for idx, v in enumerate(vs.tolist()):
        if v > EPS:
            s = torch.nonzero(pred_y[idx]).view(-1) + 1  # to julia index
            s = np.array(s.tolist())  # tensor or list works improperly when interating with julia
            if str(s) not in list_string: # remove duplicates 
                list_s.append(s)
                list_string.append(str(s))
                list_rhs.append(rhs[idx].item())
                list_z.append(pred_cut[idx].item())
            else:
                duplicated_cuts += 1


    ##############################################################################
    # Use GraphChip to find more RCI (option: use_graphchip_rci)
    ##############################################################################
    if use_graphchip_rci:
        rci_vs = vs
        subset_more, violation_more, rhs_more = check_rci_more(g_list[0], gs_hist, mapping_hist, pred_y, rci_vs, failed_only=True)
        if subset_more:
            list_s.extend(subset_more)
            list_rhs.extend(rhs_more)

    ##############################################################################
    # Use GraphChip to find FCI (option: return_fci)
    ##############################################################################
    if return_fci:
        rci_vs = vs 
        fci_bpp_subsets, fci_bpp_violations, fci_bs = check_fci(g_list[0], gs_hist, mapping_hist, pred_y, rci_vs, failed_only=False)
    else:
        fci_bpp_subsets, fci_bpp_violations, fci_bs = [], [], []

    if return_fci:
        return list_s, list_rhs, fci_bpp_subsets, fci_bpp_violations, fci_bs

    return list_s, list_rhs, [], [], []
