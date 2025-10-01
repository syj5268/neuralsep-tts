import torch
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from torch_geometric.data import Batch

from src.utils.bpp_utils import bp_mtl2
from src.utils.graph_utils import calculate_cut_value

import numpy as np
from math import ceil

EPS = 1e-3

###########################################################
# GraphCHiP for FCI separation
###########################################################
def check_fci(original_g, gs_hist, mapping_hist, y, rci_vs, called_from_julia=True, failed_only=False):
    '''
    original_g: Original graph
    gs_hist: List of coarsened graphs
    mapping_hist: List of mappings from coarsened graphs to original graph
    y: Current solution (binary tensor)
    rci_vs: List of RCI violation for batched graphs
    called_from_julia: Whether the function is called from Julia (for formatting)
    failed_only: Whether to check only for groups that failed RCI
    return: List of violated subsets, their violation, and RHS values
    '''
    
    # Check FCI only if RCI violation is not too large
    if rci_vs.max() > 1.0:  
        return [], [], []

    violations, violated_rhs, violated_deltas, violated_subsets = [], [], [], []
    
    y.view(-1).nonzero().view(-1)
    mask = (rci_vs > 0) if failed_only else torch.zeros_like(rci_vs).bool()  # True means excluding
    
    for mapping, gs in zip(mapping_hist[::-1], gs_hist[::-1]): # reverse the order
        gs = gs.to(y.device)
        selected_unique_nodes = (gs.edge_index[0] * y.view(-1)[gs.edge_index[0]]).unique()[1:]
        unselected_unique_nodes = (gs.edge_index[0] * (1 - y.view(-1)[gs.edge_index[0]])).unique()[1:]
        num_groups = scatter(torch.ones(selected_unique_nodes.shape), gs.batch[selected_unique_nodes])
        for bid in (~(mask + (num_groups == 1))).nonzero().view(-1): 
            subsets = []

            # Check for selected nodes (Inside the subset)
            for u in selected_unique_nodes[gs.batch[selected_unique_nodes] == bid]:
                assert len(mapping[u]) > 0 # Empty subset is not allowed

                nids = torch.tensor(mapping[u]) - gs.ptr[bid]
                subsets.append(nids.tolist())
            
            subsets_nodes = set(node for subset in subsets for node in subset)
            all_nodes = set(range(original_g.num_nodes))

            # Add all unselected nodes as singletons
            remaining_nodes = all_nodes - subsets_nodes - {0}
            subsets.extend([[node] for node in remaining_nodes])
            assert len(sum(subsets, [])) == original_g.num_nodes - 1

            categories = torch.zeros(original_g.num_nodes, dtype=torch.long).to(y.device)
            p = 1
            for si in subsets:
                categories[si] = p
                p += 1

            # Calculate the RHS
            onehot = torch.nn.functional.one_hot(categories).T
            batched_g = Batch.from_data_list([original_g.clone() for _ in range(onehot.shape[0])])
            assignment = onehot.clone()
            assignment[0] = 1 - assignment[0]
            assignment = assignment.reshape(-1)
            deltas = calculate_cut_value(batched_g, assignment)
            rs = scatter(batched_g.demand * assignment / batched_g.Q, batched_g.batch)

            ''' Set H and Partition S_i 
            # deltas, rs = [], []
            # # H
            # unselected = 1 - onehot[0]
            # batched_g = Batch.from_data_list([original_g.clone()])
            # deltas.append(calculate_cut_value(batched_g, unselected))
            # rhs = (batched_g.demand * unselected / batched_g.Q).sum().ceil() * 2
            # # rhs = scatter((batched_g.demand * unselected / batched_g.Q), gs.batch)
            # # rhs = rhs.ceil() * 2
            # rs.append(rhs)
            
            # # S_i
            # for partition in onehot[1:]:
            #     deltas.append(calculate_cut_value(batched_g, partition))
            #     rhs = (batched_g.demand * partition / batched_g.Q).sum().ceil() * 2
            #     rs.append(rhs)

            # deltas = torch.stack(deltas).view(-1) 
            # rs = torch.stack(rs)
            '''

            rs = (rs - EPS).ceil() * 2 # Ceiling can be wrongly rounded -> Subtract EPS

            # Check whether the RCI violation exists
            triples = [
                (r - d, r)
                for d, r in zip(deltas, rs)
                if d < r
            ]

            # Exclude if any subset forms a violation
            if triples:
                continue

            v = rs.sum(0) - deltas.sum(0)
            
            if v > -2: # Check for the potential violation
                min_n_subset = 1 
                max_n_subset = original_g.num_nodes - 1 # TODO: Remove
                if min_n_subset < len(subsets) < max_n_subset: 
                    bs = []
                    sub_items = []
                    capacity = original_g.Q[0].item()

                    for si in subsets:
                        if len(si) == 1:
                            bs.append(1.)
                            sub_items.append(original_g.demand[si].sum().item())
                        else:
                            weights = original_g.demand[si].tolist()
                            bs.append(ceil(sum(weights)/ capacity)) 
                            tot = original_g.demand[si].sum().item()

                            # Decompose subset into smaller items
                            while True:
                                if tot <= capacity:
                                    sub_items.append(tot)
                                    break
                                bs.append(1.) # sub_items.append(capacity)
                                tot -= capacity
                    s = sum(subsets, [])  # r(S, Omega)

                    bs.append(bp_mtl2(sub_items, capacity)) # Lower bound by bin-packing

                    try:
                        rhs_ = 2*sum(bs)
                    except:
                        continue
                    assert rhs_ >= rs.sum(0)

                    v = max(rhs_ - deltas.sum(), 0)
                    
                    if v > EPS:
                        if called_from_julia: 
                            find_ss = [(torch.tensor(sub)+1).tolist() for sub in subsets] + [(torch.tensor(s)+1).tolist()]
                            if find_ss not in violated_subsets:
                                if not violations or (violations[-1] != v.item()):
                                    violated_subsets.append(find_ss)
                                    violated_deltas.append(deltas.sum())
                                    violated_rhs.append(rhs_) # More tight -> BPP
                                    violations.append(v.item())
                        else:
                            find_ss = [subsets + s]
                            if find_ss not in violated_subsets:
                                violated_subsets.append([subsets + s])
                                violated_deltas.append(deltas.sum())
                                violated_rhs.append(rhs_) # More tight -> BPP
                                violations.append(v.item())
    # print(f"Add {len(violated_subsets)} FCI cuts")
    return violated_subsets, violations, violated_rhs

###########################################################
# GraphCHiP for RCI separation
###########################################################
def check_rci_more(original_g, gs_hist, mapping_hist, y, rci_vs, failed_only=True):
    '''
    original_g: Original graph
    gs_hist: List of coarsened graphs
    mapping_hist: List of mappings from coarsened graphs to original graph
    y: Current solution (binary tensor)
    rci_vs: List of RCI violation for batched graphs
    failed_only: Whether to check only for groups that failed RCI
    return: List of violated subsets, their violation, and RHS values
    '''
    violations_all, violated_rhs_all, violated_subsets_all = [], [], []
    
    y.view(-1).nonzero().view(-1)
    mask = (rci_vs > 0) if failed_only else torch.zeros_like(rci_vs).bool()  # True means excluding
    
    for mapping, gs in zip(mapping_hist[::-1], gs_hist[::-1]): # reverse the order
        gs = gs.to(y.device)
        selected_unique_nodes = (gs.edge_index[0] * y.view(-1)[gs.edge_index[0]]).unique()[1:]
        num_groups = scatter(torch.ones(selected_unique_nodes.shape), gs.batch[selected_unique_nodes])

        for bid in (~(mask + (num_groups == 1))).nonzero().view(-1): 
            subsets = []
            for u in selected_unique_nodes[gs.batch[selected_unique_nodes] == bid]:
                assert len(mapping[u]) > 0 # Empty subset is not allowed

                nids = torch.tensor(mapping[u]) - gs.ptr[bid]
                subsets.append(nids.tolist())

            categories = torch.zeros(original_g.num_nodes, dtype=torch.long).to(y.device)
            p = 1
            for si in subsets:
                categories[si] = p
                p += 1

            onehot = torch.nn.functional.one_hot(categories).T
            batched_g = Batch.from_data_list([original_g.clone() for _ in range(onehot.shape[0])])
            assignment = onehot.clone()
            assignment[0] = 1 - assignment[0]
            assignment = assignment.reshape(-1)

            deltas = calculate_cut_value(batched_g, assignment)[1:]
            rs = scatter(batched_g.demand * assignment / batched_g.Q, batched_g.batch)[1:]
            rs = (rs - 1e-3).ceil()  # Ceiling can be wrongly rounded -> Subtract EPS
            
            triples = [
                (s, 2*r - d, r)
                for d, r, s in zip(deltas, rs, subsets)
                if d < 2*r
            ]

            if triples:
                mask[bid] = True
                violated_subsets, violations, violated_rhs = zip(*triples)
                violated_subsets = [[(torch.tensor(sub)+1).tolist() for sub in s] for s in violated_subsets]
                violated_rhs = [r.item() for r in violated_rhs]

                if violated_subsets_all:
                    for s, v, rhs in zip(violated_subsets, violations, violated_rhs):
                        if s not in violated_subsets_all:
                            violated_subsets_all.append(s)
                            violations_all.append(v)
                            violated_rhs_all.append(rhs)
                else:
                    violated_subsets_all.extend(violated_subsets)
                    violations_all.extend(violations)
                    violated_rhs_all.extend(violated_rhs)

            else:
                violated_subsets, violations, violated_rhs = [], [], []

    if violated_subsets_all:
        violated_subsets_all = [np.array(s) for s in violated_subsets_all]
    # print(f"Add {len(violated_subsets_all)} more RCI cuts")
    return violated_subsets_all, violations_all, violated_rhs_all