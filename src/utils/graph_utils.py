import torch
import torch_geometric.utils as geo_utils
from torch_scatter import scatter, scatter_max, scatter_add


def get_edges(g, nodes, direction='both', return_batch=False):
    src, dest = g.edge_index
    edge_mask = torch.zeros(src.shape, device=g.x.device)
    batch = g.batch[src]

    for node in nodes:
        if direction == 'in':
            edge_mask += (dest == node).int()
        elif direction == 'out':
            edge_mask += (src == node).int()
        else:  # 'both'
            edge_mask += (dest == node).int()
            edge_mask += (src == node).int()

    if return_batch:
        return torch.nonzero(edge_mask).view(-1), batch[edge_mask.bool()]
    return torch.nonzero(edge_mask).view(-1)


def coarsen_graphs(gs, s_prob, reduced_size, mapping, using_label=False, use_search=True):
    copied_gs = gs.clone().to(gs.x.device)
    u, v = gs.edge_index
    q = s_prob[u] * s_prob[v] + (1 - s_prob[u]) * (1 - s_prob[v]) # s_prob : dtype=torch.Int
    q[get_edges(gs, gs.ptr[:-1])] = 0  # depot connected (depots are the first node of each batch)
    q = q.float()

    # sort by row
    gs.edge_index, (gs.edge_attr, q) = geo_utils.sort_edge_index(gs.edge_index, [gs.edge_attr, q])
    reduced_nodes = torch.unique(torch.cat([gs.edge_index[0], gs.edge_index[1]]))
    batch_num_nodes = scatter(torch.ones(reduced_nodes.shape, device=reduced_nodes.device), gs.batch[reduced_nodes])

    epsilon = 1e-3
    #################################################################################
    # Other Options: use_search_roulette, use_search_softmax
    use_search_roulette = False
    use_search_softmax = False
    #################################################################################

    while max(batch_num_nodes) > reduced_size:   
        
        if use_search:
            random_adjustment = torch.rand(q.shape).to(q.device) * epsilon # [0.001, 0.01]
            q_new = torch.where(q == 0, q, q + random_adjustment).to(q.device) 
            out, eid = scatter_max(q_new.view(-1), gs.batch[gs.edge_index[0]].view(-1)) # Size: batch_size
        
        elif not use_search and use_search_roulette:
            batch_indices = gs.batch[reduced_nodes].unique()
            eid = torch.full((len(batch_indices),), -1, dtype=torch.long, device=q.device)
            out = torch.full((len(batch_indices),), -1, dtype=torch.float64, device=q.device)
            for batch_idx in batch_indices: # Iterate over each batch and perform probability-based edge sampling
                batch_edge_indices = (gs.batch[gs.edge_index[0]] == batch_idx).nonzero(as_tuple=True)[0]
                batch_edge_probs = q[batch_edge_indices]
                if batch_edge_probs.sum() > 0: # Not all edges are zero
                    normalized_probs = batch_edge_probs / batch_edge_probs.sum()
                    sampled_index = torch.multinomial(normalized_probs, num_samples=1)            
                    eid[batch_idx] = batch_edge_indices[sampled_index]
                    out[batch_idx] = batch_edge_probs[sampled_index]
        
        elif not use_search and use_search_softmax:
            batch_indices = gs.batch[reduced_nodes].unique()
            eid = torch.full((len(batch_indices),), -1, dtype=torch.long, device=q.device)
            out = torch.full((len(batch_indices),), -1, dtype=torch.float64, device=q.device)
            for batch_idx in batch_indices: # Iterate over each batch and perform probability-based edge sampling
                batch_edge_indices = (gs.batch[gs.edge_index[0]] == batch_idx).nonzero(as_tuple=True)[0]
                batch_edge_probs = q[batch_edge_indices]
                if batch_edge_probs.sum() > 0: # Not all edges are zero
                    probabilities = torch.softmax(batch_edge_probs / 0.25, dim=0)
                    sampled_index = torch.multinomial(probabilities, num_samples=1)      
                    eid[batch_idx] = batch_edge_indices[sampled_index]
                    out[batch_idx] = batch_edge_probs[sampled_index]
        else:
            out, eid = scatter_max(q.view(-1), gs.batch[gs.edge_index[0]].view(-1)) # Size: batch_size

        valid_batch = (out > 0) & (batch_num_nodes > reduced_size)
        if not valid_batch.int().sum():
            break
        eid = eid[valid_batch]  # exclude if q[eid] == 0
        # gs.edge_index[:,eid] - gs.ptr[:-1]
        gs, q, mapping = _contract_edges(gs, eid, q, mapping, using_label=using_label) 
        reduced_nodes = torch.unique(torch.cat([gs.edge_index[0], gs.edge_index[1]]))  # torch.unique(gs.edge_index[0])
        batch_num_nodes = scatter(torch.ones(reduced_nodes.shape, device=reduced_nodes.device), gs.batch[reduced_nodes])
    
    return gs, mapping, copied_gs


def _contract_edges(gs, selected_edges, q, mapping, using_label=False):
    us, vs = gs.edge_index[:, selected_edges]  # us are merged into vs
    padded_vs = (-1) * torch.ones(gs.num_graphs, device=vs.device).long()
    padded_vs[gs.batch[vs]] = vs

    # update mapping info
    for u, v in zip(us, vs):
        if using_label:
            assert gs.y[u] == gs.y[v]
        mapping[v].extend(mapping[u])
        mapping[u] = []

    # add edges to vs
    in_edge_idx = get_edges(gs, us, direction='in')
    in_edges = gs.edge_index[:, in_edge_idx]
    new_in_edges = torch.stack([in_edges[0], padded_vs[gs.batch[in_edges[1]]]])
    new_in_attr = gs.edge_attr[in_edge_idx]
    new_in_q = q[in_edge_idx]
    q[in_edge_idx] = 0

    out_edge_idx = get_edges(gs, us, direction='out')
    out_edges = gs.edge_index[:, out_edge_idx]
    new_out_edges = torch.stack([padded_vs[gs.batch[out_edges[0]]], out_edges[1]])
    new_out_attr = gs.edge_attr[out_edge_idx]
    new_out_q = q[out_edge_idx]
    q[out_edge_idx] = 0

    edge_index = torch.cat([gs.edge_index, new_in_edges, new_out_edges], dim=1)
    edge_attr = torch.cat([gs.edge_attr, new_in_attr, new_out_attr], dim=0)
    q = torch.cat([q, new_in_q, new_out_q], dim=0)

    # remove self-loop first
    self_loop_mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, self_loop_mask]
    edge_attr, q = edge_attr[self_loop_mask], q[self_loop_mask]

    # add counter
    counter = torch.ones_like(q, device=q.device)
    # simplify the multi graph
    edge_index, (edge_attr, q, counter) = geo_utils.coalesce(edge_index, [edge_attr, q, counter], reduce="add")
    q = q / counter  # mean

    # update features
    gs.demand[vs] += gs.demand[us]
    gs.demand[us] = 0

    # make subgraph's info without us
    mask = (gs.demand > 0)  # torch.ones(gs.x.shape[0], device=gs.x.device)
    mask[gs.ptr[:-1]] = True  # include depots
    edge_index, edge_attr, edge_mask = geo_utils.subgraph(subset=torch.nonzero(mask)[:, 0],
                                                          edge_index=edge_index,
                                                          edge_attr=edge_attr,
                                                          return_edge_mask=True)
    q = q[edge_mask]

    # update node, edge features
    gs.x[:, 0] = gs.demand / gs.Q
    gs.edge_index, gs.edge_attr = edge_index, edge_attr
    return gs, q, mapping


def remove_isolated_nodes(gs, clone_flag=True):
    if clone_flag:
        rst_gs = gs.clone()
    else:
        rst_gs = gs

    if not gs.has_isolated_nodes():
        return rst_gs, None, torch.arange(rst_gs.num_nodes, device=rst_gs.x.device)

    src, dest = rst_gs.edge_index
    mask = torch.cat([src, dest]).unique()  # symmetric edges

    nid_mapping = torch.arange(rst_gs.num_nodes, device=src.device)[mask]

    rst_gs.x = rst_gs.x[mask]
    rst_gs.y = rst_gs.y[mask]
    rst_gs.demand = rst_gs.demand[mask]
    rst_gs.m = rst_gs.m[mask]
    rst_gs.k = rst_gs.k[mask]
    rst_gs.Q = rst_gs.Q[mask]
    rst_gs.batch = rst_gs.batch[mask]

    count = scatter(torch.ones_like(rst_gs.batch), rst_gs.batch)
    rst_gs.ptr = torch.cat([torch.tensor([0], device=count.device), torch.cumsum(count, dim=0)])

    return rst_gs, mask, nid_mapping


def assign_set(g, prediction):
    prediction = prediction.view(-1)
    assign = prediction.round()

    n_assigned = scatter(assign, g.batch)
    b_ids = torch.nonzero(n_assigned == 0).view(-1)

    if len(b_ids) > 0:
        adjusted = prediction + torch.rand(prediction.shape).to(prediction.device) * 0.001
        adjusted[g.ptr[:-1]] = 0  # depot
        _, eid = scatter_max(adjusted, g.batch)
        assign[eid[b_ids]] = 1

    return assign.long()


def uncoarsen_graphs(original_gs, reduced_assignment, nid_mapping, mapping):
    lifted_y = torch.zeros(original_gs.demand.shape, device=original_gs.demand.device)

    for nid in nid_mapping[torch.nonzero(reduced_assignment).view(-1)]:
        lifted_y[mapping[nid]] = 1.

    return lifted_y.long()


def calculate_expected_violation(gs, prob, ceil=True):
    cust = torch.nonzero(gs.demand).view(-1)
    cs = gs.subgraph(cust)
    us, vs = cs.edge_index
    us, vs = gs.edge_index
    q = (prob[us] * (1 - prob[vs]) + (1 - prob[us]) * prob[vs]) / 2
    pred_cut = scatter(q.view(-1) * gs.edge_attr.view(-1), gs.batch[us])

    rounded_weight = scatter((gs.demand/gs.Q).view(-1) * prob.round().view(-1), gs.batch)
    vertex_weight_sum = scatter((gs.demand/gs.Q).view(-1) * prob.view(-1), gs.batch)

    if ceil:
        return vertex_weight_sum.ceil() * 2 - pred_cut, pred_cut, vertex_weight_sum
    else:
        return vertex_weight_sum * 2 - pred_cut, pred_cut, vertex_weight_sum


def calculate_cut_value(gs, assignment):
    us, vs = gs.edge_index
    q = (assignment[us] * (1 - assignment[vs]) + (1 - assignment[us]) * assignment[vs]) / 2 ### divide by 2
    cut = scatter_add(q.view(-1) * gs.edge_attr.view(-1), gs.batch[us])  # directed -> indirected
    cut = torch.round(cut * 10**6) / 10**6 
    return cut
