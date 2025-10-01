import copy
import torch
import torch.nn.functional as F
from torch_scatter import scatter, scatter_max
from torch_geometric.data import Batch
from torch_geometric.utils.convert import to_networkx


from src.utils.graph_utils import coarsen_graphs, remove_isolated_nodes, assign_set, uncoarsen_graphs
EPS = 1e-3

def evaluate_batch_with_search(test_g, model, connected, max_iter=50, contraction_ratio=0.75, device='cpu', return_history=True, use_search=True):
    original_g = test_g.clone().to(device)
    mapping = torch.arange(test_g.num_nodes+1).reshape(-1, 1).tolist()
    pred_prob_list = []
    mapping_history = []
    gs_history = []

    for n_iter in range(max_iter):
        src, dest = test_g.edge_index
        valid_nodes = torch.cat([src, dest]).unique()
        prediction_node_mask = test_g.m < test_g.k
        prediction_edge_mask = test_g.m[src] < test_g.k[src]
        with torch.no_grad():
            prob = model(test_g.x[prediction_node_mask, :], test_g.edge_attr[prediction_edge_mask, :], test_g.edge_index[:, prediction_edge_mask])
        
        if connected is not None:
            prob = torch.cat([prob, connected.T.reshape(-1, 1)], dim=0)
            
        prob[test_g.ptr[:-1]] = 0
        prob = prob[valid_nodes].float()

        if n_iter == 0:
            pred_prob_list.append(prob.view(-1))

        batch_num_nodes = scatter(torch.ones(valid_nodes.shape, device=test_g.batch.device),
                                    test_g.batch[valid_nodes])

        if batch_num_nodes.max() <= 3:
            break
        
        padded_prob = (-1) * torch.ones(original_g.y.shape, device=prob.device)
        padded_prob[valid_nodes] = prob.view(-1)

        new_size = int(batch_num_nodes.min() * contraction_ratio)
        if new_size >= 3:
            test_g, mapping, bck = coarsen_graphs(test_g, padded_prob, new_size, mapping, use_search=use_search)
            if test_g.num_edges == bck.num_edges:
                break
        else:
            break
        mapping_history.append(copy.deepcopy(mapping))
        gs_history.append(test_g.clone())

    gs, mask, nid_mapping = remove_isolated_nodes(test_g)
    assert gs.batch.shape[0] == prob.shape[0]

    set_assignment = assign_set(gs, prob)
    pred_y = uncoarsen_graphs(original_gs=original_g,
                                reduced_assignment=set_assignment,
                                nid_mapping=nid_mapping,
                                mapping=mapping)

    if return_history:
        return original_g, pred_y, pred_prob_list, mapping_history, gs_history
    return original_g, pred_y, pred_prob_list, None, None


def _invert_mapping(mapping):
    inv = {}
    for u, children in mapping.items():
        for o in children:
            inv[o] = u
    return inv

def _cc_from_edges(nodes_1d, edges_2xn):
    parent = {int(v): int(v) for v in nodes_1d.tolist()}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    if edges_2xn.numel() > 0:
        for a, b in edges_2xn.t().tolist():
            union(int(a), int(b))
    comps = {}
    for v in nodes_1d.tolist():
        r = find(int(v))
        comps.setdefault(r, []).append(v)
    return list(comps.values())

def build_remaining_subsets_with_history(
    mapping, gs, bid, remaining_nodes_global,
    depot_id=0,
    split_by_connectivity=True,   # True: 강한 연결로 세분화
    weight_percentile=0.3         # 상위 30% 간선만 사용(None이면 필터 X)
):
    device = gs.edge_index.device
    bid = int(bid.item()) if hasattr(bid, "item") else int(bid)
    start = int(gs.ptr[bid].item())
    end   = int(gs.ptr[bid+1].item())

    remaining_nodes_global = [x for x in remaining_nodes_global if x != depot_id]
    rem_local = torch.tensor(remaining_nodes_global, device=device) - start
    rem_local = rem_local[(rem_local >= 0) & (rem_local < (end - start))]
    if rem_local.numel() == 0:
        return []

    inv = _invert_mapping(mapping)  # orig -> supernode u
    groups = {}  # u or 'singleton_*' -> [local ids]
    for o_loc in rem_local.tolist():
        o_glob = o_loc + start
        u = inv.get(o_glob, None)
        key = u if u is not None else f"singleton_{o_glob}"
        groups.setdefault(key, []).append(o_loc)

    if not split_by_connectivity:
        return [list(map(int, nodes)) for nodes in groups.values()]

    ei = gs.edge_index
    u_all, v_all = ei[0], ei[1]
    in_batch = (u_all >= start) & (u_all < end) & (v_all >= start) & (v_all < end)
    u_local_all = (u_all[in_batch] - start)
    v_local_all = (v_all[in_batch] - start)

    if weight_percentile is not None and getattr(gs, "edge_attr", None) is not None and u_local_all.numel() > 0:
        w_all = gs.edge_attr[in_batch].view(-1)
        q = torch.quantile(w_all, 1 - weight_percentile) if w_all.numel() > 1 else w_all.min()
        keep = (w_all >= q)
        u_local_all = u_local_all[keep]
        v_local_all = v_local_all[keep]

    edges_local = (torch.stack([u_local_all, v_local_all], dim=0)
                   if u_local_all.numel() > 0 else torch.empty(2, 0, dtype=torch.long, device=device))

    subsets = []
    for nodes in groups.values():
        nodes_t = torch.tensor(nodes, device=device)
        if edges_local.numel() == 0:
            subsets.extend([[int(x)] for x in nodes_t.tolist()])
            continue
        node_set = set(nodes_t.tolist())
        keep = torch.tensor([(int(a.item()) in node_set and int(b.item()) in node_set)
                             for a, b in edges_local.t()], device=device, dtype=torch.bool)
        e_u = edges_local[0][keep]
        e_v = edges_local[1][keep]
        if e_u.numel() == 0:
            subsets.extend([[int(x)] for x in nodes_t.tolist()])
        else:
            comps = _cc_from_edges(nodes_t, torch.stack([e_u, e_v], dim=0))
            subsets.extend([list(map(int, comp)) for comp in comps])

    return subsets
