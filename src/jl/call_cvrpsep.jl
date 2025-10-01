using CVRPLIB
using CVRPSEP


function find_rci(demand, capacity, edge_tail, edge_head, edge_x; max_n_cuts = 100)
    cut_manager = CutManager()

    S, RHS = rounded_capacity_cuts!(
        cut_manager,
        demand,
        capacity,
        edge_tail,
        edge_head,
        edge_x,
        integrality_tolerance = 1e-3,
        max_n_cuts = max_n_cuts
    )

    return S, RHS
end


function find_fci(demand, capacity, edge_tail, edge_head, edge_x; max_n_cuts = 100, max_n_tree_nodes=10)
    cut_manager = CutManager()

    S, RHS = framed_capacity_inequalities!(
        cut_manager,
        demand,
        capacity,
        edge_tail,
        edge_head,
        edge_x,
        max_n_tree_nodes = max_n_tree_nodes,
        max_n_cuts = max_n_cuts
    )

    return S, RHS
end
