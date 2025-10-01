using CPLEX, JuMP, Graphs
using CVRPLIB 
using CVRPSEP # ] add https://github.com/chkwon/CVRPSEP.jl
using LinearAlgebra
using Statistics
using PyCall
import Base.@kwdef

const py_dir = "../.." 

rounded_capacity_rhs(S::Vector{Int64}, mcvrp::CVRP) = ceil(sum(mcvrp.demand[S]) / mcvrp.capacity)

_edges(g) = Tuple.(collect(edges(g)))

_edge(i::Int, j::Int) = i < j ? (i, j) : (j, i)

δ(g::Graph, i::Int) = [_edge(i, j) for j in neighbors(g, i)]

function δ(g::Graph, S::Vector{Int})
    ret = Set{Tuple{Int, Int}}()
    for i in S
        for j in neighbors(g, i)
            if !in(j, S)
                push!(ret, _edge(i, j))
            end
        end
    end
    return ret
end

function inner(S::Vector{Int})
    ret = Set{Tuple{Int, Int}}()
    for i in 1:length(S)-1
        for j in i+1:length(S)
            push!(ret, _edge(S[i], S[j]))
        end
    end
    return ret
end

function between(S::Vector{Int}, T::Vector{Int})
    ret = Set{Tuple{Int, Int}}()
    for i in 1:length(S)
        for j in 1:length(T)
            push!(ret, _edge(S[i], T[j]))
        end
    end
    return ret
end

function complement(S::Vector{Int}, node_num::Int)
    all = collect(1:node_num)
    return setdiff(all, union(1, S))
end

function subtour_customers(edge_tail, edge_head, edge_x, cvrp)
    gx = Graph(cvrp.dimension)
    for i in eachindex(edge_x)
        if edge_x[i] > 0.0
            add_edge!(gx, edge_tail[i], edge_head[i])
        end
    end
    cc = connected_components(gx)
    sub_cc = filter(x -> !in(cvrp.depot, x), cc)
    return sub_cc
end


function exact_rounded_capacity_cuts(edge_x::Matrix{Float64}, optimizer, cvrp::CVRP)
    n_nodes = size(edge_x)[1]
    cut_counter = 0

    list_S = Vector{Int}[]
    list_rhs = Float64[]

    all_S = Vector{Int}[]
    all_rhs = Float64[]
    all_z = Float64[]

    for M in 0:ceil(Int, sum(cvrp.demand) / cvrp.capacity) - 1
        m = Model(optimizer)
        @variable(m, w[1:n_nodes, 1:n_nodes] >= 0)
        @variable(m, y[1:n_nodes], Bin)
        @objective(m, Min, sum(edge_x[i,j] * w[i,j] for i in 1:n_nodes, j in 1:n_nodes))
        for i in 1:n_nodes, j in 1:n_nodes
            @constraint(m, w[i, j] >= y[j] - y[i])
        end
        @constraint(m, y[cvrp.depot] == 0)
        @constraint(m, sum(cvrp.demand[i] * y[i] for i in cvrp.customers) >= M * cvrp.capacity + 1)
        optimize!(m)
        z = objective_value(m)

        yy = round.(Int, value.(y))
        S = findall(x -> x == 1, yy)
        rhs = rounded_capacity_rhs(S, cvrp)
        push!(all_S, S)
        push!(all_rhs, rhs)
        push!(all_z, z)

        if z < 2 * (M + 1) - 1e-6
            yy = round.(Int, value.(y))
            S = findall(x -> x == 1, yy)
            rhs = rounded_capacity_rhs(S, cvrp)
            push!(list_S, S)
            push!(list_rhs, rhs)
        end
    end
    return list_S, list_rhs, all_S, all_rhs, all_z
end


function add_exact_rounded_capacity_cuts!(m, edge_x, g, optimizer, mcvrp)
   S, RHS, _, _, _ = exact_rounded_capacity_cuts(edge_x, optimizer, mcvrp)

    len_s = Int[]
    for s in S
        push!(len_s, length(s))
    end

    x = m[:x]

    @constraint(m, [i in 1:length(RHS)], sum(x[e] for e in δ(g, S[i])) >= 2 * RHS[i])

    return S, RHS
end

function find_exact_rounded_capacity_cuts!(m, edge_x, g, optimizer, mcvrp)
    @show S, RHS, all_s, all_rhs, all_z = exact_rounded_capacity_cuts(edge_x, optimizer, mcvrp)

    return S, RHS, all_s, all_rhs, all_z
end

function add_exact_rci!(m, edge_x, g, optimizer, mcvrp)
    S, RHS, all_s, all_rhs, all_z = exact_rounded_capacity_cuts(edge_x, optimizer, mcvrp)

    x = m[:x]
    @constraint(m, [i in 1:length(RHS)], sum(x[e] for e in δ(g, S[i])) >= 2 * RHS[i])

    return S, RHS, all_s, all_rhs, all_z
end


mutable struct CutSeparationInput
    demand :: Vector{Int}
    capacity :: Int
    edge_tail :: Vector{Int}
    edge_head :: Vector{Int}
    edge_x :: Vector{Float64}
    g :: Graph
    E :: Vector{Tuple{Int, Int}}
end

function matrix_to_nested_vector(matrix::Matrix{Vector{Int64}})
    return [vec(matrix[row, :]) for row in 1:size(matrix, 1)] # Convert each row to a vector
end

@kwdef mutable struct SearchOptions
    use_pi_greedy :: Bool = true
    use_graphchip_rci :: Bool = true
end

function learned_rounded_capacity_cuts(edge, x_bar, x_mat, cvrp::CVRP, k::Int, search_options::SearchOptions, return_fci::Bool)
    use_pi_greedy = search_options.use_pi_greedy
    use_graphchip_rci = search_options.use_graphchip_rci
    push!(pyimport("sys")."path", py_dir)
    policy = pyimport("pyjulia_main")
    list_s, list_rhs, fci_subsets, fci_vs, fci_bs = policy.get_rci_with_coarsening(edge, x_bar, cvrp.demand, cvrp.capacity, use_pi_greedy, use_graphchip_rci, return_fci)
    if fci_subsets isa Matrix{Vector{Int64}} # Python -> Julia : Type conversion issue
        fci_subsets = matrix_to_nested_vector(fci_subsets)
    end

    @assert length(list_s) == length(list_rhs)
    @assert length(fci_subsets) == length(fci_bs)

    return list_s, list_rhs, fci_subsets, fci_bs
end


function add_learned_rounded_capacity_cuts!(m, edge, x_bar, x_mat, g, mcvrp, k, search_options; return_fci=false)
    # RHS = k(S)
    S, RHS, fci_S, fci_bs = learned_rounded_capacity_cuts(edge, x_bar, x_mat, mcvrp, k, search_options, return_fci)
    ########################################################################
    # S: RCI subset
    # RHS: RCI RHS, i.e., 2 * k(S)
    # fci_S: FCI subsets (the last one of each sets is for the union of others)
    # fci_bs: # of bins 2 * (r(s_i) + r(S, Omega))
    ########################################################################

    len_s = Int[]
    for s in S
        push!(len_s, length(s))
        if length(δ(g, s)) == 0
            println(δ(g, s))
            println(s)
        end
    end

    violations = Float64[]

    x = m[:x]
    x_vals = value.(x)
    for i in 1:length(S)
        n = mcvrp.dimension
        if len_s[i] > n / 2
            S_bar = complement(S[i], n)
            if length(S_bar) == 0
                @constraint(m, sum(x[e] for e in inner(S[i])) <= length(S[i]) - RHS[i])
            else
                # x(S_bar:S_bar) + 0.5 x({0}:S_bar) - 0.5 x({0}:S) <= |S_bar| - k(S)
                @constraint(m, sum(x[e] for e in inner(S_bar)) + 0.5 * (sum(x[e] for e in between([1], S_bar)) - sum(x[e] for e in between([1], S[i]))) <= length(S_bar) - RHS[i])
            end
        else
            # x(S:S) <= |S| - k(S)
            @constraint(m, sum(x[e] for e in inner(S[i])) <= length(S[i]) - RHS[i])
        end

        push!(violations, 2 * rounded_capacity_rhs(S[i], mcvrp)  - sum(x_vals[e] for e in δ(g, S[i])))
    end

    if length(violations) > 0
        max_violation = maximum(violations)
    else
        max_violation = 0.0
    end

    ####################### Learned Framed Capacity Inequalities #######################
    fci_vs = Float64[]
    for i in 1:length(fci_S)
        fci_v = fci_bs[i] - (sum(x_vals[e] for e in δ(g, fci_S[i][end])) + sum(sum(x_vals[e] for e in δ(g, Si)) for Si in fci_S[i][1:end-1]))
        @constraint(m, 
            sum(
                sum(x[e] for e in δ(g, Si))
                    for Si in fci_S[i]
                )
                >= fci_bs[i] 
            )
        # @show fci_bs[i], fci_S[i] 
    end

    return S, RHS, mean(len_s), max_violation, fci_S, x_vals
end

function add_learned_framed_capacity_cuts!(m, edge, x_bar, x_mat, g, mcvrp, k, x_vals, search_options)
    _, _, fci_S, fci_bs = learned_rounded_capacity_cuts(edge, x_bar, x_mat, mcvrp, k, search_options, true)
    ########################################################################
    # S: RCI subset
    # RHS: RCI RHS, i.e., 2 * k(S)
    # fci_S: FCI subsets (the last one of each sets is for the union of others)
    # fci_bs: # of bins 2 * (r(s_i) + r(S, Omega))
    ########################################################################

    x = m[:x]
    fci_vs = Float64[]
    for i in 1:length(fci_S)
        fci_v = fci_bs[i] - (sum(x_vals[e] for e in δ(g, fci_S[i][end])) + sum(sum(x_vals[e] for e in δ(g, Si)) for Si in fci_S[i][1:end-1]))
        push!(fci_vs, fci_v)
        @constraint(m, 
            sum(
                sum(x[e] for e in δ(g, Si))
                    for Si in fci_S[i]
                )
                >= fci_bs[i] 
            )
    end

    if length(fci_vs) > 0
        max_fci_violation = maximum(fci_vs)
    else
        max_fci_violation = 0.0
    end
    return length(fci_S), max_fci_violation
end

function add_rounded_capacity_cuts!(m, cut_manager, g, mcvrp, datapack::CutSeparationInput; max_n_cuts = 10)
    ###############################################################
    # rounded capacity cut (RCC)
    # x(S:S) <= |S| - k(S)
    ###############################################################
    S, RHS = rounded_capacity_cuts!(
        cut_manager,
        datapack.demand,
        datapack.capacity,
        datapack.edge_tail,
        datapack.edge_head,
        datapack.edge_x,
        integrality_tolerance = 1e-3,
        max_n_cuts = max_n_cuts
    )

    len_s = Int[]
    for s in S
        push!(len_s, length(s))
    end

    violations = Float64[]

    x = m[:x]
    x_vals = value.(x)
    for i in 1:length(S)
        n = length(datapack.demand)
        if len_s[i] > n / 2
            S_bar = complement(S[i], n)
            if length(S_bar) == 0
                @constraint(m, sum(x[e] for e in inner(S[i])) <= RHS[i])
            else
                # x(S_bar:S_bar) + 0.5 x({0}:S_bar) - 0.5 x({0}:S) <= |S_bar| - k(S)
                @constraint(m, sum(x[e] for e in inner(S_bar)) + 0.5 * (sum(x[e] for e in between([1], S_bar)) - sum(x[e] for e in between([1], S[i]))) <= length(S_bar) - rounded_capacity_rhs(S[i], mcvrp))
            end
        else
            # x(S:S) <= |S| - k(S)
            @constraint(m, sum(x[e] for e in inner(S[i])) <= RHS[i])
        end

        push!(violations, 2 * rounded_capacity_rhs(S[i], mcvrp)  - sum(x_vals[e] for e in δ(g, S[i])))
    end

    if length(violations) > 0
        max_violation = maximum(violations)
    else
        max_violation = 0.0
    end

    return length(S), mean(len_s), max_violation, x_vals
end

function add_framed_capacity_inequalities!(m, cut_manager, datapack, x_vals; max_n_cuts = 10, max_n_tree_nodes = 10, threshold = 0.0)
    SS, FCI_RHS = framed_capacity_inequalities!(
        cut_manager, 
        datapack.demand, 
        datapack.capacity, 
        datapack.edge_tail, 
        datapack.edge_head, 
        datapack.edge_x,
        max_n_tree_nodes = max_n_tree_nodes,
        max_n_cuts = max_n_cuts
    )
    g = datapack.g
    x = m[:x]
    # x_vals = value.(x)

    fci_violations = Float64[]

    for j in 1:length(SS)
        subsets = SS[j] 

        S = reduce(vcat, subsets)
        @assert S == unique(S)

        fci_v = FCI_RHS[j] - (sum(x_vals[e] for e in δ(g, S)) + sum(sum(x_vals[e] for e in δ(g, Si)) for Si in subsets))

        @constraint(m, 
            sum(x[e] for e in δ(g, S)) + 
            sum(
                sum(x[e] for e in δ(g, Si))
                for Si in subsets
            )
            >= FCI_RHS[j] 
        )

        push!(fci_violations, fci_v)
    end

    if length(fci_violations) > 0
        max_fci_violation = maximum(fci_violations)
    else
        max_fci_violation = 0.0
    end
    
    return length(SS), max_fci_violation
end


@kwdef mutable struct CutOptions
    use_exact_rounded_capacity_cuts :: Bool = false
    use_learned_rounded_capacity_cuts :: Bool = false
    use_learned_framed_capacity_cuts :: Bool = false
    use_rounded_capacity_cuts :: Bool = true
    use_framed_capacity_cuts :: Bool = false
end

function solve_root_node_relaxation(cvrp::CVRPLIB.CVRP, k::Int, optimizer, cut_options::CutOptions, search_options::SearchOptions; max_n_cuts = 10, max_n_tree_nodes = 10, max_iter = -1, max_time=-1)

    _weights = Float64.(cvrp.weights[1:end-1, 1:end-1])
    @assert issymmetric(_weights)

    z_list = Float64[]
    list_time = Float64[]
    iter_time = Float64[]
    iter_cut = Int[]
    max_violations = Float64[]

    root_start = time()

    # If there is a non-diagonal zero term in _weights,
    # add a very small number, so that it can create an edge.
    for i in 1:size(_weights, 1)
        for j in 1:size(_weights, 2)
            if i != j && _weights[i, j] == 0
                _weights[i, j] = 1e-8
            end
        end
    end

    g = Graph(_weights)
    E = _edges(g)
    datapack = CutSeparationInput(
        cvrp.demand[1:cvrp.dimension],
        cvrp.capacity,
        Int[],
        Int[],
        Float64[],
        g,
        E
    )

    edge_cost(e::Tuple) = _weights[e...]

    m = Model(optimizer)

    set_optimizer_attribute(m, "CPX_PARAM_NUMERICALEMPHASIS", 1)

    @variable(m, x[e in E] >= 0)
    @objective(m, Min, sum(edge_cost(e) * x[e] for e in E))

    for e in E
        if in(e, δ(g, cvrp.depot))
            @constraint(m, x[e] <= 2)
        else
            @constraint(m, x[e] <= 1)
        end
    end

    @constraint(m, [i in cvrp.customers], sum(x[e] for e in δ(g, i)) == 2)
    @constraint(m, sum(x[e] for e in δ(g, cvrp.depot)) == 2 * k)

    total_n_cuts = 0
    total_iter = 0

    cut_manager = CutManager()
    prev_obj = 0.0
    count = 0

    total_fci_cuts = 0 # Check total FCI cuts added

    while true
        iter_start = time()
        optimize!(m)
        push!(z_list, objective_value(m))
        if objective_value(m) <= prev_obj + 1e-4
            count += 1
            if count >= 3 # max no. of iterations without improvements
                println("No more Improvement: ", count)
                break
            end
        else
            count = 0
        end
        prev_obj = objective_value(m)

        if (max_iter > 0 && total_iter >= max_iter)
            break
        end

        edge_tail = first.(E)
        edge_head = last.(E)
        edge_x = value.(x).data
        idx = findall(v -> v > 0.0, edge_x)
        datapack.edge_tail = edge_tail[idx]
        datapack.edge_head = edge_head[idx]
        datapack.edge_x = edge_x[idx]

        n_cuts = 0

        if (max_time > 0) && (time() - root_start > max_time)
            break
        end

        #################################################################
        # learned rounded capacity cuts
        #################################################################
        if n_cuts == 0 && cut_options.use_learned_rounded_capacity_cuts
            edge_x_mat = zeros(size(_weights))
            for i in 1:length(edge_tail)
                edge_x_mat[edge_tail[i], edge_head[i]] = edge_x[i]
                edge_x_mat[edge_head[i], edge_tail[i]] = edge_x[i]
            end
            start = time()
            s, rhs, len_s, max_violation, fci_s, x_vals = add_learned_rounded_capacity_cuts!(m, E[idx], edge_x[idx], edge_x_mat, g, cvrp, k, search_options, return_fci=cut_options.use_learned_framed_capacity_cuts)
            push!(list_time, time() - start)
            n_new_cuts = length(s)
            new_fci_cuts = length(fci_s)
            
            total_fci_cuts += new_fci_cuts
            n_new_cuts += new_fci_cuts
            
            n_cuts += n_new_cuts

            push!(iter_cut, n_new_cuts)
            push!(iter_cut, new_fci_cuts)
            push!(max_violations, max_violation)
            
        end

        #################################################################
        # rounded capacity cuts
        #################################################################
        if n_cuts == 0 && cut_options.use_rounded_capacity_cuts
            start = time()
            n_new_cuts, len_s_mean, max_violation, x_vals = add_rounded_capacity_cuts!(m, cut_manager, g, cvrp, datapack; max_n_cuts=max_n_cuts)
#             println("RCC Cuts = $n_new_cuts")
            push!(list_time, time() - start)
            push!(iter_cut, n_new_cuts)
            push!(max_violations, max_violation)
            n_cuts += n_new_cuts
        end

        #################################################################
        # EXACT rounded capacity cuts
        #################################################################
        if n_cuts == 0 && cut_options.use_exact_rounded_capacity_cuts
            edge_x_mat = zeros(size(_weights))
            for i in 1:length(edge_tail)
                edge_x_mat[edge_tail[i], edge_head[i]] = edge_x[i]
                edge_x_mat[edge_head[i], edge_tail[i]] = edge_x[i]
            end
            start = time()
            s, rhs = add_exact_rounded_capacity_cuts!(m, edge_x_mat, g, optimizer, cvrp)
            push!(list_time, time() - start)
            n_new_cuts = length(s)
            push!(iter_cut, n_new_cuts)
            n_cuts += n_new_cuts

            @show n_new_cuts
        end

        #################################################################
        # learned framed capacity inequalities ONLY
        #################################################################
        if cut_options.use_learned_framed_capacity_cuts
            if !cut_options.use_learned_rounded_capacity_cuts
                if max_violation <= 1.0
                    edge_x_mat = zeros(size(_weights))
                    for i in 1:length(edge_tail)
                        edge_x_mat[edge_tail[i], edge_head[i]] = edge_x[i]
                        edge_x_mat[edge_head[i], edge_tail[i]] = edge_x[i]
                    end
                    new_fci_cuts, max_fci_violation_n = add_learned_framed_capacity_cuts!(m, E[idx], edge_x[idx], edge_x_mat, g, cvrp, k, x_vals, search_options)
                else
                    new_fci_cuts, max_fci_violation_n = 0, 0.0
                end
                total_fci_cuts += new_fci_cuts
                n_cuts += new_fci_cuts
            end
        end
            
            
        if cut_options.use_framed_capacity_cuts
            if max_violation <= 1.0
                new_fci_cuts, max_fci_violation = add_framed_capacity_inequalities!(m, cut_manager, datapack, x_vals; max_n_cuts=max_n_cuts, max_n_tree_nodes=max_n_tree_nodes, threshold=max_violation) # max_violation, 0.0
            else
                new_fci_cuts, max_fci_violation = 0, 0.0
            end

            total_fci_cuts += new_fci_cuts
            n_cuts += new_fci_cuts

            push!(iter_cut, new_fci_cuts)

        end

        total_n_cuts += n_cuts
        total_iter += 1

        push!(iter_time, time() - iter_start)

        ( n_cuts == 0 ) && break

    end

    @show total_fci_cuts
    @show total_n_cuts
    
    optimize!(m)

    return objective_value(m), iter_time, list_time, iter_cut, total_fci_cuts, z_list, max_violations
end
