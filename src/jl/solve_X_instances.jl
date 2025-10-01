using(Pickle)
using TOML
include("cvrp_cutting.jl")

instances = [
# ("X-n101-k25", 27591)
# ("X-n106-k14", 26362)
# ("X-n110-k13", 14971)
# ("X-n115-k10", 12747)
# ("X-n120-k6", 13332)
# ("X-n125-k30", 55539)
# ("X-n129-k18", 28940)
# ("X-n134-k13", 10916)
# ("X-n139-k10", 13590)
# ("X-n143-k7", 15700)
# ("X-n148-k46", 43448)
("X-n153-k22", 21220)
# ("X-n157-k13", 16876)
# ("X-n162-k11", 14138)
# ("X-n167-k10", 20557)
# ("X-n172-k51", 45607)
# ("X-n176-k26", 47812)
# ("X-n181-k23", 25569)
# ("X-n186-k15", 24145)
# ("X-n190-k8", 16980)
# ("X-n195-k51", 44225)
# ("X-n200-k36", 58578)
# ("X-n204-k19", 19565)
# ("X-n209-k16", 30656)
# ("X-n214-k11", 10856)
# ("X-n219-k73", 117595)
# ("X-n223-k34", 40437)
# ("X-n228-k23", 25742)
# ("X-n233-k16", 19230)
# ("X-n237-k14", 27042)
# ("X-n242-k48", 82751)
# ("X-n247-k50", 37274)
# ("X-n251-k28", 38684)
# ("X-n256-k16", 18839)
# ("X-n261-k13", 26558)
# ("X-n266-k58", 75478)
# ("X-n270-k35", 35291)
# ("X-n275-k28", 21245)
# ("X-n280-k17", 33503)
# ("X-n284-k15", 20215)
# ("X-n289-k60", 95151)
# ("X-n294-k50", 47161)
# ("X-n298-k31", 34231)
# ("X-n401-k29", 66154)
# ("X-n411-k19", 19712)
# ("X-n420-k130", 107798)
# ("X-n429-k61", 65449)
# ("X-n439-k37", 36391)
# ("X-n449-k29", 55233)
# ("X-n459-k26", 24139)
# ("X-n469-k138", 221824)
# ("X-n480-k70", 89449)
# ("X-n491-k59", 66483)
]


# Load options from config file
config = TOML.parsefile("../../config/options.toml")
to_kwargs(d::AbstractDict) = (; (Symbol(k) => v for (k,v) in d)...)

cut_options    = CutOptions(; to_kwargs(config["cut_options"])...)
search_options = SearchOptions(; to_kwargs(config["search_options"])...)
@show cut_options
@show search_options

for ins in instances
    name = ins[1]
    opt = ins[2]
    k = parse(Int, split(name, 'k')[end]) 

    path = "../../data/X/" 
    cvrp = readCVRP(string(path, name, ".vrp"), add_dummy=true)

    cplex = optimizer_with_attributes(
        CPLEX.Optimizer,
        "CPX_PARAM_SCRIND" => 0, # output level
        # "CPX_PARAM_LPMETHOD" => 2, # 1:Primal Simplex, 2: Dual Simplex, 3: Network Simplex, 4: Barrier
        # "CPXPARAM_Barrier_ConvergeTol" => 1e-12,
        # "CPXPARAM_Simplex_Tolerances_Optimality" => 1e-9
        # "CPXPARAM_Advance" => 1,
    )
    my_optimizer = cplex

    size = parse(Int, split(split(name, '-')[2], 'n')[end])
    println("Solving instance: ", name, ", size: ", size, ", k: ", k, ", opt: ", opt)
    cvrpsep_max_n_cuts = max(100, k)

    # Parameters for cutting plane method: Iterations and time limit
    max_iter = -1
    max_time = 3600 # 60*60 (1hr)

    start = time()
    @time lowerbound, iter_time, list_time, iter_cut, total_fci_cuts, z_list, violations = solve_root_node_relaxation(cvrp, k, my_optimizer, cut_options, search_options; max_n_cuts=cvrpsep_max_n_cuts, max_iter = max_iter, max_time=max_time)
    root_time = time() - start
    @show name, opt, lowerbound

    if cut_options.use_learned_rounded_capacity_cuts
        method = "_neuralsep_rci" 
    elseif cut_options.use_exact_rounded_capacity_cuts
        method = "_exact"
    elseif cut_options.use_rounded_capacity_cuts
        method = "_cvrpsep_rci"
    end

    if cut_options.use_framed_capacity_cuts
        method = string(method, "_cvrpsep_fci")
    elseif cut_options.use_learned_framed_capacity_cuts
        method = string(method, "_neuralsep_fci")
    end

    # Save results
    file = string("../../data/results/exe_", name, method, ".pkl")
    store(file, [lowerbound, root_time, iter_time, list_time, iter_cut, total_fci_cuts, z_list, violations])
end
