using(Pickle)
using TOML
include("cvrp_cutting.jl")

instances = [  
("random-24-X-n50", 6686.0, 3)
# ("random-4-X-n50", 7684.0, 4)
# ("random-28-X-n50", 11557.0, 7)
# ("random-0-X-n50", 12735.0, 8)
# ("random-16-X-n50", 10190.0, 5)
# ("random-20-X-n50", 10126.0, 6)
# ("random-32-X-n50", 7483.0, 4)
# ("random-12-X-n50", 8317.0, 4)
# ("random-36-X-n50", 14188.0, 8)
# ("random-8-X-n50", 6733.0, 3) 
# ("random-17-X-n75", 34431.0, 20)
# ("random-21-X-n75", 9726.0, 7)
# ("random-25-X-n75", 10931.0, 6)
# ("random-5-X-n75", 14930.0, 11)
# ("random-9-X-n75", 9361.0, 5)
# ("random-29-X-n75", 11893.0, 7)
# ("random-1-X-n75", 11225.0, 7)
# ("random-13-X-n75", 9652.0, 6)
# ("random-37-X-n75", 9479.0, 4)
# ("random-33-X-n75", 16151.0, 13)
# ("random-2-X-n100", 17242.0, 11)
# ("random-30-X-n100", 12151.0, 6)
# ("random-14-X-n100", 16934.0, 12)
# ("random-6-X-n100", 24832.0, 14)
# ("random-34-X-n100", 10752.0, 8)
# ("random-26-X-n100", 15654.0, 11)
# ("random-18-X-n100", 11171.0, 7)
# ("random-22-X-n100", 27022.0, 20)
# ("random-38-X-n100", 13766.0, 11)
# ("random-10-X-n100", 17727.0, 12)
# ("random-19-X-n200", 16102.0, 12)
# ("random-27-X-n200", 21609.0, 13)
# ("random-23-X-n200", 26556.0, 23) 
# ("random-7-X-n200", 17182.0, 13)
# ("random-11-X-n200", 29495.0, 20)
# ("random-35-X-n200", 24854.0, 15)
# ("random-15-X-n200", 23121.0, 18)
# ("random-3-X-n200", 19644.0, 13)
# ("random-31-X-n200", 31684.0, 21)
# ("random-39-X-n200", 18271.0, 11)
# ("random-7-X-n300", 27269.0, 19)
# ("random-1-X-n300", 30181.0, 25) 
# ("random-5-X-n300", 53305.0, 40)
# ("random-0-X-n300", 55614.0, 46) 
# ("random-8-X-n300", 20636.0, 15)
# ("random-6-X-n300", 42079.0, 40)
# ("random-4-X-n300", 24324.0, 20)
# ("random-2-X-n300", 47765.0, 32)
# ("random-3-X-n300", 23414.0, 20)
# ("random-9-X-n300", 21740.0, 17) 
# ("random-14-X-n400", 61276.0, 46) 
# ("random-6-X-n400", 52025.0, 54) 
# ("random-10-X-n400", 59487.0, 47) 
# ("random-2-X-n400", 61447.0, 42)
# ("random-12-X-n400", 37545.0, 30)
# ("random-0-X-n400", 77967.0, 61) 
# ("random-16-X-n400", 57273.0, 38)
# ("random-18-X-n400", 34643.0, 27)
# ("random-4-X-n400", 29917.0, 27)
# ("random-8-X-n400", 30953.0, 20)
# ("random-17-X-n500", 134831.0, 128) 
# ("random-15-X-n500", 48692.0, 44)
# ("random-19-X-n500", 34861.0, 28)
# ("random-1-X-n500", 40287.0, 41)
# ("random-9-X-n500", 43607.0, 29)
# ("random-13-X-n500", 46414.0, 36) 
# ("random-5-X-n500", 85332.0, 67) 
# ("random-11-X-n500", 66720.0, 48)
# ("random-3-X-n500", 56591.0, 33)
# ("random-7-X-n500", 48159.0, 32)
# ("random-6-X-n750", 137429.0, 99) 
# ("random-12-X-n750", 81988.0, 56)
# ("random-4-X-n750", 55938.0, 49) 
# ("random-14-X-n750", 98971.0, 87) 
# ("random-8-X-n750", 54145.0, 37)
# ("random-2-X-n750", 76471.0, 79) 
# ("random-16-X-n750", 77111.0, 72) 
# ("random-0-X-n750", 137798.0, 115) 
# ("random-18-X-n750", 85348.0, 50)
# ("random-10-X-n750", 119393.0, 88)
# ("random-21-X-n1000", 109492.0, 81)  
# ("random-19-X-n1000", 73749.0, 56)
# ("random-15-X-n1000", 112383.0, 88)
# ("random-11-X-n1000", 102522.0, 96)
# ("random-13-X-n1000", 92888.0, 72)  
# ("random-5-X-n1000", 130385.0, 133) 
# ("random-1-X-n1000", 86706.0, 81) 
# ("random-7-X-n1000", 81052.0, 64)
# ("random-3-X-n1000", 76005.0, 65)
# ("random-9-X-n1000", 81448.0, 57)
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
    k = ins[3]

    path = "../../data/instances/" 
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

    size = parse(Int, split(split(name, '-')[end], 'n')[end]) 
    println("Solving instance: ", name, ", size: ", size, ", k: ", k, ", opt: ", opt)
    cvrpsep_max_n_cuts = max(100, k)

    # Parameters for cutting plane method: Iterations and time limit
    max_iter = -1
    max_time = -1 # 60*60 (1hr)

    start = time()
    @time lowerbound, iter_time, list_time, iter_cut, total_fci_cuts, z_list, violations = solve_root_node_relaxation(cvrp, k, my_optimizer, cut_options, search_options; max_n_cuts=cvrpsep_max_n_cuts, max_iter=max_iter, max_time=max_time)
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
    if !isdir("../../data/results/")
        mkpath("../../data/results/")
    end
    file = string("../../data/results/exe_", name, method, ".pkl")
    store(file, [lowerbound, root_time, iter_time, list_time, iter_cut, total_fci_cuts, z_list, violations])
end
