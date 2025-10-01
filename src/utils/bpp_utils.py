import cplex
from cplex.exceptions import CplexError
import numpy as np

from math import ceil
import sys

def solve_bpp(w, c):
    """Obtain the lower bound of the bin packing problem using CPLEX. (Limits: 2 minutes)"""
    # print("CPLEX")
    n = len(w)
    ub = n

    try:
        # Initialize the CPLEX model
        model = cplex.Cplex()
        model.set_log_stream(None)
        model.set_error_stream(None)
        model.set_warning_stream(None)
        model.set_results_stream(None) # sys.stdout


        # Set the objective to minimize
        model.objective.set_sense(model.objective.sense.minimize)

        # Add y variables (binary)
        y = ["y" + str(j) for j in range(ub)]
        model.variables.add(names=y, types=[model.variables.type.binary] * ub, obj=[1.0] * ub)

        # Add x variables (binary)
        x = [["x" + str(i) + "_" + str(j) for j in range(ub)] for i in range(n)]
        for i in range(n):
            model.variables.add(names=x[i], types=[model.variables.type.binary] * ub)

        # Add constraints: each item in exactly one bin
        for i in range(n):
            indices = [x[i][j] for j in range(ub)]
            values = [1.0] * ub
            model.linear_constraints.add(lin_expr=[[indices, values]], senses=["E"], rhs=[1.0])

        # Add constraints: bin capacity
        for j in range(ub):
            indices = [x[i][j] for i in range(n)] + [y[j]]
            values = w + [-c]
            model.linear_constraints.add(lin_expr=[[indices, values]], senses=["L"], rhs=[0.0])

        # Solve the model
        model.parameters.timelimit.set(120) # 2 minutes
        model.solve()

        return ceil(model.solution.MIP.get_best_objective())  # model.solution.get_objective_value() : UpperBound

    except CplexError as exc:
        print(exc)
        return None


import math
def bp_mtl2(items: list[int], capacity: int) -> int:
    """Calculates the L2 lower bound from Martello and Toth (1990)."""

    items_sorted = items.copy()
    items_sorted.sort(reverse=True)
    
    n = len(items_sorted)
    if n == 0: return 0
    if n > 1 and (items_sorted[n-2] + items_sorted[n-1]) > capacity: return n
    j_star = -1
    for i in range(n):
        if items_sorted[i] * 2 <= capacity:
            j_star = i
            break
    if j_star == -1: return n
    if j_star == 0: return math.ceil(sum(items_sorted) / capacity)
    cj12 = j_star
    j_prime = j_star
    limit = capacity - items_sorted[j_star]
    for i in range(j_star):
        if items_sorted[i] <= limit:
            j_prime = i
            break
    cj2 = j_star - j_prime
    sj2 = sum(items_sorted[i] for i in range(j_prime, j_star))
    jd_prime = j_star
    sj3 = items_sorted[jd_prime]
    while jd_prime + 1 < n and items_sorted[jd_prime + 1] == items_sorted[jd_prime]:
        jd_prime += 1
        sj3 += items_sorted[jd_prime]
    l2 = cj12
    while True:
        add_size_sum = sj3 + sj2 - (cj2 * capacity)
        add_bins = 0
        if add_size_sum > 0:
            add_bins = math.ceil(add_size_sum / capacity)
        if (cj12 + add_bins) > l2:
            l2 = cj12 + add_bins
        jd_prime += 1
        if jd_prime >= n: break
        sj3 += items_sorted[jd_prime]
        while jd_prime + 1 < n and items_sorted[jd_prime + 1] == items_sorted[jd_prime]:
            jd_prime += 1
            sj3 += items_sorted[jd_prime]
        if j_prime > 0:
            limit = capacity - items_sorted[jd_prime]
            while j_prime > 0 and items_sorted[j_prime - 1] <= limit:
                j_prime -= 1
                cj2 += 1
                sj2 += items_sorted[j_prime]
    return l2


if __name__ == '__main__':
    c = 45
    w = [1, 2, 3, 4, 5, 1, 3, 3, 4, 2, 3, 4, 2, 3, 3, 7, 8, 19, 10, 11, 30, 43, 12, 38, 30, 25, 10, 5, 6, 6, 6, 17,
         1, 2, 3, 4, 5, 1, 3, 3, 4, 2, 3, 4, 2, 3, 3, 7, 8, 19, 10, 11, 30, 43, 12, 38, 30, 25, 10, 5, 6, 6, 6, 17,
         1, 2, 3, 4, 5, 1, 3, 3, 4, 2, 3, 4, 2, 3, 3, 7, 8, 19, 10, 11, 30, 43, 12, 38, 30, 25, 10, 5, 6, 6, 6, 17,
         1, 2, 3, 4, 5, 1, 3, 3, 4, 2, 3, 4, 2, 3, 3, 7, 8, 19, 10, 11, 30, 43, 12, 38, 30, 25, 10, 5, 6, 6, 6, 11,
         1, 2, 3, 4, 5, 1, 3, 3, 4, 2, 3, 4, 2, 3, 3, 7, 8, 19, 10, 11, 30, 43, 12, 38, 30, 25, 10, 5, 6, 6, 6, 1,
         1, 2, 3, 4, 5, 1, 3, 3, 4, 2, 3, 4, 2, 3, 3, 7, 8, 19, 10, 11, 30, 43, 12, 38, 30, 25, 10, 5, 6, 6, 6, 1,
         1, 2, 3, 4, 5, 1, 3, 3, 4, 2, 3, 4, 2, 3, 3, 7, 8, 19, 10, 11, 30, 43, 12, 38, 30, 25, 10, 5, 6, 6, 6, 17] # 32 items
    print(bp_mtl2(w, c))
    c2 = 979
    w2 = [226, 26, 60, 27, 93, 234, 90, 32, 63, 61, 192, 179, 965, 102]
    print(bp_mtl2(w2, c2))


