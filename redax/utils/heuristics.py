from redax.utils.bv import bv_var_name, bv_var_idx


def order_heuristic(mgr):

    vars = list(mgr.vars)

    max_granularity = max([int(bv_var_idx(i)) for i in vars])
    order_seed = []
    for i in range(max_granularity + 1):
        order_seed.extend([v for v in vars if int(bv_var_idx(v)) == i])

    return {var: idx for idx, var in enumerate(order_seed)}