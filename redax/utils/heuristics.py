from redax.utils.bv import bv_var_name, bv_var_idx


def order_heuristic(mgr):

    vars = list(mgr.vars)

    max_granularity = max([int(bv_var_idx(i)) for i in vars])
    order_seed = []
    for i in range(max_granularity + 1):
        order_seed.extend([v for v in vars if int(bv_var_idx(v)) == i])

    return {var: idx for idx, var in enumerate(order_seed)}


def order_heuristic_vars(mgr, var_priority = None):
    """
    Most signifiant bits are ordered first in BDD.

    var_priority is a list of variable names. Resolves ties if two bits, are of the same priority, it resolves based.
    e.g. var_priority = ['x','y','z'] would impose an order ['x_0', 'y_0', 'z_0']
    """

    # def _name(i):
    #     return i.split('_')[0]

    # def _idx(i):
    #     return int(i.split('_')[1])

    vars = list(mgr.vars)

    max_granularity = max([int(bv_var_idx(i)) for i in vars])
    order_seed = []
    for i in range(max_granularity + 1):
        level_bits = [v for v in vars if int(bv_var_idx(v)) == i]
        if var_priority is not None:
            level_bits.sort(key = lambda x: {k:v for v, k in enumerate(var_priority)}[ bv_var_name(x)])
        order_seed.extend(level_bits)

    return {var: idx for idx, var in enumerate(order_seed)}