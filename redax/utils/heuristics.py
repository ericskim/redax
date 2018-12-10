def order_heuristic(mgr):

    def _name(i):
        return i.split('_')[0]

    def _idx(i):
        return int(i.split('_')[1])

    vars = list(mgr.vars)

    max_granularity = max([_idx(i) for i in vars])
    order_seed = []
    for i in range(max_granularity + 1):
        order_seed.extend([v for v in vars if _idx(v) == i])

    return {var: idx for idx, var in enumerate(order_seed)}