

def flatten(l):
    """Flatten a singly nested list.
    
    Examples
    -------
    >>> flatten([[0,1,2], [4,5]])
    [0, 1, 2, 4, 5]
    """
    return [item for sublist in l for item in sublist]

def int2bv(index: int, nbits: int):
    """
    A really high nbits just right pads the bitvector with "False"
    """

    return tuple(True if ((index >> i) % 2 == 1) else False for i in range(nbits - 1, -1, -1))


def increment_index(index: int, increment: int, nbits=None, graycode=False):
    if graycode:
        assert nbits is not None
        index = graytobin(index)
        index = (index + increment) % 2**nbits
        return bintogray(index)
    else:
        return index + increment


def bv_interval(lb, ub, graycode = False):
    assert len(lb) == len(ub)
    nbits = len(lb)
    if not graycode and lb > ub:
        return []

    lb = bv2int(lb)
    ub = bv2int(ub)

    return map(lambda x: int2bv(x,nbits),
                    index_interval(lb, ub, nbits, graycode))


def index_interval(lb: int, ub: int, nbits=None, graycode=False):
    """Construct an integer interval that includes both ends lb and ub."""
    if graycode:
        assert nbits is not None
    else:
        assert lb <= ub

    window = []
    i = lb
    while True:
        window.append(i)
        if i == ub:
            break
        i = increment_index(i, 1, nbits, graycode)
    return window


def num_with_name(name: str, x):
    return len([i for i in x if name in i])


def bv2pred(mgr, name: str, bv):
    """Convert a variable precision bitvector into a predicate."""

    for i in range(len(bv)):
        mgr.declare(name + "_" + str(i))
    b = mgr.true
    for i in range(len(bv)):
        if bv[i]:
            b &=  mgr.var(name + "_" + str(i))
        else:
            b &= ~mgr.var(name + "_" + str(i))
    return b


def increment_bv(bv, increment: int, graycode=False, saturate=False):
    """
    Increment a bitvector's value +1 or -1.
    """
    assert increment == 1 or increment == -1
    nbits = len(bv)
    if graycode:
        index = graytobin(bv2int(bv))
        index = (index+increment) % 2**nbits
        return int2bv( bintogray(index), nbits)
    else:
        if bv == tuple(True for i in range(nbits)) and increment > 0:
            if saturate:
                return bv
            raise ValueError("Bitvector overflow for nonperiodic domain.")
        if bv == tuple(False for i in range(nbits)) and increment < 0:
            if saturate:
                return bv
            raise ValueError("Bitvector overflow for nonperiodic domain.")
        return int2bv(bv2int(bv) + increment, nbits)


def bv2int(bv):
    """
    Converts bitvector (list or tuple) with the standard binary encoding into an integer.
    """
    nbits = len(bv)
    index = 0
    for i in range(nbits):
        if bv[i]:
            index += 2**(nbits - i - 1)
    return index

def bintogray(x: int):
    """
    Convert a binary encoded positive integer into gray code.
    """
    assert x >= 0
    return x ^ (x >> 1)


def graytobin(x: int):
    """
    Convert a gray code encoded positive integer into the standard binary encoding.
    """
    assert x >= 0
    mask = x >> 1
    while(mask != 0):
        x = x ^ mask
        mask = mask >> 1
    return x


def bvwindow(left: int, right: int, nbits: int):
    """
    Convert a window [left, right] inclusive into a collection of variable precision bitvectors
    """
    assert left >= 0
    assert right >= 0
    assert right <= 2**nbits - 1

    bvs = []
    # Empty window
    if right < left:
        return bvs 

    while(True):
        
        if nbits == 0:
            return [(True,), (False,)]

        if left == right:
            bvs += [int2bv(left, nbits)]
            break

        # Catch left edge
        if left % 2 == 1:
            bvs += [int2bv(left, nbits)]
            left += 1

        # Catch right edge
        if right % 2 == 0:
            bvs += [int2bv(right, nbits)]
            right -= 1

        if left > right:
            break

        # Reduce precision
        nbits = nbits-1
        left = left // 2
        right = right // 2

    return bvs


def bvwindowgray(left: int, right: int, nbits: int):
    bvs = []

    assert right <= 2**nbits - 1
    assert left <= 2**nbits - 1

    # Split window into [0,right] and [left, 2**nbits-1]
    if left > right:
        return bvwindowgray(left, 2**nbits-1, nbits) + bvwindowgray(0, right, nbits)

    while(True):
        if nbits == 0:
            return [(True,), (False,)]

        if left == right:
            bvs += [int2bv(bintogray(left), nbits)]
            break

        # Catch left edge
        if left % 4 in (1,3):
            bvs += [int2bv(bintogray(left), nbits)]
            left += 1

        # Catch right edge
        if right % 4 in (0,2):
            bvs += [int2bv(bintogray(right), nbits)]
            right -= 1

        if left > right:
            break

        nbits = nbits - 1
        left = left // 2
        right = right // 2

    return bvs
