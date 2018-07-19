"""
Symbolic Intervals

#TODO: Think about combining the fixed and dynamic precision grids 
# to have bits that represent fixed and dynamic aspects to them. 

# Discrete vs continuous
# fixed vs dynamic
# linear vs periodic

# discrete => fixed, linear
# continuous => either linear or periodic, either fixed or dynamic 

Nomenclature:
    bv = Bit vector 
    box = 
    point = element of set 
"""

import math 
from abc import abstractmethod

from bidict import bidict

import dd

def _bintogray(x:int):
    assert x >= 0 
    return x ^ (x >> 1)

def _graytobin(x:int):
    assert x >= 0
    mask = x >> 1
    while(mask != 0):
        x = x ^ mask
        mask = mask >> 1
    return x

def _int2bv(index:int, nbits:int):
    """
    A really high nbits just right pads the bitvector with "False"
    """
    bv = [False] * nbits
    for i in range(nbits):
        if ((index >> i) % 2 == 1):
            bv[nbits-1-i] = True
        # else:
        #     bv[]
    return bv

def _bv2int(bv):
    nbits = len(bv)
    index = 0
    for i in range(nbits):
        if bv[i]:
            index += 2**(nbits - i - 1)

    return index 

def increment_bv(bv, increment, graycode = False, saturate = False):
    """
    increment a bitvector's value +1 or -1. 
    """
    assert increment == 1 or increment == -1
    nbits = len(bv)
    if graycode:
        index = _graytobin(_bv2int(bv))
        index = (index+increment) % 2**nbits
        return _int2bv( _bintogray(index), nbits)
    else:
        if bv == [True] * nbits and increment > 0:
            if saturate:
                return bv
            raise ValueError("Bitvector overflow for nonperiodic domain.")
        if bv == [False] * nbits and increment < 0:
            if saturate:
                return bv
            raise ValueError("Bitvector overflow for nonperiodic domain.")
        return _int2bv(_bv2int(bv) + increment, nbits)

def increment_index(index, increment, nbits = None, graycode = False):
    if graycode:
        assert nbits is not None 
        index = _graytobin(index)
        index = (index + increment) % 2**nbits
        return _bintogray(index)
    else:
        return index + increment

def bv_interval(lb, ub, graycode = False):
    assert len(lb) == len(ub)
    nbits = len(lb)
    if not graycode and lb > ub:
        return []

    lb = _bv2int(lb)
    ub = _bv2int(ub)

    return map(lambda x: _int2bv(x,nbits), 
                    index_interval(lb, ub, nbits, graycode))

def index_interval(lb, ub, nbits = None , graycode = False):
    """
    Constructs an integer interval that includes both ends lb and ub
    """
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

def num_with_name(name, x):
    return len([i for i in x if name in i])

# SymbolicInstance?
class SymbolicInterval(object):
    """
    Class to handle translation from symbolic BDD encoding to the underlying space

    Instance of an interval
    """

    def __init__(self, name, mgr, num_bits):
        assert isinstance(name, str)
        self.mgr = mgr
        self.name = name
        self.num_bits = num_bits
        
        newvars = [self.name + "_" + str(i) for i in range(num_bits)]
        for v in newvars:
            self.mgr.declare(v)

    @property
    def bits(self):
        newvars = [self.name + "_" + str(i) for i in range(self.num_bits)]
        return bidict({v: self.mgr.var(v) for v in newvars})

    @property
    def bitorder(self):
        return [self.name + "_" + str(i) for i in range(self.num_bits)]

    def ith_sig_var(self, i:int):
        return self.bitorder[i]

    def ith_sig_var_cube(self, i: int):
        return self.bits[self.bitorder[i]]

    def bv2bdd(self, bv):
        """
        If bv's size is smaller than the bits allocated, then it takes the most significant ones
        """
        if len(bv) > len(self.bitorder):
            raise ValueError("Bitvector is longer than number of BDD variables assigned to interval")
        b = self.mgr.true
        for i in range(len(bv)):
            if bv[i]:
                b = b &  self.mgr.var(self.bitorder[i])
            else:
                b = b & ~self.mgr.var(self.bitorder[i])
        return b

    def pt2bdd(self, point):
        return self.bv2bdd(self.pt2bv(point))

    @abstractmethod
    def pt2bv(self, point):
        raise NotImplementedError 

    @abstractmethod
    def rename(self, newname):
        """
        Renames and returns a dictionary from old to new variable names
        """
        # oldname = self.name 
        # self.name = newname 
        # newvars = [self.name + "_" + str(i) for i in range(self.num_bits)]
        # for v in newvars:
        #     self.mgr.declare(v)
        raise NotImplementedError

    # TODO: Implement generic iterator for multiple nbitss? 
    def __iter__(self):
        pass

class DiscreteInterval(SymbolicInterval):
    """
    Discrete Interval 
    """
    def __init__(self, name, mgr, num_vals):
        self.num_vals = num_vals
        num_bits = math.ceil(math.log2(num_vals))
        SymbolicInterval.__init__(self, name, mgr, num_bits)

    def pt2bv(self, point):
        assert point < self.num_vals
        return _int2bv(point, self.num_bits)

    def __repr__(self):
        s = "Discrete Interval, "
        s += "Bounds: [0,...,{0}], ".format(str(self.num_vals-1))
        s += "Variables: " + " ".join([str(k) for k in self.bits])
        return s

class ContinuousInterval(SymbolicInterval):
    """
    Continuous Interval
    """
    def __init__(self, name, mgr, lb, ub, num_bits, periodic = False):
        assert ub - lb >= 0.0, "Upper bound is smaller than lower bound"
        SymbolicInterval.__init__(self, name, mgr, num_bits)
        self.periodic = periodic
        self.lb = float(lb)
        self.ub = float(ub)

    @property
    def bounds(self):
        return (self.lb, self.ub)

    def _wrap(self, point):
        """
        Helper function for periodic intervals
        """

        if point == self.ub:
            return point
        
        width = self.ub - self.lb
        point =  (point - self.lb) % width
        return point + self.lb

    def bitscale(self, left, right):
        """
        Get an approximation of how many bits are needed to cover the interval given by the points
        """
        if self.periodic:
            raise NotImplementedError
        left_idx = int((left - self.lb) / (self.ub - self.lb) * 2**self.num_bits)
        right_idx = int((left - self.lb) / (self.ub - self.lb) * 2**self.num_bits)

        # Interval is small and need many bits FIXME: Nothing's being changed ??? 
        if right_idx == left_idx:
            self.num_bits

        # Interval is large and only need fewer bits 
        return self.num_bits - math.ceil(math.log2(right_idx - left_idx))

class DynamicInterval(ContinuousInterval):
    """
    Unlike Fixed, all bits assignments are considered valid and correspond to a quad/oct-ant of the space


    """
    def __init__(self, name, mgr, lb, ub, num_bits:int = 0, periodic = False):
        ContinuousInterval.__init__(self, name, mgr, lb, ub, num_bits, periodic)

    def renamed(self, newname):
        return DynamicInterval(newname, self.mgr, self.lb, self.ub, self.num_bits, self.periodic)

    def withbits(self, nbits):
        return DynamicInterval(self.name, self.mgr, self.lb, self.ub, nbits, self.periodic)

    def __repr__(self):
        s = "Dynamic Interval, "
        if self.periodic:
            s += "Periodic, "
        s += "Bounds: {0}, ".format(str(self.bounds))
        s += "# Bits: " + str(self.num_bits) + ", "
        s += "Variables: " + " ".join([str(k) for k in self.bits])

        return s

    def get_sub_interval(self, left, right, nbits = None):
        raise NotImplementedError

    def pt2bv(self, point, tol = 0.0, nbits = None):
        """
        Continuous point to a bitvector 
        """
        if nbits is None:
            nbits = self.num_bits
        if self.periodic:
            point = self._wrap(point)

        assert point <= self.ub + tol, "Point {0} exceeds upper bound {1}".format(point, self.ub + tol)
        assert point >= self.lb - tol, "Point {0} exceeds lower bound {1}".format(point, self.lb - tol)

        index = int((point - self.lb) / (self.ub - self.lb) * 2**nbits)

        # Catch numerical errors when point == self.ub 
        if index >= 2**nbits:
            index = 2**nbits - 1

        if self.periodic:
            index = _bintogray(index)

        # pdb.set_trace()
        return _int2bv(index,nbits)#[0:nbits]

    def bv2box(self, bv):
        nbits = len(bv)
        
        index = _bv2int(bv)
        
        if self.periodic:
            index = _graytobin(index)

        eps = (self.ub - self.lb) / (2**nbits)
        left  = self.lb + index * eps
        right = self.lb + (index+1) * eps
        
        return (left, right)

    def pt2box(self, point, nbits = None):
        if nbits is None:
            nbits = self.num_bits

        return self.bv2box(self.pt2bv(point, nbits = nbits))

    def box2bvs(self, box, innerapprox = False, tol = .0000001, nbits = None):
        raise NotImplementedError

    def box2bdd(self, box, innerapprox = False, tol = .0000001, nbits = None):
        """
        Overapproximation of a concrete box with its BDD

        @param innerapprox
        @param tol - tolerance for numerical errors as a fraction of the grid size. 
                     Must be in continuous interval [0,1]
        """

        left, right = box

        if nbits is None:
            nbits = self.num_bits
        if nbits > self.num_bits:
            raise ValueError("nbits exceeds currently allocated BDD variables")

        assert tol >= 0 and tol <= 1, "Tolerance is not 0 <= tol <= 1"
        eps = (self.ub - self.lb) / (2**nbits)
        abs_tol = eps * tol

        if not innerapprox:
            left_bv  = self.pt2bv(left - abs_tol, nbits = nbits)
            right_bv = self.pt2bv(right + abs_tol, nbits = nbits)
        else:
            # Inner approximations move in the box
            left_bv  = self.pt2bv(left - abs_tol, abs_tol, nbits)
            left_bv = increment_bv(left_bv, 1, self.periodic, saturate= True)
            right_bv = self.pt2bv(right + abs_tol, abs_tol, nbits)
            right_bv = increment_bv(right_bv, -1, self.periodic, saturate= True)

        if self.periodic: #TODO: Implement 
            raise NotImplementedError

        box = self.mgr.false
        if not self.periodic and (left_bv > right_bv):
            return box

        b = bv_interval(left_bv, right_bv, self.periodic)
        for i in map(lambda x: self.bv2bdd(x), b):
            box |= i

        assert len(box.support) <= nbits, "Support " + str(box.support) + "exceeds " + nbits + " bits"
        return box

    def ___compare_attr(self, other):
        if type(self) != type(other):
            return False 
        if self.mgr != other.mgr:
            return False # ValueError("Different BDD managers")
        if self.name != other.name:
            return False # ValueError("Different BDD names")
        if self.periodic != other.periodic:
            return False # ValueError("")
        if self.lb != other.lb or self.ub != other.ub:
            return False # raise ValueError("Different boundaries")
        return True 

    # Compare precisions 
    def __lt__(self, other):
        if self.___compare_attr(other) and self.num_bits < other.num_bits:
            return True
        return False

    def __le__(self, other):
        if self.___compare_attr(other) and self.num_bits <= other.num_bits:
            return True
        return False

    def __gt__(self, other):
        if self.___compare_attr(other) and self.num_bits > other.num_bits:
            return True
        return False

    def __ge__(self, other):
        if self.___compare_attr(other) and self.num_bits >= other.num_bits:
            return True
        return False

    def __eq__(self,other):
        if self.___compare_attr(other) and self.num_bits == other.num_bits:
            return True
        return False

    # TODO: Iterate over all boxes in the symbolic interval 
    def __iter__(self):
        raise NotImplementedError


class FixedInterval(ContinuousInterval):
    """
    There are some assignments to bits that are not valid for this set
    Fixed number of "bins" 
    """
    def __init__(self, name, mgr, lb, ub, bins: int, periodic = False):
        # Interval spacing 
        assert bins > 0, "Cannot have negative grid spacing"
        self.bins = bins
        #self.binwidth = (ub-lb) / bins 
        num_bits = math.ceil(math.log2(bins))
        
        #self.num_bits = math.ceil(math.log2((ub-lb)/eta + 1))
        ContinuousInterval.__init__(self, name, mgr, lb, ub, num_bits, periodic)

    def renamed(self, newname):
        return FixedInterval(newname, self.mgr, self.lb, self.ub, self.bins, self.periodic)

    def __repr__(self):
        s = "Fixed Interval, "
        if self.periodic:
            s += "Periodic, "
        s += "Bounds: {0}, ".format(str(self.bounds))
        s += "# Bins: " + str(self.bins) + ", "
        s += "Variables: " + " ".join([str(k) for k in self.bits]) 
        return s

    @property
    def binwidth(self):
        return (self.ub-self.lb) / self.bins 

    def pt2box(self, point):
        return self.bv2box(self.pt2bv(point))

    def pt2bv(self, point, tol = 0.0):

        index = self.pt2index(point, tol)
        return _int2bv(index, self.num_bits)
    
    def pt2index(self, point, tol = 0.0):
        if self.periodic:
            point = self._wrap(point)
        
        assert point <= self.ub + tol and point >= self.lb - tol

        index = int(((point - self.lb) / (self.ub - self.lb)) * self.bins)

        # Catch numerical errors when point == self.ub 
        if index >= self.bins:
            index = self.bins - 1

        return index

    def bv2box(self, bv):
        # TODO: Check that the bitvector isn't out of bounds!
        left = self.lb
        left += _bv2int(bv) * self.binwidth
        right = left + self.binwidth
        return (left, right)

    def box2bvs(self, box, innerapprox = False, tol = .0000001):
        left, right = box

        eps = self.binwidth
        abs_tol = eps * tol

        assert left <= right
        
        if not innerapprox:
            left  = self.pt2index(left - abs_tol)
            right = self.pt2index(right + abs_tol)
        else:
            # Inner approximations move in the box
            left  = self.pt2index(left - abs_tol, abs_tol) + 1 
            right = self.pt2index(right + abs_tol, abs_tol) - 1

        if self.periodic:
            raise NotImplementedError

        left_bv  = _int2bv( left, self.num_bits)
        right_bv = _int2bv(right, self.num_bits)

        return bv_interval(left_bv, right_bv)

    def box2bdd(self, box, innerapprox = False, tol = .0000001):

        left, right = box

        # eps = self.binwidth
        # abs_tol = eps * tol

        # assert left <= right
        # if not innerapprox:
        #     left_bv  = self.pt2bv(left - abs_tol)
        #     right_bv = self.pt2bv(right + abs_tol)
        # else:
        #     # Inner approximations move in the box
        #     left  = self.pt2index(left - abs_tol, abs_tol) + 1 
        #     right = self.pt2index(right + abs_tol, abs_tol) - 1

        # if self.periodic:
        #     raise NotImplementedError

        # left_bv  = _int2bv( left, self.num_bits)
        # right_bv = _int2bv(right, self.num_bits)

        # b = bv_interval(left_bv, right_bv)
        b = self.box2bvs(box, innerapprox, tol)
        boxbdd = self.mgr.false
        for i in map(lambda x: self.bv2bdd(x), b):
            boxbdd |= i

        return boxbdd

