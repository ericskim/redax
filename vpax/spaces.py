"""
Symbolic Intervals

# Discrete vs continuous, fixed vs dynamic, linear vs periodic

# discrete => fixed, linear
# continuous => either linear or periodic, either fixed or dynamic

Nomenclature:
    bv = Bit vector 
    box = 
    point = element of set
    grid = a countable set of points embedded in continuous space 
"""

import math 
import itertools

import numpy as np
from abc import abstractmethod

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def _bintogray(x:int):
    """
    Converts a binary encoded positive integer into gray code 
    """
    assert x >= 0 
    return x ^ (x >> 1)

def _graytobin(x:int):
    """
    Converts a gray code encoded positive integer into the standard binary encoding 
    """
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

    return tuple(True if ((index >> i) % 2 == 1) else False for i in range(nbits-1,-1,-1))

def _bv2int(bv):
    """
    Converts bitvector (list or tuple) with the standard binary encoding into an integer
    """
    nbits = len(bv)
    index = 0
    for i in range(nbits):
        if bv[i]:
            index += 2**(nbits - i - 1)
    return index

def increment_bv(bv, increment, graycode = False, saturate = False):
    """
    Increment a bitvector's value +1 or -1. 
    """
    assert increment == 1 or increment == -1
    nbits = len(bv)
    if graycode:
        index = _graytobin(_bv2int(bv))
        index = (index+increment) % 2**nbits
        return _int2bv( _bintogray(index), nbits)
    else:
        if bv == tuple(True for i in range(nbits)) and increment > 0: 
            if saturate:
                return bv
            raise ValueError("Bitvector overflow for nonperiodic domain.")
        if bv == tuple(False for i in range(nbits)) and increment < 0:
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

def bv2pred(mgr, name, bv):
    """
    If bv's size is smaller than the bits allocated, then it takes the most significant ones
    """

    for i in range(len(bv)):
        mgr.declare(name + "_" + str(i))
    b = mgr.true 
    for i in range(len(bv)):
        if bv[i]:
            b = b &  mgr.var(name + "_" + str(i))
        else:
            b = b & ~mgr.var(name + "_" + str(i))
    return b

class SymbolicSet(object):
    """
    Abstract class for representing a concrete underlying space and handling translation to symbolic encodings

    """

    def __init__(self):
        pass 

    @abstractmethod
    def pt2bv(self, point):
        raise NotImplementedError 
    
    
class DiscreteSet(SymbolicSet):
    """
    Discrete Interval 
    """
    def __init__(self, num_vals):
        SymbolicSet.__init__(self)
        self.num_vals = num_vals
        self.num_bits = math.ceil(math.log2(num_vals))

    def pt2bv(self, point):
        assert point < self.num_vals
        return _int2bv(point, self.num_bits)

    @property 
    def bounds(self):
        return (0, self.num_vals-1)

    def __repr__(self):
        s = "Discrete Set, "
        s += "Bounds: [0,...,{0}]".format(str(self.num_vals-1))
        return s

    def abs_space(self, mgr, name):
        """
        Returns the predicate of the discrete set abstract space

        Args:
            mgr: bdd manager
            name: BDD variable name, e.g. "x" for names "x_1", "x_2", etc. 
        """
        left_bv =  _int2bv(0, self.num_bits)
        right_bv = _int2bv(self.num_vals-1,self.num_bits)
        bvs = bv_interval(left_bv, right_bv)
        boxbdd = mgr.false
        for i in map(lambda x: bv2pred(mgr, name, x), bvs):
            boxbdd |= i
        return boxbdd 


class EmbeddedGrid(DiscreteSet):
    """
    A discrete grid of points embedded in continuous space

    Args:
        left (float): Left point (inclusive)
        right (float): Right point (inclusive)
        num (int): Number of points

    EmbeddedGrid(-2,2,4) corresponds with points [-2, -2/3, 2/3, 2]
    """
    def __init__(self, left, right, num):
        if num <= 0:
            raise ValueError("Grid must have at least one point")
        if left > right:
            raise ValueError("Left point is greater than right")
        if num == 1 and left != right:
            raise ValueError("Single point but left and right are not equal")

        DiscreteSet.__init__(self, num)
        self.left = left
        self.right = right
        self.pts = np.linspace(self.left, self.right, self.num_vals)


    def pt2index(self, pt, snap = False):
        """
        Args:
            pt  (float): Continuous point to be converted into a bitvector
            snap (bool): Snaps pt to the nearest discrete point with preference towards greater point.
                         Otherwise requires exact equality.

        Returns:
            int 
        """
        if snap:
            return find_nearest(self.pts, pt)
        elif pt in self.pts:
            return np.searchsorted(self.pts, pt)
        else:
            raise ValueError("Cannot find an exact match without snapping")


    def conc2pred(self, mgr, name, concrete, snap = True):
        """
        Translates from a concrete value to a the associated predicate

        Args: 
            mgr (dd mgr):
            name:
            concrete:
            snap:
        Returns:
            bdd: 

        """
        bv = _int2bv(self.pt2index(concrete, snap), self.num_bits)
        return bv2pred(mgr, name, bv)

    def conc_iter(self):
        """
        Iterable of points 
        """
        return self.pts
    
    def __repr__(self):
        s = "Embedded Grid({0},{1},{2})".format(str(self.left), str(self.right), str(self.num_vals))
        return s

class ContinuousPartition(SymbolicSet):
    """
    Continuous Interval
    """
    def __init__(self, lb, ub, periodic = False):
        assert ub - lb >= 0.0, "Upper bound is smaller than lower bound"
        SymbolicSet.__init__(self)
        self.periodic = periodic
        self.lb = float(lb)
        self.ub = float(ub)

    @property
    def bounds(self):
        return (self.lb, self.ub)

    def width(self):
        return self.ub - self.lb

    def _wrap(self, point):
        """
        Helper function for periodic intervals
        """

        if point == self.ub:
            return point
        width = self.ub - self.lb
        point =  (point - self.lb) % width
        return point + self.lb

    @abstractmethod
    def pt2bv(self, point):
        raise NotImplementedError 

    def __eq__(self, other):
        if self.periodic != other.periodic:
            return False
        if self.lb != other.lb:
            return False
        if self.ub != other.ub:
            return False
        return True

    def conc_space(self):
        """
        Returns the concrete space 
        """
        return self.bounds

    @abstractmethod
    def abs_space(self):
        """
        Abstract space predicate 
        """
        raise NotImplementedError 

class DynamicPartition(ContinuousPartition):
    """
    Dynamically partitions the space with a variable number of bits. 
    Number of partitions is always a power of two. 

    Args:
        lb (float): Lower bound of interval being partitioned
        ub (float): Upper bound of interval being partitioned
        periodic (bool): If true, uses gray code encoding. 
    """
    def __init__(self, lb, ub, periodic = False):
        ContinuousPartition.__init__(self, lb, ub, periodic)

    def __repr__(self):
        s = "DynamicPartition({0}, {1}, periodic={2})".format(self.lb, self.ub, self.periodic)
        # s += "Bounds: {0} ".format(str(self.bounds))
        # if self.periodic:
        #     s += ", Periodic"
        return s

    def abs_space(self, mgr, name = None):
        """
        Returns the predicate of the abstract space
        """
        return mgr.true 

    def pt2bv(self, point, nbits, tol = 0.0): 
        """
        Continuous point to a bitvector 
        """
        assert isinstance(nbits, int)

        if self.periodic:
            point = self._wrap(point)

        assert point <= self.ub + tol, "Point {0} exceeds upper bound {1}".format(point, self.ub + tol)
        assert point >= self.lb - tol, "Point {0} exceeds lower bound {1}".format(point, self.lb - tol)
        
        # FIXME: Use a binary search and queries for membership in intervals instead. 
        index = int((point - self.lb) / (self.ub - self.lb) * 2**nbits)

        # Catch numerical errors when point == self.ub 
        if index >= 2**nbits:
            index = 2**nbits - 1

        if self.periodic:
            index = _bintogray(index)

        return _int2bv(index,nbits)

    def pt2bdd(self, mgr, name, pt, nbits, innerapprox = False, tol = .00001):
        return bv2pred(mgr, name, self.pt2bv(pt, nbits))

    def bv2box(self, bv):
        nbits = len(bv)

        if nbits == 0:
            return (self.lb, self.ub)

        index = _bv2int(bv)
        
        if self.periodic:
            index = _graytobin(index)

        eps = (self.ub - self.lb) / (2**nbits)
        left  = self.lb + index * eps
        right = self.lb + (index+1) * eps
        
        return (left, right)

    def pt2box(self, point, nbits):
        return self.bv2box(self.pt2bv(point, nbits = nbits))

    def box2bvs(self, box, nbits, innerapprox = False, tol = .0000001):
        """
        Returns a list of bitvectors corresponding to a box

        Args:
            box (2-tuple): Left and right floating points
            nbits (int): Number of bits 
            innerapprox: 
            tol:         Numerical tolerance 
        """
        left, right = box

        assert tol >= 0 and tol <= 1, "Tolerance is not 0 <= tol <= 1"
        eps = (self.ub - self.lb) / (2**nbits)
        abs_tol = eps * tol

        # TODO: Check for out of bounds error here!
        # TODO: Variable declarations here!

        if innerapprox:
            # Inner approximations move in the box
            left_bv  = self.pt2bv(left - abs_tol, nbits, tol = abs_tol)
            right_bv = self.pt2bv(right + abs_tol, nbits, tol = abs_tol)
            if left_bv == right_bv: # In same box e.g. [.4,.6] <= [0,1]
                return [] 
            left_bv = increment_bv(left_bv, 1, self.periodic, saturate = True)
            if left_bv == right_bv: # Adjacent boxes [.4,.6] overlaps [0,.5] and [.5,1]
                return []
            right_bv = increment_bv(right_bv, -1, self.periodic, saturate = True)
        else:
            left_bv  = self.pt2bv(left - abs_tol, nbits = nbits, tol = abs_tol)
            right_bv = self.pt2bv(right + abs_tol, nbits = nbits, tol = abs_tol)

        if not self.periodic and (left_bv > right_bv):
            raise ValueError("{0}: {1}\n{2}: {3}".format(left, left_bv,right, right_bv))

        return bv_interval(left_bv, right_bv, self.periodic)

    def conc2pred(self, mgr, name, box, nbits, innerapprox = False, tol = .00001):
        """
        Overapproximation of a concrete box with its BDD

        Args: 
            name
            box 
            innerapprox - 
            tol - tolerance for numerical errors as a fraction of the grid size. 
                     Must lie within [0,1]
        """ 

        predbox = mgr.false
        for bv in self.box2bvs(box, nbits, innerapprox, tol):
            predbox |= bv2pred(mgr, name, bv)

        assert len(predbox.support) <= nbits, "Support " + str(predbox.support) + "exceeds " + nbits + " bits"
        return predbox

    def __eq__(self, other):
        if type(self) != type(other):
            return False 
        if self.periodic != other.periodic:
            return False
        if self.lb != other.lb or self.ub != other.ub:
            return False
        return True 

    def conc_iter(self, prec):
        i = 0
        while(i < 2**prec):
            yield self.bv2box(_int2bv(i, prec))
            i += 1

class FixedPartition(ContinuousPartition):
    """
    There are some assignments to bits that are not valid for this set
    Fixed number of "bins"

    Example:
        FixedPartition(2, 10, 4) corresponds to the four bins
            [2, 4] [4,6] [6,8] [8,10] 
    """
    def __init__(self, lb: float, ub:float, bins: int, periodic = False):
        # Interval spacing 
        assert bins > 0, "Cannot have negative grid spacing"
        self.bins = bins
        
        ContinuousPartition.__init__(self, lb, ub, periodic)

    def __eq__(self, other):
        if ContinuousPartition.__eq__(self, other):
            if self.bins == other.bins:
                return True
        return False
    @property
    def num_bits(self):
        return math.ceil(math.log2(self.bins))

    def __repr__(self):
        s = "FixedPartition({0}, {1}, bins = {2}, periodic={3})".format(self.lb, self.ub, self.bins, self.periodic)
        return s

    def abs_space(self, mgr, name = None):
        """
        Returns the predicate of the fixed partition abstract space
        """
        left_bv =  _int2bv(0, self.num_bits)
        right_bv = _int2bv(self.bins-1,self.num_bits)
        bvs = bv_interval(left_bv, right_bv)
        boxbdd = mgr.false
        for i in map(lambda x: bv2pred(mgr, name, x), bvs):
            boxbdd |= i
        return boxbdd 

    @property
    def binwidth(self):
        return (self.ub-self.lb) / self.bins 

    def dividers(self, nbits):
        raise NotImplementedError

    def pt2box(self, point):
        return self.bv2box(self.pt2bv(point))

    def pt2bv(self, point, tol = 0.0):
        """
        Maps a point to bitvector corresponding to the bin that contains it 
        """
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
        if len(bv) == 0:
            return (self.lb, self.ub)

        left = self.lb
        left += _bv2int(bv) * self.binwidth
        right = left + self.binwidth
        return (left, right)

    def box2bvs(self, box, innerapprox = False, tol = .0000001):
        left, right = box

        eps = self.binwidth
        abs_tol = eps * tol

        # assert left <= right

        left  = self.pt2index(left - abs_tol)
        right = self.pt2index(right + abs_tol)        
        if innerapprox:
            # Inner approximations move in the box
            if left == right or left == right - 1:
                return []
            
            if self.periodic and left == self.bins-1 and right == 0:
                return []
            else: 
                left  = (left + 1) % self.bins
                right = (right - 1) % self.bins 

        left_bv  = _int2bv( left, self.num_bits)
        right_bv = _int2bv(right, self.num_bits)

        if self.periodic and left > right:
            zero_bv = _int2bv(0, self.num_bits)
            max_bv = _int2bv(self.bins-1, self.num_bits)
            return itertools.chain(bv_interval(zero_bv, right_bv), 
                                   bv_interval(left_bv, max_bv))
        else: 
            return bv_interval(left_bv, right_bv)

    def conc2pred(self, mgr, name, box, innerapprox = False, tol = .0000001):

        # left, right = box

        b = self.box2bvs(box, innerapprox, tol)
        boxbdd = mgr.false
        for i in map(lambda x: bv2pred(mgr, name, x), b):
            boxbdd |= i

        return boxbdd 
