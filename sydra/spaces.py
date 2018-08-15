r"""

Nomenclature:
bv = Bit vector \n
box = \n
point = element of set \n
cover = \n
grid = a countable set of points embedded in continuous space \n
"""

import itertools
import math
from abc import abstractmethod
from typing import Iterable, Tuple

import numpy as np

from sydra.utils import bv2pred, bvwindow, bvwindowgray, BitVector, int2bv, bv2int, bintogray, graytobin, increment_bv, bv_interval



class SymbolicSet(object):
    """
    Abstract class for representing a concrete underlying space and handling translation to symbolic encodings.

    """

    def __init__(self) -> None:
        pass


class DiscreteSet(SymbolicSet):
    """
    Discrete set abstract class
    """
    def __init__(self, num_vals) -> None:
        SymbolicSet.__init__(self)
        self.num_vals = num_vals
        self.num_bits = math.ceil(math.log2(num_vals))

    def pt2bv(self, point) -> BitVector:
        assert point < self.num_vals
        return int2bv(point, self.num_bits)

    @property
    def bounds(self) -> Tuple[float, float]:
        return (0, self.num_vals-1)

    def __repr__(self):
        s = "Discrete Set, "
        s += "Bounds: [0,...,{0}]".format(str(self.num_vals-1))
        return s

    def abs_space(self, mgr, name: str):
        """
        Returns the predicate of the discrete set abstract space

        Parameters
        ----------
            mgr: bdd 
                manager
            name: str
                BDD variable name, e.g. "x" for names "x_1", "x_2", etc.

        """
        left_bv =  int2bv(0, self.num_bits)
        right_bv = int2bv(self.num_vals-1,self.num_bits)
        bvs = bv_interval(left_bv, right_bv)
        boxbdd = mgr.false
        for i in map(lambda x: bv2pred(mgr, name, x), bvs):
            boxbdd |= i
        return boxbdd


class EmbeddedGrid(DiscreteSet):
    """
    A discrete grid of points embedded in continuous space.

    Parameters
    ----------
        left : float
            Left point (inclusive)
        right : float
            Right point (inclusive)
        num : int
            Number of points in the grid

    Example
    -------
    EmbeddedGrid(-2,2,4) corresponds with points [-2, -2/3, 2/3, 2]

    """

    def __init__(self, left: float, right: float, num: int) -> None:
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

    def find_nearest_index(self, array, value) -> int:
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    def pt2index(self, pt: float, snap=False) -> int:
        """
        Parameters
        ----------
        pt : float
            Continuous point to be converted into a bitvector
        snap : bool
            Snaps pt to the nearest discrete point with preference towards greater point. Otherwise requires exact equality.

        Returns
        -------
        int

        """
        if snap:
            return self.find_nearest_index(self.pts, pt)
        elif pt in self.pts:
            return np.searchsorted(self.pts, pt)
        else:
            raise ValueError("Cannot find an exact match without snapping")

    def conc2pred(self, mgr, name: str, concrete, snap=True):
        """
        Translate from a concrete value to a the associated predicate.

        Parameters
        ----------
            mgr (dd mgr):
            name:
            concrete:
            snap:

        Returns
        -------
            bdd:

        """
        bv = int2bv(self.pt2index(concrete, snap), self.num_bits)
        return bv2pred(mgr, name, bv)

    def conc_iter(self) -> Iterable:
        """
        Iterable of points
        """
        return self.pts

    def __repr__(self):
        s = "Embedded Grid({0}, {1}, {2})".format(str(self.left), str(self.right), str(self.num_vals))
        return s

    def bv2conc(self, bv: BitVector) -> float:
        """
        Converts a bitvector into a concrete grid point
        """
        return self.pts[bv2int(bv)]

class ContinuousCover(SymbolicSet):
    """
    Continuous Interval
    """
    def __init__(self, lb, ub, periodic=False) -> None:
        assert ub - lb >= 0.0, "Upper bound is smaller than lower bound"
        SymbolicSet.__init__(self)
        self.periodic = periodic
        self.lb = float(lb)
        self.ub = float(ub)

    @property
    def bounds(self) -> Tuple[float, float]:
        return (self.lb, self.ub)

    def width(self) -> float:
        return self.ub - self.lb

    def _wrap(self, point: float):
        """Helper function for periodic intervals."""

        if point == self.ub:
            return point
        width = self.ub - self.lb
        return ((point - self.lb) % width) + self.lb

    @abstractmethod
    def pt2bv(self, point: float):
        raise NotImplementedError

    def __eq__(self, other) -> bool:
        if self.periodic != other.periodic:
            return False
        if self.lb != other.lb:
            return False
        if self.ub != other.ub:
            return False
        return True

    def conc_space(self) -> Tuple[float, float]:
        """Concrete space."""
        return self.bounds

class DynamicCover(ContinuousCover):
    """
    Dynamically covers the space with a variable number of bits.
    Number of covers is always a power of two.

    """
    def __init__(self, lb: float, ub: float, periodic=False) -> None:
        """
        Parameters
        ----------
        lb : float
            Lower bound of interval being covered
        ub : float
            Upper bound of interval being covered
        periodic : bool
            If true, uses gray code encoding.
        """
        ContinuousCover.__init__(self, lb, ub, periodic)

    def __repr__(self):
        s = "DynamicCover({0:.4}, {1:.4}, periodic={2})".format(self.lb,
                                                                self.ub,
                                                                self.periodic)
        # s += "Bounds: {0} ".format(str(self.bounds))
        # if self.periodic:
        #     s += ", Periodic"
        return s

    def abs_space(self, mgr, name:str):
        """
        Returns the predicate of the abstract space
        """
        return mgr.true

    def pt2bv(self, point: float, nbits: int, tol=0.0):
        """
        Continuous point to a bitvector

        Parameters
        ----------
        point : float
            Continuous point
        nbits : int
            Number of bits to encode

        Returns
        -------
        Tuple of bools:
            Left-most element is most significant bit. Encoding is the standard binary encoding or gray code. 

        """
        assert isinstance(nbits, int)

        if self.periodic:
            point = self._wrap(point)

        index = self.pt2index(point, nbits, alignleft=True)

        if self.periodic:
            index = bintogray(index)

        return int2bv(index, nbits)

    def pt2index(self, point: float, nbits: int, alignleft=True, tol=0.0) -> int:
        """Convert a floating point into an integer index of the cover the point lies in."""
        assert isinstance(nbits, int)

        if self.periodic:
            point = self._wrap(point)

        assert point <= self.ub + tol, "Point {0} exceeds upper bound {1}".format(point, self.ub + tol)
        assert point >= self.lb - tol, "Point {0} exceeds lower bound {1}".format(point, self.lb - tol)

        bucket_fraction = 2**nbits * (point - self.lb) / (self.ub - self.lb)

        index = math.floor(bucket_fraction) if alignleft else math.ceil(bucket_fraction)

        # Catch numerical errors when point == self.ub
        # if alignleft is True and index >= 2**nbits:
        #     index = (2**nbits) - 1

        return index

    def pt2bdd(self, mgr, name, pt: float, nbits: int):
        return bv2pred(mgr, name, self.pt2bv(pt, nbits))

    def bv2conc(self, bv: BitVector) -> Tuple[float, float]:
        nbits = len(bv)

        if nbits == 0:
            return (self.lb, self.ub)

        index = bv2int(bv)

        if self.periodic:
            index = graytobin(index)

        eps = (self.ub - self.lb) / (2**nbits)
        left = self.lb + index * eps
        right = self.lb + (index+1) * eps

        return (left, right)

    def pt2box(self, point: float, nbits: int):
        return self.bv2conc(self.pt2bv(point, nbits=nbits))

    def box2bvs(self, box, nbits: int, innerapprox=False, tol=.0000001):
        """
        Returns a list of bitvectors corresponding to a box

        Parameters
        ----------
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

        if innerapprox:
            # Inner approximations move in the box
            left_bv = self.pt2bv(left - abs_tol, nbits, tol=abs_tol)
            right_bv = self.pt2bv(right + abs_tol, nbits, tol=abs_tol)
            if left_bv == right_bv:  # In same box e.g. [.4,.6] <= [0,1]
                return []
            left_bv = increment_bv(left_bv, 1, self.periodic, saturate=True)
            if left_bv == right_bv:  # Adjacent boxes [.4,.6] overlaps [0,.5] and [.5,1]
                return []
            right_bv = increment_bv(right_bv, -1, self.periodic, saturate=True)
        else:
            left_bv = self.pt2bv(left - abs_tol, nbits=nbits, tol=abs_tol)
            right_bv = self.pt2bv(right + abs_tol, nbits=nbits, tol=abs_tol)

        if not self.periodic and (left_bv > right_bv):
            raise ValueError("{0}: {1}\n{2}: {3}".format(left, left_bv, right, right_bv))

        return bv_interval(left_bv, right_bv, self.periodic)


    def box2indexwindow(self, box, nbits: int, innerapprox=False, tol=.0000001):
        """
        Returns
        -------
        None or Tuple:
            None if the index window is empty and (left, right) index tuple otherwise
        """
        left, right = box
        assert tol >= 0 and tol <= 1, "Tolerance is not 0 <= tol <= 1"
        eps = (self.ub - self.lb) / (2**nbits)
        abs_tol = eps * tol

        if nbits == 0 and self.periodic:
            return None if innerapprox else (0, 0)

        if self.periodic:
            left = self._wrap(left)
            right = self._wrap(right)

        # If true, then the XORs flip the usual value
        flip = True if self.periodic and right < left else False

        if innerapprox:
            leftidx = self.pt2index(left, nbits, alignleft=False)
            rightidx = self.pt2index(right, nbits, alignleft=True)

            # Left and right were in the same bin. Return empty window! 
            if self.periodic is False and leftidx >= rightidx:
                return None 
            if self.periodic is True and not flip and leftidx >= rightidx:
                return None
            # Left boundary is near the upper value, right boundary is near lower value
            if flip and leftidx == (2**nbits) and rightidx == 0:
                return None
            
        else: 
            leftidx = self.pt2index(left, nbits, alignleft=True)
            rightidx = self.pt2index(right, nbits, alignleft=False)

            if flip and (leftidx + 1) == rightidx:
                return (leftidx + 1) % (2**nbits), (rightidx-1) % (2**nbits)
            if flip and leftidx == (2**nbits) and rightidx == 0:
                return (0,(2**nbits)-1)

        # Weird off by one errors happen.
        rightidx = (rightidx - 1) % (2**nbits)
        leftidx = leftidx % (2**nbits)

        return leftidx, rightidx


    def conc2pred(self, mgr, name: str, box, nbits: int, innerapprox=False, tol=.00001):
        predbox = mgr.false

        window = self.box2indexwindow(box, nbits, innerapprox, tol)
        if window == None:
            return predbox
        leftidx, rightidx = window

        if self.periodic: 
            for bv in bvwindowgray(leftidx, rightidx, nbits):
                predbox |= bv2pred(mgr, name, bv)
        else:
            for bv in bvwindow(leftidx, rightidx, nbits):
                predbox |= bv2pred(mgr, name, bv)
        return predbox

    def conc2predold(self, mgr, name: str, box, nbits: int, innerapprox=False, tol=.00001):
        """
        Overapproximation of a concrete box with its BDD.

        Parameters
        ----------
        name
        box
        innerapprox -
        tol - tolerance for numerical errors as a fraction of the grid size. Must lie within [0,1]
        """
        assert nbits >= 0

        predbox = mgr.false
        for bv in self.box2bvs(box, nbits, innerapprox, tol):
            predbox |= bv2pred(mgr, name, bv)

        assert len(predbox.support) <= nbits, "Support " + str(predbox.support) + "exceeds " + str(nbits) + " bits"
        return predbox

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.periodic != other.periodic:
            return False
        if self.lb != other.lb or self.ub != other.ub:
            return False
        return True

    def conc_iter(self, prec: int):
        """
        Generator for iterating over the space with fixed precision

        Yields
        ------
        Tuple:
            (left, right) box of floats 
        """
        i = 0
        while(i < 2**prec):
            yield self.bv2conc(int2bv(i, prec))
            i += 1


class FixedCover(ContinuousCover):
    """
    There are some assignments to bits that are not valid for this set
    Fixed number of "bins"

    Parameters
    ----------
    lb: float
        Lower bound
    ub: float
        Upper bound
    bins: int
        Number of bins
    periodic: bool, default False
        If true, cover loops around so lb and ub represent the same point

    Examples
    --------
    FixedCover(2, 10, 4, False) corresponds to the four bins
        [2, 4] [4,6] [6,8] [8,10]
    """
    def __init__(self, lb: float, ub: float, bins: int, periodic: bool=False) -> None:
        # Interval spacing
        assert bins > 0, "Cannot have negative grid spacing"
        self.bins = bins

        ContinuousCover.__init__(self, lb, ub, periodic)

    def __eq__(self, other):
        if ContinuousCover.__eq__(self, other):
            if self.bins == other.bins:
                return True
        return False
    @property
    def num_bits(self):
        return math.ceil(math.log2(self.bins))

    def __repr__(self):
        s = "FixedCover({0}, {1}, bins = {2}, periodic={3})".format(self.lb, self.ub, self.bins, self.periodic)
        return s

    def abs_space(self, mgr, name: str):
        """
        Returns the predicate of the fixed cover abstract space
        """
        left_bv = int2bv(0, self.num_bits)
        right_bv = int2bv(self.bins - 1, self.num_bits)
        bvs = bv_interval(left_bv, right_bv)
        boxbdd = mgr.false
        for i in map(lambda x: bv2pred(mgr, name, x), bvs):
            boxbdd |= i
        return boxbdd

    @property
    def binwidth(self):
        return (self.ub-self.lb) / self.bins

    def dividers(self, nbits: int):
        raise NotImplementedError

    def pt2box(self, point: float):
        return self.bv2conc(self.pt2bv(point))

    def pt2bv(self, point: float, tol=0.0):
        """
        Maps a point to bitvector corresponding to the bin that contains it
        """
        index = self.pt2index(point, tol)
        return int2bv(index, self.num_bits)

    def pt2index(self, point: float, alignleft=True, tol=0.0) -> int:
        if self.periodic:
            point = self._wrap(point)

        assert point <= self.ub + tol and point >= self.lb - tol

        # index = int(((point - self.lb) / (self.ub - self.lb)) * self.bins)
        bucket_fraction = ((point - self.lb) / (self.ub - self.lb)) * self.bins
        index = math.floor(bucket_fraction) if alignleft else math.ceil(bucket_fraction)

        # Catch numerical errors when point == self.ub
        # if index >= self.bins:
        #     index = self.bins - 1

        return index

    def bv2conc(self, bv: BitVector) -> Tuple[float, float]:
        if len(bv) == 0:
            return (self.lb, self.ub)

        left = self.lb
        left += bv2int(bv) * self.binwidth
        right = left + self.binwidth
        return (left, right)

    def box2bvs(self, box, innerapprox=False, tol=.0000001):
        left, right = box

        eps = self.binwidth
        abs_tol = eps * tol

        # assert left <= right

        left = self.pt2index(left - abs_tol)
        right = self.pt2index(right + abs_tol)
        if innerapprox:
            # Inner approximations move in the box
            if left == right or left == right - 1:
                return []

            if self.periodic and left == self.bins-1 and right == 0:
                return []
            else:
                left = (left + 1) % self.bins
                right = (right - 1) % self.bins

        left_bv = int2bv(left, self.num_bits)
        right_bv = int2bv(right, self.num_bits)

        if self.periodic and left > right:
            zero_bv = int2bv(0, self.num_bits)
            max_bv = int2bv(self.bins-1, self.num_bits)
            return itertools.chain(bv_interval(zero_bv, right_bv),
                                   bv_interval(left_bv, max_bv))
        else:
            return bv_interval(left_bv, right_bv)

    def box2indexwindow(self, box, innerapprox=False, tol=.00001):
        left, right = box
        if self.bins == 1 and self.periodic:
            return None if innerapprox else (0, 0)

        if self.periodic:
            left = self._wrap(left)
            right = self._wrap(right)
        
        flip = True if self.periodic and right < left else False

        if innerapprox:
            leftidx = self.pt2index(left, alignleft=False)
            rightidx = self.pt2index(right, alignleft=True)

            # Left and right were in the same bin. Return empty window! 
            if self.periodic is False and leftidx >= rightidx:
                return None 
            if self.periodic is True and not flip and leftidx >= rightidx:
                return None
            # Left boundary is near the upper value, right boundary is near lower value
            if flip and leftidx == self.bins and rightidx == 0:
                return None
        else:
            leftidx = self.pt2index(left, alignleft=True)
            rightidx = self.pt2index(right, alignleft=False)
            print(leftidx, right, rightidx)
            # Contained in same interval but bloating 
            if flip and (leftidx + 1) == rightidx:
                return (leftidx + 1) % self.bins, (rightidx-1) % self.bins
            if flip and leftidx == self.bins and rightidx == 0:
                return (0, self.bins-1)  # Entire interval

        rightidx = (rightidx - 1) % self.bins
        leftidx = leftidx % self.bins

        return leftidx, rightidx

    def conc2pred(self, mgr, name: str, box, innerapprox=False, tol=.00001):
        predbox = mgr.false

        window = self.box2indexwindow(box, innerapprox, tol)
        if window is None:
            return predbox
        leftidx, rightidx = window

        if leftidx > rightidx:
            for bv in bvwindow(leftidx, self.bins-1, self.num_bits):
                predbox |= bv2pred(mgr, name, bv)
            for bv in bvwindow(0, rightidx, self.num_bits):
                predbox |= bv2pred(mgr, name, bv)
        else: 
            for bv in bvwindow(leftidx, rightidx, self.num_bits):
                predbox |= bv2pred(mgr, name, bv)
        return predbox

    def conc2predold(self, mgr, name, box, innerapprox=False, tol=.0000001):

        # left, right = box

        b = self.box2bvs(box, innerapprox, tol)
        boxbdd = mgr.false
        for i in map(lambda x: bv2pred(mgr, name, x), b):
            boxbdd |= i

        return boxbdd
