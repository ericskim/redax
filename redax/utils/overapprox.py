import numpy as np

def shiftbox(box, fraction=0.0):
    width = box[1] - box[0]
    return box[0] + fraction*width, box[1] + fraction*width

def bloatbox(box, factor = .001):
    assert factor > -1.0
    center = box[0] + .5 * (box[1]-box[0])
    return box[0] - factor*(center - box[0]), box[1] + factor*(box[1] - center)

def containszero(left, right):
    """Determine if 0 is contained in a periodic interval [left,right]."""

    assert left <= right

    # Map to interval [-pi,pi]
    left = ((left + np.pi) % (np.pi*2) ) - np.pi
    left = left - 2*np.pi if left > np.pi else left
    right = ((right + np.pi) % (np.pi*2) ) - np.pi
    right = right - 2*np.pi if right > np.pi else right

    if right < left:
        if left <= 0:
            return True
    else:
        if left <= 0 and right >= 0:
            return True

    return False

def maxmincos(left, right):
    """Compute the maximum and minimum values of cos in an interval."""

    assert left <= right
    if right - left > np.pi * 2:
        return (-1, 1)

    if containszero(left, right) is True:
        maxval = 1
    else:
        maxval = max([np.cos(left), np.cos(right)])

    if containszero(left + np.pi, right + np.pi) is True:
        minval = -1
    else:
        minval = min([np.cos(left), np.cos(right)])

    return (minval, maxval)

def maxminsin(left, right):
    """Compute the maximum and minimum values of sin in an interval."""
    return maxmincos(left - np.pi/2, right - np.pi/2)