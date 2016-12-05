import math
import numpy as np
import itertools
import scipy.misc
import scipy.ndimage.filters
import scipy.special
import random
import copy
import pickle

def plot(net, fileNum):
    nG = len(net[0][0])     # Plot width
    nI = len(net[0]   )     # Input size / 2
    nL = len(net      )     # num layers

    pI = nI * nG + nI
    pJ = nL * nG + nL
    im = [[0.0 for i in range(pI)] for j in range(pJ)]
    for i, layer in enumerate(net):
        for j, unit in enumerate(layer):
            for k in range(nG):
                for l in range(nG):
                    K = k / float(nG - 1)
                    L = l / float(nG - 1)
                    im[k + i*nG + i][l + j*nG + j] = evalUnit(unit, K, L)

    # Save to files
    im = scipy.misc.toimage(im, cmin=0.0, cmax=1.0)
    im.save('plots/img-%09d.png' % fileNum)

def clip(x, min=0.0, max=1.0):
    if x < min: return min
    if x > max: return max
    return x

# bin
def binPair(res, x):
    bl = int(x * (res - 1))
    bl = clip(bl, 0, res - 2)
    bu = bl + 1
    return bl, bu

# bins
def binPairs(res, x, y):
    return binPair(res, x), binPair(res, y)

def binDistPair(res, x):
    ''' Assumes 0.0 <= x <= 1.0 '''
    bl, bu = binPair(res, x)

    sl = bl / float(res - 1) # Snapped upper
    su = bu / float(res - 1) # Snapped lower

    return bl, bu, x - sl, su - x

def binDistPairs(res, x, y):
    return binDistPair(res, x), binDistPair(res, y)

def evalUnit(unit, x, y):
    res = len(unit)
    (xbl, xbu, xdl, xdu), (ybl, ybu, ydl, ydu) = binDistPairs(res, x, y)

    return (res - 1) * (res - 1) * (
            unit[xbl][ybl] * xdu * ydu + \
            unit[xbu][ybl] * xdl * ydu + \
            unit[xbl][ybu] * xdu * ydl + \
            unit[xbu][ybu] * xdl * ydl   \
    )

################################################################################
for i in range(0, 10000):
    execfile('plots/net-%09d.net' % i)
    plot(net, i)
