import math
import numpy as np
import itertools
import scipy.misc
import scipy.ndimage.filters
import scipy.special
import random
import copy
import pickle

import os
import struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
import Image

saveDir = './tmp-400/'
fileNum = 0

def checkpoint(net):
    plot(net)
    save(net)

def plot(net):
    global saveDir
    global fileNum

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
    im.save('%simg-%09d.png' % (saveDir, fileNum))

def save(net):
    global saveDir
    global fileNum

    with open('%snet-%06d.net' % (saveDir, fileNum), 'wb') as netFile:
        netString = pickle.dumps(net)
        netFile.write(netString)

def load(index):
    global saveDir
    global fileNum

    global res
    global sequenceLength

    fileNum = index
    with open('%snet-%06d.net' % (saveDir, fileNum), 'rb') as netFile:
        netString = netFile.read()
        net       = pickle.loads(netString)

        res            = len(net[0][0])
        sequenceLength = len(net[0]   ) * 2
        fileNum       += 1

        return net

def loadMnist(dataset="training"):
    if dataset == "training":
        fname_img = 'mnist/train-images.idx3-ubyte'
        fname_lbl = 'mnist/train-labels.idx1-ubyte'
    elif dataset == "testing":
        fname_img = 'mnist/t10k-images.idx3-ubyte'
        fname_lbl = 'mnist/t10k-labels.idx1-ubyte'
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, numRows, numCols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    images = [None for _ in range(size)]
    labels = [None for _ in range(size)]
    for i in range(size):
        szImg     = numRows * numCols

        image     = list(img[szImg*i : szImg*(i+1)])
        image     = [float(p) for p in image]
        image     = [p/255.0  for p in image]
        image    += [0.0 for _ in range(1024 - szImg)]
        images[i] = image * 2

        labels[i] = lbl[i]

    #for I in range(10):
        #for i in range(numRows):
            #for j in range(numCols):
                #print ('%1.f' % images[I][i*numCols + j]),
            #print ''
        #print labels[I], labels[I] == 9

    return images, labels

def loadMnist(dataset="training", path="."):
    if dataset == "training":
        fname_img = os.path.join(path, 'mnist/train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist/train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'mnist/t10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist/t10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    images = []
    labels = []
    for i in range(size):
        resMod = 256
        resImg = 16

        image     = array(img[ i*rows*cols : (i+1)*rows*cols ]).reshape((rows, cols))
        image     = scipy.misc.toimage(image, cmin=0.0, cmax=255.0)

        image     = image.crop((2, 2, 28-2, 28-2))
        image     = image.resize((resImg, resImg), Image.ANTIALIAS)

        #image     = image.crop((2, 2, 28-2, 28-2))
        #image     = image.resize((resImg, resImg), Image.NEAREST)

        image     = np.array(image)
        image     = image/255.0
        image     = np.reshape(image, resImg*resImg)
        image     = np.lib.pad(image, (0, resMod - resImg*resImg), 'constant', constant_values=(0, 0))
        image     = image.tolist()
        #image     = image * 2

        label     = lbl[i]



        if dataset == 'training' and label == 0 and False:
            for _ in range(9):
                images.append(image)
                labels.append(label)
        else:
            images.append(image)
            labels.append(label)

    #for I in range(10):
        #for i in range(8):
            #for j in range(8):
                #print ('%1.1f' % images[I][i*8 + j]),
            #print ''
        #print labels[I], labels[I] == 9

    return images, labels

def loadTestingMnist():
    m = loadMnist(dataset='testing')

    Xs = m[0]
    Ys = [float(m[1][i] == 0) for i in range(len(m[1]))]

    #Xs = Xs[:1000]
    #Ys = Ys[:1000]

    return zip(Xs, Ys)

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

def multiFun(net, Xs):
    tmp       = Xs[:]
    for     i in range(len(net   )):
        for j in range(len(net[i])):
            res = len(net[i][j])
            tmp[j] = evalUnit(
                    net[i][j    ],
                    tmp   [j*2+0],
                    tmp   [j*2+1]
            )
    return tmp[0]

def dxUnit(unit, x, y):
    res    = len(unit)
    (xbl, xbu, xdl, xdu), (ybl, ybu, ydl, ydu) = binDistPairs(res, x, y)

    return (res - 1) * (res - 1) * (
            ydu * (unit[xbu][ybl] - unit[xbl][ybl]) + \
            ydl * (unit[xbu][ybu] - unit[xbl][ybu])   \
    )

def dyUnit(unit, x, y):
    res    = len(unit)
    (xbl, xbu, xdl, xdu), (ybl, ybu, ydl, ydu) = binDistPairs(res, x, y)

    return (res - 1) * (res - 1) * (
            xdu * (unit[xbl][ybu] - unit[xbl][ybl]) + \
            xdl * (unit[xbu][ybu] - unit[xbu][ybl])   \
    )

def dwUnit(unit, x, y):
    (xbl, xbu, xdl, xdu), (ybl, ybu, ydl, ydu) = binDistPairs(res, x, y)

    norm = (res - 1) * (res - 1)

    wll = norm * xdu * ydu
    wlu = norm * xdu * ydl
    wul = norm * xdl * ydu
    wuu = norm * xdl * ydl

    return wll, wlu, wul, wuu

def dxErr(x, t):
    return x - t

def backprop(net, Xs, t):
    Zs  = forward (net, Xs);              # print 'Zs ', Zs
    DXs = dxUnits (net, Zs);              # print 'DXs', DXs
    DYs = dyUnits (net, Zs);              # print 'DYs', DYs
    DWs = dwUnits (net, Zs);              # print 'DWs', DWs
    Ds  = backward(Zs, DXs, DYs, DWs, t); # print 'Ds ', Ds
    As  = activationLocations(net, Zs)
    return zipDs(Ds, As)

def forward(net, Xs):
    width = len(net[0]) * 2

    depth = math.log(width, 2) + 1
    depth = int(depth)

    Zs  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    for i, x in enumerate(Xs):
        Zs[0][i] = x

    for     i in range(len(Zs     ) - 1):
        for j in range(0, len(Zs[i]), 2):
            x    = Zs [i][j]
            y    = Zs [i][j+1]
            unit = net[i][j/2]

            Zs[i+1][j/2] = evalUnit(unit, x, y)

    return Zs

def dxUnits(net, Zs):
    return dUnits(net, Zs, dxUnit)

def dyUnits(net, Zs):
    return dUnits(net, Zs, dyUnit)

def dwUnits(net, Zs):
    width = len(Zs[0]) / 2
    depth = len(Zs   ) - 1

    Ds  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    for     i in range(len(Zs     ) - 1):
        for j in range(0, len(Zs[i]), 2):
            x    = Zs [i][j]
            y    = Zs [i][j+1]
            unit = net[i][j/2]

            Ds[i][j/2] = dwUnit(unit, x, y)

    return Ds

def dUnits(net, Zs, dUnit):
    width = len(Zs[0]) / 2 / 2
    depth = len(Zs   ) - 1 - 1

    Ds  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    for     i in range(len(Zs     ) - 2):
        for j in range(0, len(Zs[i+1]), 2):
            x    = Zs [i+1][j]
            y    = Zs [i+1][j+1]
            unit = net[i+1][j/2]

            Ds[i][j/2] = dUnit(unit, x, y)

    return Ds

def backward(Zs, DXs, DYs, DWs, t):
    width = len(Zs[0]) / 2
    depth = len(Zs   ) - 1

    output = Zs[-1][-1]
    dErr   = dxErr(output, t)

    Ds  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    Ds[-1][-1] = dErr
    #Ds[-1][-1] = 1.0
    # Backward pass: Multiply the DXs and DYs appropriately
    for     i in range(len(Ds) - 2, -1,         -1):
        for j in range(0,           len(Ds[i]),  2):
            Ds[i][j  ] = (Ds[i+1][j/2]) * DXs[i][j/2]
            Ds[i][j+1] = (Ds[i+1][j/2]) * DYs[i][j/2]

    # Multiply by DWs and dErr element-wise
    for     i in range(len(Ds   )):
        for j in range(len(Ds[i])):
            Ds[i][j] = [dw * Ds[i][j] for dw in DWs[i][j]]

    return Ds

def sgn(x):
    if x > 0.0: return + 1.0
    if x < 0.0: return - 1.0
    else:       return   0.0

def activationLocations(net, Zs):
    res = len(net[0][0])

    As  = [[(i, j/2) + binPairs(res, Zs[i][j], Zs[i][j+1]) for j in range(0, len(Zs[i]),  2)] \
                                                           for i in range(0, len(Zs   )-1  )]
    return As

def zipDs(Ds, As):
    # Assume Ds and As have the same dimensions
    return [(Ds[i][j], As[i][j]) for i in range(len(Ds   )) \
                                 for j in range(len(Ds[i]))]

def regGrad(net, i, j, k, l):
    res = len(net[0][0])
    x   = net[i][j][k][l]
    grad = 0.0
    for     k_  in (-1, 0, +1):
        for l_  in (-1, 0, +1):
            K = k_ + k
            L = l_ + l
            if K < 0: continue
            if L < 0: continue
            if K >= res: continue
            if L >= res: continue

            grad += x - net[i][j][K][L]

    return grad

def reguarliserGrad(net):
    return [[[[regGrad(net, i, j, k, l) for l in range(len(net[i][j][k]))]
                                        for k in range(len(net[i][j]   ))]
                                        for j in range(len(net[i]      ))]
                                        for i in range(len(net         ))]

def regulariseNet(net, rate, alpha):
    grad = reguarliserGrad(net)
    return [[[[net[i][j][k][l] - grad[i][j][k][l] * alpha * rate for l in range(len(net[i][j][k]))]
                                                                 for k in range(len(net[i][j]   ))]
                                                                 for j in range(len(net[i]      ))]
                                                                 for i in range(len(net         ))]

def cdfTriangular(x):
    if   0.0 <= x <  0.5: return 2.0 * x * x
    elif 0.5 <= x <= 1.0: return 2.0 * x * x - (2.0 * x - 1.0)**2.0
    else:                 return None

def makeUnit(res, c=None):
    lo = +0.0
    hi = +1.0
    unit = [[None for i in range(res)] for j in range(res)]
    for i in range(len(unit)):
        for j in range(len(unit)):
            if   c == None:
                unit[i][j] = cdfTriangular((i+j)/(2.0*(res-1)))
            elif c == 'n':
                unit[i][j] = np.random.uniform()
            else:
                unit[i][j] = c
    return unit

def makeNet(dim, res):
    assert np.log2(dim) % 1 - 0.001 < 0.0

    net = []
    layerNum = 1
    while dim >= 2:
        dim /= 2
        if dim >= 2:
            layer = [makeUnit(res     ) for i in range(dim)]
            #layer = [makeUnit(res, 'n') for i in range(dim)]
        else:
            layer = [makeUnit(res, 0.5) for i in range(dim)]
        net.append(layer)

        print 'layer', layerNum, 'resolution:', res, 'x', res

        layerNum += 1

    print
    return net

def normaliseNet(net):
    for     i in range(len(net   ) - 1):
        for j in range(len(net[i])):
            normaliseUnit(net[i][j])

def normaliseUnit(unit):
    rowMins = [min(row) for row in unit]; unitMin = min(rowMins)
    rowMaxs = [max(row) for row in unit]; unitMax = max(rowMaxs)
    div     = unitMax - unitMin

    if div == 0.0: return

    for     i in range(len(unit   )):
        for j in range(len(unit[i])):
            unit[i][j] = (unit[i][j] - unitMin) / div

def obj(net, Xs, Ts):
    assert len(Xs) == len(Ts)

    out = 0.0
    for i, x in enumerate(Xs):
        x = Xs[i]
        ybar = multiFun(net, x)
        out += (ybar - Ts[i])**2
    return out / len(Xs)

def numOnes(bits):
    n = 0
    for b in bits:
        if b > 0.5:
            n += 1
    return n

def oddParity(bits):
    #half = len(bits)/2
    #bits = bits[:half]

    #bits = ['1' if x > 0.5 else '0' for x in bits]
    #bits = ''.join(bits)
    #num  = int(bits, 2)
    #return float(num % 15 == 0)

    return int(bits[0] > bits[-1])

    #S = sorted(bits)
    #m = max   (bits)
    #for i, b in enumerate(bits):
        #if b == m:
            #return i / float(len(bits)-1)

    #return sorted(bits)[-2]

    #return np.sin((bits[0] + bits[1]) * 3.14)**2.0

    if numOnes(bits) % 2 == 0: return 0.0
    else:                      return 1.0

def classificationError(net, examples, verbose=False):
    err = 0.0
    for x, t in examples:
        y = multiFun(net, x)
        #print t, y, round(y), int(y+0.5), id(y)
        y = round(y)

        err += float(t != y) / len(examples) * 100.0

        if verbose and t != y:
            for i in range(16):
                for j in range(16):
                    print ('%1.1f' % x[i*16 + j]),
                print ''
            print t, multiFun(net, x)

    return err

# Search #######################################################################

# for each example:
#     for each unit:
#         for each pixel triggered by example:
#             try a different intensity
#             check objective with {example}
#             keep change on improvement of objective

def clip(x, min=0.0, max=1.0):
    if x < min: return min
    if x > max: return max
    return x

def search(net, Xs, Ts, limit=None):
    res            = len(net[0][0])
    rate           = 1.0
    alpha          = 0.00005

    zipped       = zip(Xs, Ts)
    randomChoice = random.choice
    batchSize    = 1000

    global fileNum
    print 'input dim: ', len(Xs[0])
    print 'rate:      ', rate
    print 'alpha:     ', alpha
    print 'batch size:', batchSize
    print

    #testing = loadTestingMnist()
    #print 'error (%):', classificationError(net, zipped, verbose=False)
    #exit()

    while True:
        # Make a batch
        batch = random.sample(zipped, batchSize)
        GAs   = [backprop(net, X, T) for X, T in batch]

        net = regulariseNet(net, rate, alpha)

        # Take a step for each sample in batch
        for GA in GAs:
            for grad, act in GA:
                gradWll, gradWlu, gradWul, gradWuu = grad
                i, j, (xbl, xbu), (ybl, ybu)       = act

                net[i][j][xbl][ybl] -= rate * gradWll / float(batchSize)
                net[i][j][xbl][ybu] -= rate * gradWlu / float(batchSize)
                net[i][j][xbu][ybl] -= rate * gradWul / float(batchSize)
                net[i][j][xbu][ybu] -= rate * gradWuu / float(batchSize)

        for GA in GAs:
            for grad, act in GA:
                i, j, (xbl, xbu), (ybl, ybu)       = act

                net[i][j][xbl][ybl]  = clip(net[i][j][xbl][ybl])
                net[i][j][xbl][ybu]  = clip(net[i][j][xbl][ybu])
                net[i][j][xbu][ybl]  = clip(net[i][j][xbu][ybl])
                net[i][j][xbu][ybu]  = clip(net[i][j][xbu][ybu])

        normaliseNet(net)

        if fileNum % 50 == 0:
            checkpoint(net)
            print 'plot', fileNum
        if fileNum % 500 == 0:
            sample = random.sample(zipped, 5000)
            Xs_, Ts_ = zip(*sample)
            print 'mse:      ', obj(net, Xs_, Ts_)

            print 'error (%):', classificationError(net, testing)

        # Increment loop counter or break out
        if limit is not None and fileNum >= limit:
            break
        fileNum += 1

################################################################################

# Make some data
trainingSize   = 100000
sequenceLength = 256
res            = 5

m = loadMnist()
Xs = m[0]
Ys = [float(m[1][i] == 0) for i in range(len(m[1]))]

# Fit
bl = makeNet(sequenceLength*1, res)
#bl = load(176500)

thing = 1000000
search(bl, Xs, Ys, limit=thing*1)
bl = load(thing*1); bl[0][0] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit= thing*2)
bl = load(thing*2); bl[0][1] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=thing*3)
bl = load(thing*3); bl[1][0] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=thing*4)

search(bl, Xs, Ys, limit= 1e8)
search(bl, Xs, Ys, limit= 500)

bl = load( 500); bl[0][0] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=1000)
bl = load(1000); bl[0][1] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=1500)
bl = load(1500); bl[0][2] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=2000)
bl = load(2000); bl[0][3] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=2500)

bl = load(2500); bl[1][0] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=3000)
bl = load(3000); bl[1][1] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=3500)
