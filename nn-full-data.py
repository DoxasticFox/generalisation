import math
import numpy as np
import itertools
import scipy.misc
import scipy.ndimage.filters
import scipy.special
import random
import copy
import pickle

saveDir = './tmp-100/'
fileNum = 0

def checkpoint(net):
    plot(net)
    save(net)

def plot(net):
    global saveDir
    global fileNum

    nG = len(net[0][0]) * 2 # Plot width
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
                    im[k + i*nG + i][l + j*nG + j] = evalUnit(K, L, unit)

    # Save to files
    im = scipy.misc.toimage(im, cmin=0.0, cmax=1.0)
    im.save('%simg-%06d.png' % (saveDir, fileNum))

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

def bin(res, x):
    bx = res * x
    bx = max(0,   bx)
    bx = min(res-1, bx)
    bx = int(bx)
    return bx

def bins(x, y, res):
    return bin(res, x), bin(res, y)

def evalUnit(x, y, unit):
    res    = len(unit)
    bx, by = bins(x, y, res)
    return unit[bx][by]

def multiFun(Xs, net):
    activated     = []
    addActivation = activated.append

    tmp       = Xs[:]
    for     i in range(len(net   )):
        for j in range(len(net[i])):
            res = len(net[i][j])
            a = (i, j) + bins(tmp[j*2+0], tmp[j*2+1], res)
            addActivation(a)

            tmp[j] = evalUnit(tmp   [j*2+0],
                              tmp   [j*2+1],
                              net[i][j    ])

    return tmp[0], activated

def dxUnit(x, y, unit):
    res    = len(unit)
    bx, by = bins(x, y, res)

    if   bin(res, x) == 0:               return 2.0 * res * (unit[bx+1][by] - unit[bx  ][by])
    elif bin(res, x) == res - 1:         return 2.0 * res * (unit[bx  ][by] - unit[bx-1][by])
    elif x % (1.0/res) <= 1.0/(2 * res): return 2.0 * res * (unit[bx  ][by] - unit[bx-1][by])
    else:                                return 2.0 * res * (unit[bx+1][by] - unit[bx  ][by])

def dyUnit(x, y, unit):
    res    = len(unit)
    bx, by = bins(x, y, res)

    if   bin(res, y) == 0:               return 2.0 * res * (unit[bx][by+1] - unit[bx][by  ])
    elif bin(res, y) == res - 1:         return 2.0 * res * (unit[bx][by  ] - unit[bx][by-1])
    elif y % (1.0/res) <= 1.0/(2 * res): return 2.0 * res * (unit[bx][by  ] - unit[bx][by-1])
    else:                                return 2.0 * res * (unit[bx][by+1] - unit[bx][by  ])

def dwUnit(x, y, unit):
    return 1.0

def dxErr(x, t):
    return (x - t) / 2.0

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

            Zs[i+1][j/2] = evalUnit(x, y, unit)

    return Zs

def dxUnits(net, Zs):
    return dUnits(net, Zs, dxUnit)

def dyUnits(net, Zs):
    return dUnits(net, Zs, dyUnit)

def dwUnits(net, Zs):
    return None
    width = len(Zs[0]) / 2
    depth = len(Zs   ) - 1

    Ds  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    for     i in range(len(Zs     ) - 1):
        for j in range(0, len(Zs[i]), 2):
            x    = Zs [i][j]
            y    = Zs [i][j+1]
            unit = net[i][j/2]

            Ds[i][j/2] = dwUnit(x, y, unit)

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

            Ds[i][j/2] = dUnit(x, y, unit)

    return Ds

def backward(Zs, DXs, DYs, DWs, t):
    width = len(Zs[0]) / 2
    depth = len(Zs   ) - 1

    output = Zs[-1][-1]
    dErr   = dxErr(output, t)

    Ds  = [[0.0 for j in range(width / 2**i)] for i in range(depth)]

    Ds[-1][-1] = dErr
    # Backward pass: Multiply the DXs and DYs appropriately
    for     i in range(len(Ds) - 2, -1,         -1):
        for j in range(0,           len(Ds[i]),  2):
            Ds[i][j  ] = Ds[i+1][j/2] * DXs[i][j/2]
            Ds[i][j+1] = Ds[i+1][j/2] * DYs[i][j/2]

    # Multiply by DWs element-wise
    #for     i in range(len(Ds   )):
        #for j in range(len(Ds[i])):
            #Ds[i][j] *= DWs[i][j]

    return Ds

def activationLocations(net, Zs):
    res = len(net[0][0])
    As  = [[(i, j/2) + bins(Zs[i][j], Zs[i][j+1], res) for j in range(0, len(Zs[i]),  2)] \
                                                       for i in range(0, len(Zs   )-1  )]
    return As

def zipDs(Ds, As):
    # Assume Ds and As have the same dimensions
    return [(Ds[i][j], As[i][j]) for i in range(len(Ds   )) \
                                 for j in range(len(Ds[i]))]

def makeUnit(res, c=None):
    lo = +0.0
    hi = +1.0
    unit = [[None for i in range(res)] for j in range(res)]
    for i in range(len(unit)):
        for j in range(len(unit)):
            if   c == None:
                unit[i][j] = (i+j)/(2.0*(res-1))
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
        else:
            layer = [makeUnit(res, 0.5) for i in range(dim)]
        net.append(layer)

        print 'layer', layerNum, 'resolution:', res, 'x', res

        layerNum += 1

    return net

def blurHorz(net, i, j, k, l):
    unit  = net[i][j]
    #kMax = len(unit   ) - 1
    lMax  = len(unit[0]) - 1

    #if i != 0: return unit[k][l]

    if 0 <  l  < lMax:
        return (unit[k  ][l-1] + unit[k  ][l  ] + unit[k  ][l+1])/3.0
    if 0 == l:
        return (                 unit[k  ][l  ] + unit[k  ][l+1])/2.0
    if      l == lMax:
        return (unit[k  ][l-1] + unit[k  ][l  ]                 )/2.0

def blurVert(net, i, j, k, l):
    unit  = net[i][j]
    kMax  = len(unit   ) - 1
    #lMax = len(unit[0]) - 1

    #if i != 0: return unit[k][l]

    if 0 <  k  < kMax:
        return (unit[k-1][l  ] + unit[k  ][l  ] + unit[k+1][l  ])/3.0
    if 0 == k:
        return (                 unit[k  ][l  ] + unit[k+1][l  ])/2.0
    if      k == kMax:
        return (unit[k-1][l  ] + unit[k  ][l  ]                 )/2.0

def applyBlur(net):
    horz = [[[[blurHorz(net,  i, j, k, l) for l in range(len(net [i][j][k]))]
                                          for k in range(len(net [i][j]   ))]
                                          for j in range(len(net [i]      ))]
                                          for i in range(len(net          ))]

    vert = [[[[blurVert(horz, i, j, k, l) for l in range(len(horz[i][j][k]))]
                                          for k in range(len(horz[i][j]   ))]
                                          for j in range(len(horz[i]      ))]
                                          for i in range(len(horz         ))]
    return vert

def selectiveBlur(net, exclude, repeat=3):
    if repeat == 0:
        return net

    blurred = applyBlur(net)

    # BLUR STRENGTH. A == 1.0 -> FULL STRENGTH A == 0.0 -> NO BLUR
    #a = 1.0 / 4096.0
    #for i, layer in enumerate(net):
        #for j, unit in enumerate(layer):
            #for k, pr in enumerate(unit):
                #for l, pc in enumerate(pr):
                    #blurred[i][j][k][l] = \
                    #blurred[i][j][k][l] * (a)  + \
                        #net[i][j][k][l] * (1.0 - a)

    for i, j, k, l in exclude:
        blurred[i][j][k][l] = \
            net[i][j][k][l]

    return selectiveBlur(blurred, exclude, repeat-1)

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
        ybar = multiFun(x, net)[0]
        out += (ybar - Ts[i])**2
    return out

def normaliseGradient(gradientVector):
    return [sgn(g) for g in gradientVector]

def sgn(x):
    if x > 0.0:
        return + 1.0
    if x < 0.0:
        return - 1.0
    else:
        return   0.0

def abs(x):
    return math.fabs(x)

def numOnes(bits):
    n = 0
    for b in bits:
        if b > 0.5:
            n += 1
    return n

def oddParity(bits):
    if numOnes(bits) % 2 == 0: return 0.0
    else:                      return 1.0

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
    rate           = 1.0 / res

    zipped       = zip(Xs, Ts)
    randomChoice = random.choice
    #batchSize    = res * res / 128
    batchSize    = res * res

    global fileNum
    print 'batch size:', batchSize

    while True:
        # Make a batch
        #if fileNum <= res * 3:
            #batch = [] # Warm it up!
        #else:
            #batch = [randomChoice(zipped) for i in range(batchSize)]
        batch = [randomChoice(zipped) for i in range(batchSize)]

        if fileNum % 10 == 0:
            checkpoint(net)
            print 'plot', fileNum
        if fileNum % 80 == 0:
            sample = random.sample(zipped, 500)
            Xs_, Ts_ = zip(*sample)
            print 'mse:', obj(net, Xs_, Ts_)

        # Take a step for each sample in batch
        for X, T in batch:
            GAs = backprop(net, X, T)

            grads, acts = zip(*GAs)
            grads = normaliseGradient(grads)
            GAs   = zip(grads, acts)

            for ga in GAs:
                grad, act    = ga
                i, j, pi, pj = act

                step = - rate * grad

                net[i][j][pi][pj] += step
                net[i][j][pi][pj]  = clip(net[i][j][pi][pj])

        activations = []
        for X, T in batch:
            _, acts = multiFun(X, net)
            activations += acts

        # Selectively blur based on activations after batch update
        net = selectiveBlur(net, exclude=activations)
        normaliseNet(net)

        if limit is not None and fileNum >= limit:
            break

        fileNum += 1


################################################################################

# Make some data
trainingSize   = 100000
sequenceLength = 8
res            = 36

#m  = loadMnist()
#Xs = m[0]
#Ys = [float(m[1][i] == 0) for i in range(len(m[1]))]

#Xs = [[int(np.random.uniform() + 0.5) for i in range(sequenceLength)] for j in range(trainingSize)]
Xs = [[np.random.uniform() for i in range(sequenceLength)] for j in range(trainingSize)]
Ys = [oddParity(x) for x in Xs]

# Fit
bl = makeNet(sequenceLength, res)

search(bl, Xs, Ys, limit= 500)

bl = load( 500); bl[0][0] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=1000)
bl = load(1000); bl[0][1] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=1500)
bl = load(1500); bl[0][2] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=2000)
bl = load(2000); bl[0][3] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=2500)

bl = load(2500); bl[1][0] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=3000)
bl = load(3000); bl[1][1] = makeUnit(res, 0.5); search(bl, Xs, Ys, limit=3500)
