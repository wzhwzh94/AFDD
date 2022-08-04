import numpy as np
import math
from numpy.ma import sin
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt


def pickpeaks(V=None, select=None, display=None, *args, **kwargs):
    # -------------------------------------------------------------
    # Scale-space peak picking
    # ------------------------
    # This function looks for peaks in the data using scale-space theory.
    #
    # input :
    #   * V : data, a vector
    #   * select : either:
    #       - select >1 : the number of peaks to detect
    #       - 0<select<1 : the threshold to apply for finding peaks
    #         the closer to 1, the less peaks, the closer to 0, the more peaks
    #   * display : whether or not to display a figure for the results. 0 by
    #               default
    #   * ... and that's all ! that's the cool thing about the algorithm =)
    #
    # outputs :
    #   * peaks : indices of the peaks
    #   * criterion : the value of the computed criterion. Same
    #                 length as V and giving for each point a high value if
    #                 this point is likely to be a peak
    #
    # The algorithm goes as follows:
    # 1°) set a smoothing horizon, initially 1;
    # 2°) smooth the data using this horizon
    # 3°) find local extrema of this smoothed data
    # 4°) for each of these local extrema, link it to a local extremum found in
    #     the last iteration. (initially just keep them all) and increment the
    #     corresponding criterion using current scale. The
    #     rationale is that a trajectory surviving such smoothing is an important
    #     peak
    # 5°) Iterate to step 2°) using a larger horizon.

    # data is a vector
    V = V - min(V)

    n = len(V)
    # definition of local variables
    buffer = np.zeros((n, 1))
    criterion = np.zeros((n, 1))
    if select < 1:
        minDist = n / 20
    else:
        minDist = n / select

    horizons = np.unique(np.round(np.logspace(0, 2, 50) / 100 * math.ceil(n / 20)))

    Vorig = V
    # all this tempMat stuff is to avoid calling findpeaks which is horribly
    # slow for our purpose
    tempMat = np.zeros((n, 3))
    tempMat[0, 0] = float("inf")
    tempMat[len(tempMat) - 1, 2] = float("inf")
    # loop over scales
    for is_ in range(0, len(horizons)):
        # sooth data, using fft-based convolution with a half sinusoid
        horizon = horizons[is_]
        if horizon > 1:
            w = sin(2 * math.pi * np.arange(0, horizon) / 2 / (horizon - 1))
            w = w / sum(w)
            V = ifft(fft(V, int(n + horizon)) * fft(w, int(n + horizon))).real
            V = V[np.arange(math.floor(horizon / 2), len(V) - math.ceil(horizon / 2))]
        # find local maxima
        tempMat[np.arange(1, len(tempMat)), 0] = V[np.arange(0, len(V) - 1)]
        tempMat[:, 1] = V
        tempMat[np.arange(0, len(tempMat) - 1), 2] = V[np.arange(1, len(V))]
        posMax = tempMat.argmax(1)
        I = np.array(np.where(posMax == 1))
        newBuffer = np.zeros((len(buffer), 1))

        if is_ == 0:
            # if first iteration, keep all local maxima
            newBuffer[I, 0] = Vorig[I]
        else:
            old = np.array(np.where(buffer[:, 0]))
            if not old.any():
                continue
            c = old
            c.sort()
            p = np.argsort(old)
            edge = (c[0, np.arange(0, len(c[0]) - 1)] + c[0, np.arange(1, len(c[0]))]) / 2
            ic = np.zeros((len(I[0]), 1), dtype='int')
            for i in range(0, len(I[0])):
                for j in range(0, len(edge)):
                    if I[0, i] < edge[j]:
                        ic[i] = j
                        break
                    if j == len(edge) - 1:
                        ic[i] = j + 1

            iOld = p[0, ic[:, 0]]
            d = abs(I[0, :] - old[0, iOld])
            neighbours = iOld[np.array(np.where(d < minDist))]
            if neighbours.any():
                newBuffer[old[0, neighbours], 0] = V[old[0, neighbours]] * pow(is_, 2)
        # update stuff
        buffer = newBuffer
        criterion = criterion + newBuffer

    # normalize criterion
    criterion = criterion / max(criterion)
    # find peaks based on criterion
    if select < 1:
        peaks = np.where(criterion[:, 0] > select)
    else:
        #     sorted = find(criterion>1E-3);
        #     [~,order] = sort(criterion(sorted),'descend');
        #     peaks = sorted(order(1:min(length(sorted),select)));
        order = np.argsort(criterion[:, 0])
        peaks = order[np.arange(len(order) - select, len(order))]

    if display:
        # display
        plt.figure()
        plt.plot(Vorig, lw=2, label="data")
        plt.plot(criterion * max(Vorig), c='r', label="computed criterion")
        plt.plot(peaks, Vorig[peaks], 'ro', lw=2, label="selected peaks")
        plt.title('Scale-space peak detection')
        plt.legend()
        plt.grid()
        plt.show()

    return peaks, criterion
