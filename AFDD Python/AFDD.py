import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy import interpolate
from scipy import fftpack
from numpy.ma import cos
from scipy.optimize import curve_fit
import math

import pickpeaks


def nextpow2(n):
    return np.ceil(np.log2(np.abs(n))).astype('long')


def mag2db(x):
    return 20 * np.log10(x)


def median(data):
    data.sort()
    half = len(data[0]) // 2
    return (data[0][half] + data[0][~half]) / 2


def cxcorr(a, v):
    nom = np.linalg.norm(a[:]) * np.linalg.norm(v[:])
    return fftpack.irfft(fftpack.rfft(a) * fftpack.rfft(v[::-1])) / nom


def generate_response_SDOF(freq=None, t=None, S=None, fn=None, LB=None, UB=None, N=None, *args, **kwargs):
    # lower boundary for selected peak
    if np.isnan(LB).all():
        f_lower = fn * 0.9
    else:
        f_lower = LB

    # lower boundary for selected peak
    if np.isnan(UB).all():
        f_upper = fn * 1.1
    else:
        f_upper = UB

    indLB = abs(freq - f_lower).argmin()
    indUB = abs(freq - f_upper).argmin()

    # Time series generation - Monte Carlo simulation
    Nfreq = len(S[np.arange(indLB, indUB)])
    df = median(np.diff(freq).reshape(1, -1))
    w = 2 * math.pi * freq
    A = np.sqrt(2.0 * S[np.arange(indLB, indUB)] * df).reshape(1, -1)
    B = cos(w[np.arange(indLB, indUB)].reshape(1, -1).T * t + 2 * math.pi * np.tile(np.random.rand(Nfreq, 1), [1, N]))
    x = np.dot(A, B)
    return x


def manualPickPeaking(f=None, S=None, Nmodes=None, *args, **kwargs):
    print('Peak selection procedure')
    print('q: Quit')

    fig = plt.figure()
    plt.plot(f, mag2db(S))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('1st Singular values of the PSD matrix (db)')
    plt.grid()
    plt.ylim((min(mag2db(S)), max(mag2db(10 * S))))
    plt.xlim((f[1], f[len(f) - 1]))
    Fp = []
    while True:
        left = input("Input Left Frequency：")
        if left == 'q':
            break
        right = input("Input Right Frequency：")
        if right == 'q':
            break
        left = int(left)
        right = int(right)
        if left >= right:
            continue
        P1 = abs(f - left).argmin()
        P2 = abs(f - right).argmin()
        P3 = S[np.arange(P1, P2)].argmax()
        indPeak = P3 + P1 - 1
        plt.plot(f[indPeak], mag2db(S[indPeak]), 'go')
        Fp.append(indPeak)

    plt.show()
    # Number selected peaks, respectively
    Fp.sort()
    return Fp


def NExT(y=None, dt=None, Ts=None, method=None, *args, **kwargs):
    Ny = len(y)
    N1 = len(y[0])
    if Ny > N1:
        y = y.T
        Ny = len(y)
        N1 = len(y[0])

    # get the maximal segment length fixed by T
    M1 = round(Ts / dt)

    if method == 1:
        IRF = np.zeros((Ny, Ny, M1))
        for pp in np.arange(0, Ny):
            for qq in np.arange(0, Ny):
                y1 = fftpack.fft(y[pp, :])
                y2 = fftpack.fft(y[qq, :])
                h0 = fftpack.ifft(y1 * np.conj(y2))
                IRF[pp, qq, :] = h0[np.arange(0, M1)].real
        # get time vector t associated to the IRF
        t = np.linspace(0, dt * (M1 - 1), M1)
        if Ny == 1:
            IRF = np.squeeze(IRF).T
    else:
        if method == 2:
            IRF = np.zeros((Ny, Ny, M1 + 1))
            for pp in np.arange(0, Ny):
                for qq in np.arange(0, Ny):
                    dummy = cxcorr(y[pp, :], y[qq, :])
                    dummy = dummy[np.arange(round(len(dummy) / 2) - M1, round(len(dummy) / 2) + M1 + 1)]
                    lag = np.arange(- M1, M1)
                    IRF[pp, qq, :] = dummy[len(dummy) - round(len(dummy) / 2): len(dummy)]
            if Ny == 1:
                IRF = np.squeeze(IRF).T
            # get time vector t associated to the IRF
            t = dt * lag[len(lag) - round(len(lag) / 2): len(lag)]

    # normalize the IRF
    if Ny == 1:
        IRF = IRF / IRF[0]

    return IRF, t


def myFun(x, a, b):
    return a * np.exp(- b * x)


def expoFit(y=None, t=None, wn=None, *args, **kwargs):
    # [zeta] = expoFit(y,t,wn) returns the damping ratio calcualted by fiting
    # an exponential decay to the envelop of the Impulse Response Function.

    # y: envelop of the IRF: vector of size [1 x N]
    # t: time vector [ 1 x N]
    # wn: target eigen frequencies (rad/Hz) :  [1 x 1]
    # zeta: modal damping ratio:  [1 x 1]
    #  optionPlot: 1 to plot the fitted function, and 0 not to plot it.

    coeff, cov = curve_fit(myFun, t, y, [1, 0.01])
    # modal damping ratio:
    zeta = abs(coeff[1] / wn)
    return zeta


def AFDD(Az=None, t=None, Nmodes=None, M=None, fnTarget=np.array([]).reshape(1, -1),
         PickingMethod='auto', ModeNormalization=1, dataPlot=0, Ts=30,
         LB=np.array([]).reshape(1, -1), UB=np.array([]).reshape(1, -1)):
    # [phi,fn,zeta] = AFDD(Az,t,Nmodes,varargin) calculate the mode shapes,
    # eigen frequencies and modal damping ratio of the acceleration data using
    # the Automated Frequency Domain Decomposition (AFDD) method
    # which is based on the Frequency Domain Decomposition (FDD) [1,2]
    #
    # Input
    #  * Az: acceleration data. Matrix of size [Nyy x N] where Nyy
    # is the number of sensors, and N is the number of time steps
    #  * fs: sampling frequencies
    #  * fn: Vecteur "target eigen frequencies". ex: fn = [f1,f2,f3]
    # Optional inputs as inputParser:
    #  * M: [1 x 1 ]  integer.  number of FFT points
    #  * PickingMethod: automated or manual peak picking ('auto' or 'manual')
    #  * fn [1 x Nmodes]:  prescribed or not prescribed eigen frequencies (empty or scalar/vector)
    #  * ModeNormalization: [1 x 1 ]: 1 or 0: option for mode normalization
    #  * dataPlot: 1 or 0: option for intermediate data plot (e.g. checking procedure)
    #  * Ts: [1 x 1]: float: option for duration of autocorrelation function (for estimation of damping ratio only)
    #  * LL: [1 x Nmodes]: float: option for selectin the lower boundary for cut-off frequency
    #  * UL: [1 x Nmodes]: float: option for selectin the uper boundaty for cut-off frequency
    #
    # Output
    # phi: matrix of the measured mode shapes. Size(phi)=  [Nyy x numel(fn)]
    # fn: matrix of the measured eigen frequencies. Size(phi)=  [Nyy x numel(fn)]
    # phi: matrix of the measured mode shapes. Size(phi)=  [Nyy x numel(fn)]

    # Check error and unexpected inputs
    # check if M is empty
    if not M:
        M = pow(2, (nextpow2(len(t[0]) / 8)))

    # Check picking method:
    if (PickingMethod != 'auto') and (PickingMethod != 'manual'):
        print("PickingMethod is not recognized. It must be a either 'auto' or 'manual'")

    # Check if len(fn) is different from Nmodes
    if fnTarget.any() and len(fnTarget[0]) != Nmodes:
        print('The number of eigen frequencies specified fn is different from Nmodes. They must be identical')

    # Check if dataPLot is different from 0 or 1
    if dataPlot != 1 and dataPlot != 0:
        print("The value of 'dataPlot' is not recognized. it must be 0 or 1 ")

    if not (LB.any()):
        LB = np.full((1, Nmodes), np.nan)
    else:
        if len(LB[0]) < Nmodes:
            print('numel(LB) ~= Nmodes')
        else:
            if len(LB) > 1:
                print('LB should be a vector, not a matrix')

    if not (UB.any()):
        UB = np.full((1, Nmodes), np.nan)
    else:
        if len(UB[0]) < Nmodes:
            print('numel(UB) ~= Nmodes')
        else:
            if len(UB) > 1:
                print('UB should be a vector, not a matrix')

    # Pre-processing
    Nyy = len(Az)
    N = len(Az[0])
    fs = 1 / median(np.diff(t))
    z = np.linspace(0, 1, Nyy)

    # Computation of the spectral matrix G
    #  size(G) is [N x Nyy x Nyy]
    if M % 2:
        G = np.zeros((Nyy, Nyy, round(M / 2)), dtype=np.complex)
        U = np.zeros((Nyy, Nyy, round(M / 2)), dtype=np.complex)
        V = np.zeros((Nyy, Nyy, round(M / 2)), dtype=np.complex)
    else:
        G = np.zeros((Nyy, Nyy, round(M / 2) + 1), dtype=np.complex)
        U = np.zeros((Nyy, Nyy, round(M / 2) + 1), dtype=np.complex)
        V = np.zeros((Nyy, Nyy, round(M / 2) + 1), dtype=np.complex)

    for ii in range(0, Nyy):
        for jj in range(0, Nyy):
            G[ii, jj], f = mlab.csd(Az[ii], Az[jj], NFFT=M, Fs=fs, detrend=None, window=None,
                                    noverlap=round(M / 2), pad_to=None, sides=None, scale_by_freq=True)

    # Application of SVD to G
    S = np.zeros((Nyy, len(G[0][0])))

    for ii in range(0, len(G[0][0])):
        U[:, :, ii], S[:, ii], V[:, :, ii] = np.linalg.svd(G[:, :, ii])

    if dataPlot == 1:
        plt.figure()
        x = np.log10(f+1e-16)
        y = np.log10(S+1e-16)
        for i in range(0, Nyy):
            plt.plot(x, y[i, :])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Singular values of the PSD matrix')
        plt.show()

    # interpolation to improve accuracy of peak picking and damping estimation
    Ninterp = 3
    newF = np.linspace(f[0], f[len(f) - 1], Ninterp * len(f))
    interpfunc = interpolate.interp1d(f, S[0, :])
    newS = interpfunc(newF)
    newS = newS / max(newS)
    interpfunc2 = interpolate.interp2d(z, f, np.squeeze(U[:, 0, :].real).T)
    newU = interpfunc2(z, newF).T

    # Peak picking algorithm
    if not fnTarget.any():
        if PickingMethod == 'auto':
            indMax, criterion = pickpeaks.pickpeaks(newS, Nmodes, 0)
            if dataPlot == 1:
                plt.figure()
                x = np.log10(newF+1e-16)
                y = np.log10(newS+1e-16)
                x1 = np.log10(newF[indMax]+1e-16)
                y1 = np.log10(newS[indMax]+1e-16)
                plt.plot(x, y, 'k-', label="1st Singular values")
                plt.plot(x1, y1, 'ko', c='r', label="peaks")
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Normalized PSD ')
                plt.show()
        else:
            indMax = manualPickPeaking(newF, newS, Nmodes)
        # Eigen-frequencies
        fn = np.array(newF[indMax]).reshape(1, -1)
        phi = np.zeros((len(indMax), Nyy))
        for ii in range(0, len(indMax)):
            phi[ii, :] = newU[:, indMax[ii]].real
    else:
        fn = fnTarget
        phi = np.zeros((Nmodes, Nyy))
        for ii in range(0, Nmodes):
            indF = abs(newF - fn[0,ii]).argmin(0)
            phi[ii, :] = newU[:, indF].real

    # normalisation of the modes
    if ModeNormalization == 1:
        for ii in range(0, len(phi)):
            phi[ii, :] = phi[ii, :] / max(abs(phi[ii, :]))

    # Get damping ratio
    zeta = np.zeros((1, len(fn[0])))
    # sort eigen frequencies
    indSort = np.argsort(fn[0])
    fn[0].sort()
    phi = phi[indSort, :]
    for ii in range(0, len(fn[0])):
        x = generate_response_SDOF(newF, t, newS, fn[0,ii], LB[0, ii], UB[0, ii], N)
        # We want segments of 30 seconds for the autocorrelation function
        method = 1  # cross-covariance calculated with ifft
        IRF, newT = NExT(x, median(np.diff(t, 1, -1)), Ts, method)
        # get the envelop of the curve with the hilbert transform:
        envelop = abs(fftpack.hilbert(IRF))
        envelop[0] = IRF[0]
        wn = 2 * math.pi * fn[0,ii]  # -> obtained with peak picking method (fast way)
        zeta[0, ii] = expoFit(envelop, newT, wn)

    return phi, fn, zeta
