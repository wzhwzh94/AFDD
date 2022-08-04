# Study case: Clamped-free beam
# The dynamic response of a 100 m high clamped-free steel beam is studied.
# Simulated time series are used, where the first three eigenmodes have been
# taken into account.
import scipy.io as scio
import numpy as np
import math
import time
import matplotlib.pyplot as plt

import AFDD


def median(data):
    data.sort()
    half = len(data[0]) // 2
    return (data[0][half] + data[0][~half]) / 2


def tic():
    globals()['tt'] = time.process_time()


def toc():
    print('\nElapsed time: %.8f seconds\n' % (time.process_time() - globals()['tt']))


if __name__ == '__main__':
    data = scio.loadmat('data.mat')
    fn = data['wn'] / (2 * math.pi)
    Nmodes = len(fn[0])
    fs = 1 / median(np.diff(data['t']))

    Az=data['Az']
    t=data['t']
    phi=data['phi']

    # Manual procedure
    tic()
    phi_FDD, fn_FDD, zeta = AFDD.AFDD(Az, t, Nmodes)
    toc()

    # we plot the mode shapes
    plt.figure()
    for ii in range(0,len(phi_FDD)):
        plt.subplot(2,2,ii+1)
        plt.plot(np.linspace(0,1,len(phi_FDD[0])),phi_FDD[ii,:],'ro',lw=1.5)
        plt.plot(np.linspace(0,1,len(phi[0])),- phi[ii,:],'k-',lw=1.5)
        plt.xlabel('$y$ (a.u.)')
        plt.ylabel('$phi_'+str(ii)+'$')
    plt.show()

    # The theoretical and measured eigenfrequencies agrees well !
    print('left: target eigen frequencies. Right: Measured eigenfrequencies')
    print([fn,fn_FDD[0:Nmodes].T])
    print('left: target damping. Right: Measured damping')
    print([0.005* np.ones((Nmodes,1)),zeta])

    # Automated procedure 1: minimalist example
    # Minimalist example with automated procedure for those who don't want to
    # read too much

    phi_FDD,fn_FDD,zeta=AFDD.AFDD(Az,t,Nmodes)
    # plot the mode shapes
    plt.figure()
    for ii in range(0,len(phi_FDD)):
        plt.subplot(2,2,ii+1)
        plt.plot(np.linspace(0, 1, len(phi_FDD[0])), phi_FDD[ii, :], 'gd', lw=1.5)
        plt.plot(np.linspace(0, 1, len(phi[0])), - phi[ii, :], 'k-', lw=1.5)
        plt.xlabel('$y$ (a.u.)')
        plt.ylabel('$phi_' + str(ii) + '$')
    plt.show()

    # Comparison between the measured and target eigen freq. and mode shapes
    print('left: target eigen frequencies. Right: Measured eigenfrequencies')
    print([fn, fn_FDD[0:Nmodes].T])
    print('left: target damping. Right: Measured damping')
    print([0.005 * np.ones((Nmodes, 1)), zeta])

    # Automated procedure 2: 2-step analysis

    # First step: determination of the eigenfrequencies
    __,fn_FDD,__=AFDD.AFDD(Az[np.arange(1,len(Az),5),:],t,Nmodes,dataPlot=1)

    # we show that the estimated zeta is ca. 10 x larger for the first 2 modes
    # than expected.
    # the theoritical and measured eigen frequencies agrees however well!
    print('left: target eigen frequencies. Right: Measured eigenfrequencies')
    print([fn, fn_FDD[0:Nmodes].T])

    # Second step: determination of the modal damping ratio
    # We use a high value for M and prescribed eigenfrequencies
    # We use the option 'dataPlot' to plot intermediate figures, to illustrate
    # the method, and to check the accuracy of the results.
    phi_FDD,fn_FDD,zeta=AFDD.AFDD(Az,t,Nmodes,fnTarget=fn_FDD.reshape(1, -1),M=8192)
    # Plot the mode shapes
    plt.figure()
    for ii in range(0, len(phi_FDD)):
        plt.subplot(2, 2, ii + 1)
        plt.plot(np.linspace(0, 1, len(phi_FDD[0])), phi_FDD[ii, :], 'cs', lw=1.5)
        plt.plot(np.linspace(0, 1, len(phi[0])), - phi[ii, :], 'k-', lw=1.5)
        plt.xlabel('$y$ (a.u.)')
        plt.ylabel('$phi_' + str(ii) + '$')
    plt.show()

    print('left: target damping. Right: Measured damping')
    print([0.005 * np.ones((Nmodes, 1)), zeta])

    # Case of user-defined boundaries for the selected peaks.
    # The boundaries for the selected peaks (lines 209 in the main function AFDD)
    # may not be adapted if the eigenfrequency values range from low to high
    # frequencies. For this reason, it is possible to manually give the upper
    # boundaries (UB) and the lower boundaries(LB) as shown below for the first 4
    # eigenfrequencies of the beam studied:

    # lower boundary for the first four modes (Default: LB = 0.9*fn)
    LB=[0.15,0.9,2.8,5.5]
    # upper boundary boundary for the first four modes (Default: UB = 0.9*fn)
    UB=[0.18,1.15,3.1,6.1]

    # Visualization of the boundaries
    plt.figure()
    plt.plot(np.arange(0,4),np.squeeze(fn),'ko-', label="Measured eigen frequency")
    plt.plot(np.arange(0,4),LB,'r-', label="user-defined lower boundary")
    plt.plot(np.arange(0,4),UB,'b-', label="user-defined upper boundary")
    plt.ylabel('$f_n$ (Hz)')
    plt.xlabel('Mode number')
    plt.legend()
    plt.show()

    # Calculation of the modal parameters with user-defined UBs and LBs
    phi_FDD,fn_FDD,zeta=AFDD.AFDD(Az,t,Nmodes,M=8192,UB=np.array(UB).reshape(1, -1),LB=np.array(LB).reshape(1, -1))

    plt.figure()
    for ii in range(0, len(phi_FDD)):
        plt.subplot(2, 2, ii + 1)
        plt.plot(np.linspace(0, 1, len(phi_FDD[0])), phi_FDD[ii, :], 'ms', lw=1.5)
        plt.plot(np.linspace(0, 1, len(phi[0])), - phi[ii, :], 'k-', lw=1.5)
        plt.xlabel('$y$ (a.u.)')
        plt.ylabel('$phi_' + str(ii) + '$')
    plt.show()

    print('left: target damping. Right: Measured damping')
    print([0.005 * np.ones((Nmodes, 1)), zeta])


