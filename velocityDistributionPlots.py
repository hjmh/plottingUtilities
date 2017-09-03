"""
Functions for generating visualisations of walking velocities.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from basicPlotting import myAxisTheme


def plotVeloTrace(timeVal, veloVal, gsLoc, ylimVal, xlimVal):
    """ Velocity trace plot of raw signal """
    veloTrace = plt.subplot(gsLoc)
    veloTrace.plot(timeVal, veloVal, 'k')
    plt.ylim(ylimVal)
    plt.xlim(xlimVal)
    myAxisTheme(veloTrace)

    return veloTrace


def plotVeloFiltTrace(timeVal, veloVal, veloFiltVal, gsLoc, ylimVal, xlimVal):
    """ Velocity trace plot with raw and filtered signal """
    veloTrace = plt.subplot(gsLoc)
    veloTrace.plot(timeVal, veloVal, 'k')
    veloTrace.plot(timeVal, veloFiltVal)
    plt.ylim(ylimVal)
    plt.xlim(xlimVal)
    myAxisTheme(veloTrace)

    return veloTrace


def plotVeloHistogram(veloVal, gsLoc, histRange):
    """ Velocity histogram plot """
    veloHist = plt.subplot(gsLoc)
    plt.hist(veloVal, bins=50, range=histRange)
    myAxisTheme(veloHist)

    return veloHist


def plotVeloHistogram_fancy(veloVal, gsLoc, histRange, colorVal, alphaVal):
    """ Velocity histogram plot """
    veloHist = plt.subplot(gsLoc)
    try:
        plt.hist(veloVal, bins=50, range=histRange, color=colorVal, alpha=alphaVal)
    except ValueError:
        print('Not enough values for histogram.')
    veloHist.set_xlim(histRange)
    myAxisTheme(veloHist)

    return veloHist


def velocitySummaryPlot(time, vTrans, vTransFilt, vRot, vRotFilt, angle, rotLim, transLim, angleLim, titleString):
    """ Summary figure of walking velocities (heading angle, velocity traces and distribution) """

    summaryFig = plt.figure(figsize=(10, 12))
    summaryFig.set_canvas(plt.gcf().canvas)

    gs = gridspec.GridSpec(5, 3, height_ratios=np.hstack((1, 1, 1, 1, 1)))

    # Velocity traces
    vTransTrace = plotVeloFiltTrace(time, vTrans, vTransFilt, gs[0, :], transLim, (0, time[-1]))
    vTransTrace.set_ylabel('raw and filtered\ntransl. velocity [mm/s]')

    vTransTrace.set_title(titleString)

    vRotTrace = plotVeloFiltTrace(time, vRot, vRotFilt, gs[1, :], rotLim, (0, time[-1]))
    vRotTrace.set_ylabel('raw and filtered\nrot. velocity [rad/s]')

    angleTrace = plotVeloTrace(time, angle, gs[2, :], angleLim, (0, time[-1]))
    angleTrace.set_ylabel('heading angle\n [rad]')
    angleTrace.set_xlabel('time [s]')

    # velocity distributions
    vTransTraceShort = plotVeloFiltTrace(time, vTrans, vTransFilt, gs[3, 0], transLim, (0, 10))
    vTransTraceShort.set_ylabel('transl. velocity [mm/s]')

    transVHist = plotVeloHistogram(vTrans, gs[3, 1], transLim)
    transVHist.set_xlabel('trans. velocity\n [mm/s]')

    transVFiltHist = plotVeloHistogram(vTransFilt, gs[3, 2], transLim)
    transVFiltHist.set_xlabel('filtered trans. velocity\n [mm/s]')

    # Rotational velocity
    vRotTraceShort = plotVeloFiltTrace(time, vRot, vRotFilt, gs[4, 0], rotLim, (0, 10))
    vRotTraceShort.set_ylabel('rot. velocity [rad/s]')

    rotVHist = plotVeloHistogram(vRot, gs[4, 1], rotLim)
    rotVHist.set_xlabel('rot. velocity\n [deg/s]')

    rotVFiltHist = plotVeloHistogram(vRotFilt, gs[4, 2], rotLim)
    rotVFiltHist.set_xlabel('filtered rot. velocity\n [rad/s]')

    summaryFig.tight_layout()

    return summaryFig
