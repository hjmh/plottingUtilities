"""
General plotting functions used to visualise tracking data (e.g. collected in a free walking arena)
Some plots assume single fly data, others are for Ctrax multi-fly tracking data.
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import seaborn as sns

sns.set_style('ticks')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
from matplotlib import gridspec


def plotBodyAngle(ax, x, y, angle, markerColor, alphaVal, arrowScale):
    try:
        newArrow = patches.FancyArrow(x, y, np.cos(angle).squeeze()*arrowScale, np.sin(angle).squeeze()*arrowScale,
                                      width=0.25, head_width=0.25,
                                      edgecolor=markerColor, facecolor=markerColor, alpha=alphaVal)
        ax.add_patch(newArrow)
    except:
        print("could not draw arrow")


def plotBodyVector(ax, x, y, vector, markerColor, alphaVal, arrowScale):
    try:
        newArrow = patches.Arrow(x, y, vector[0]*arrowScale, vector[1]*arrowScale, width=2,
                                 edgecolor=markerColor, facecolor=markerColor, alpha=alphaVal)
        ax.add_patch(newArrow)
    except:
        print("could not draw arrow")


def plotPosInRange(ax1, ax2, frameRange, time, xPos, yPos, angle, currCmap, arrowScale, alphaValue, markerSize):

    cNorm  = colors.Normalize(vmin=-0.5*len(frameRange), vmax=1*len(frameRange))
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=currCmap)

    for ind, frame in enumerate(frameRange):
        currCol=scalarMap.to_rgba(ind)
        if alphaValue < 0:
            alphaVal = 1.0-1.0*ind/len(frameRange)
        else:
            alphaVal = alphaValue

        ax1.plot(xPos[frame], yPos[frame], marker='.', markersize=markerSize, linestyle='none', alpha=alphaVal,
                 color=currCol)
        plotBodyAngle(ax1, xPos[frame], yPos[frame], angle[frame], currCol, alphaVal, arrowScale)

        ax2.plot(time[frame], 1, marker='.', markersize=10, linestyle='none',
                 alpha=alphaVal, color=currCol)

    ax1.set_aspect('equal')
    ax2.set_ylim([0.9, 1.1])
    sns.despine(right=True, offset=0)
    sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': '.8'})


def plotPosInRangeMultifly(ax, frameRange, xPos, yPos, angle, flyID, fly, currCmap, imagePx):
    # Plot trajectories of multi-fly tracking data (obtained with Ctrax)
    cNorm  = colors.Normalize(vmin=-0.5*len(frameRange), vmax=1*len(frameRange))
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=currCmap)

    for ind, frame in enumerate(frameRange):
        currCol = scalarMap.to_rgba(len(frameRange) - ind)

        if fly == -1:
            # iterate over all available flies
            ax.plot(xPos[frame], yPos[frame], marker='.', markersize=10, linestyle='none', alpha=0.5,
                    color=currCol)

            for fly in flyID[frame]:
                plotBodyAngle(ax, xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                              angle[frame][flyID[frame] == fly], currCol, 0.5, 20)

        else:
            # just plot trace of specified fly
            ax.plot(xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                    marker='.', markersize=6, linestyle='none', alpha=0.5, color=currCol)

            plotBodyAngle(ax, xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                          angle[frame][flyID[frame] == fly], currCol, 0.5, 20)

    ax.set_aspect('equal')
    sns.despine(right=True, offset=0)  # trim=True)
    sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': '.8', })

    plt.xlim([imagePx[0] / 2.0 - 500, imagePx[0] / 2.0 + 500])
    plt.ylim([imagePx[1] / 2.0 - 500, imagePx[1] / 2.0 + 500])


def plotActiveSegments(activeFragments, frameRange, subplt, titleString, xPos, yPos, angle, flyID, imageSizePx):
    # Plot ctrax trajectory fragments with colour-coded identity and alpha-coded time (multi-fly Ctrax data)

    if len(activeFragments) == 0:
        print('No active fly fragments')
        return

    flyRange = range(min(activeFragments), max(activeFragments) + 1)

    cNorm = colors.Normalize(vmin=min(activeFragments), vmax=max(activeFragments + 1))
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap='Accent')

    arrowScale = 30

    for currFly in flyRange:
        counter = 0.0
        for frame in frameRange:
            currX = xPos[frame][flyID[frame] == currFly]
            currY = yPos[frame][flyID[frame] == currFly]

            try:
                newArrow = patches.Arrow(currX.squeeze(), currY.squeeze(),
                                         np.cos(angle[frame][flyID[frame] == currFly]).squeeze() * arrowScale,
                                         np.sin(angle[frame][flyID[frame] == currFly]).squeeze() * arrowScale, width=2,
                                         edgecolor=scalarMap.to_rgba(currFly), alpha=0.5)
                subplt.add_patch(newArrow)
            except:
                continue

            subplt.plot(currX, currY, marker='.', color=scalarMap.to_rgba(currFly), markersize=8,
                        alpha=1 - counter / (len(frameRange)))

            counter += 1.0
    subplt.set_aspect('equal')
    subplt.set_title(titleString)

    plt.xlim([imageSizePx[0] / 2 - 500, imageSizePx[0] / 2 + 500])
    plt.ylim([imageSizePx[1] / 2 - 500, imageSizePx[1] / 2 + 500])

    return


def plotFlyVRtimeStp(plotStp, FOtime, titleString):
    tstpfig = plt.figure(figsize=(10, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=np.hstack((2, 1)))
    tstpfig.suptitle(titleString, fontsize=14)
    histRange = (0, 0.011)

    ax = tstpfig.add_subplot(gs[0])

    FOtimeStp = np.hstack((0, (FOtime[1:-1]-FOtime[0:-2]))).astype('float')

    ax.plot(FOtime[np.arange(0, len(FOtime)-plotStp, plotStp)], FOtimeStp[np.arange(0, len(FOtime)-plotStp, plotStp)],
            '.', alpha=0.1)
    ax.set_ylim(histRange)
    ax.set_xlim((0, FOtime[-1]))
    ax.set_xlabel('time [s]')
    ax.set_ylabel('time step [1/s]')
    myAxisTheme(ax)

    ax = tstpfig.add_subplot(gs[1])
    ax.hist(FOtimeStp, 50, histRange)
    ax.set_xlabel('time step [1/s]')
    ax.set_ylabel('count')
    ax.set_title('mean time step = ' + str(round(np.mean(FOtimeStp*1000.0), 2)) + 'ms')
    myAxisTheme(ax)

    tstpfig.tight_layout()

    return tstpfig


def myAxisTheme(myax):
    myax.get_xaxis().tick_bottom()
    myax.get_yaxis().tick_left()
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)