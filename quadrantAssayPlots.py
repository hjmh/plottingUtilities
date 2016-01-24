"""
Plotting functions for analysing ctrax tracking data from activation experiments
collected in the free walking arena
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import matplotlib.colors as colors
import numpy as np
import seaborn as sns
from scipy import sparse as sps

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec

from flyTracePlots import plotPosInRangeMultifly, plotActiveSegments


def lightONindicator(subplt, x, y, width, height):
    subplt.add_patch(
        patches.Rectangle((x, y), width, height, alpha=0.2, facecolor='red', edgecolor='none')
    )


def makeBeginIndexMatrix(stimStartFrame, numRepeat, stimS, pauseS, fps):
    beginChr = np.array([[stimStartFrame + (repeat * stimS + repeat * pauseS) * fps,
                          stimStartFrame + ((repeat + 1) * stimS + repeat * pauseS) * fps]
                         for repeat in range(numRepeat)])
    return beginChr


def makeEndIndexMatrix(stimStartFrame, numRepeat, stimS, pauseS, fps):
    endChr = np.array([(stimStartFrame + ((repeat + 1) * stimS + repeat * pauseS) * fps,
                        stimStartFrame + ((repeat + 1) * stimS + (repeat + 1) * pauseS) * fps)
                       for repeat in range(numRepeat)])
    return endChr


def plotStimulationBlock(blockfig, figGridSpace, blockFrames, blockColors, xPos, yPos, imagePx, angle, flyID):

    for block in range(len(blockFrames)):
        frames = range(blockFrames[block][0],blockFrames[block][1],blockFrames[block][2])
        blocksubplt = blockfig.add_subplot(figGridSpace[block])
        plotPosInRangeMultifly(blocksubplt, frames, xPos, yPos, angle, flyID, -1, blockColors[block], imagePx)

        if block % 2:
            blocksubplt.set_title('break block ' + str(block/2 + 1))
        else:
            blocksubplt.set_title('stimulation block ' + str(block/2 + 1))

    return blockfig


def plotStimulationBlock_timeBar(blockFrames, xPos, yPos, angle, flyID,
                                 blockColors, title, figureSize, colNumber, imageSizePx):

    blockTfig = plt.figure(figsize=figureSize)
    blockTfig.set_canvas(plt.gcf().canvas)
    sns.set_style('ticks')
    blockTfig.suptitle(title, fontsize=16)

    rowNumber = len(blockFrames)/colNumber
    gs = gridspec.GridSpec(colNumber+1, rowNumber, height_ratios=np.hstack((np.repeat([4], colNumber), 1)))
    blockTfig = plotStimulationBlock(blockTfig, gs, blockFrames, blockColors, xPos, yPos, imageSizePx, angle, flyID)

    timeplt = blockTfig.add_subplot(gs[len(blockFrames):len(blockFrames)+1])
    frameRange = range(blockFrames[0][0], blockFrames[0][1], blockFrames[0][2])
    timeCol = colors.Normalize(vmin=0, vmax=2.5*len(frameRange))

    for col in range(2):
        for frame in frameRange:
            scalarMap = plt.cm.ScalarMappable(norm=timeCol, cmap=blockColors[col])
            timeplt.plot(frame, 0.5*col+1, marker='.', markersize=12, linestyle='none', alpha=0.5,
                         color=scalarMap.to_rgba(frame-frameRange[0]))

    timeplt.get_xaxis().tick_bottom()
    timeplt.get_yaxis().set_visible(False)
    timeplt.spines['top'].set_visible(False)
    timeplt.spines['right'].set_visible(False)
    timeplt.spines['left'].set_visible(False)
    plt.xlim([frameRange[0], frameRange[-1]])
    plt.ylim([0.5, 2.5])
    plt.xlabel('Color code for time')

    return blockTfig


def plotQuadrantDetail(subplt, titleString, frameRange, flyIDperFrame, xPos, yPos, angle, flyID, imagePx):
    FOI = flyIDperFrame[frameRange]  # frames of interest
    activeFragments = np.array(np.nonzero(sum(FOI))).squeeze()

    if len(activeFragments) == 0 :
        print('No active fly fragments')
    else:
        plotActiveSegments(activeFragments, frameRange, subplt, titleString, xPos, yPos, angle, flyID, imagePx)


def addQuadrantLightIndicator(indicatorSubPlt, imageSizePx, quadPattern):
    if quadPattern == 0:
        lightONindicator(indicatorSubPlt, 0, 0, imageSizePx[0]/2, imageSizePx[1]/2)
        lightONindicator(indicatorSubPlt, imageSizePx[0]/2, imageSizePx[1]/2, imageSizePx[0]/2, imageSizePx[1]/2)
    else:
        lightONindicator(indicatorSubPlt, 0, imageSizePx[1]/2, imageSizePx[0]/2, imageSizePx[1]/2)
        lightONindicator(indicatorSubPlt, imageSizePx[0]/2, 0, imageSizePx[0]/2, imageSizePx[1]/2)


def defineQuadrantDetail(strtFrame, stimSecQ, timeWindow, fps, flyIDperFrame, skipFrames, windowType):

    if windowType == 'pre':
        fRange = range(strtFrame + (stimSecQ-timeWindow)*fps, strtFrame + stimSecQ*fps, skipFrames)
        titleString = str(strtFrame + (stimSecQ-timeWindow)*fps) + ' to ' + str(strtFrame + stimSecQ*fps)
    elif windowType == 'post':
        fRange = range(strtFrame + stimSecQ*fps, strtFrame + (stimSecQ+timeWindow)*fps, skipFrames)
        titleString = str(strtFrame + stimSecQ*fps) + ' to ' + str(strtFrame + (stimSecQ+timeWindow)*fps)
    else:
        print('Invalid window type')
        return

    activeFragments = np.array(np.nonzero(sum(flyIDperFrame[fRange]))).squeeze()

    return activeFragments, fRange, titleString


def plotFractionFlyInOn(quadFractionFig, frameRange, stimSec, quadrantRepeat, patternA, patternB,
                        xPos, yPos, imageSizePx, numQuadRepeat):
    time = np.linspace(0, stimSec, len(frameRange))

    fliesInLEDon = np.zeros((len(frameRange), 1))
    fractionInLEDon = np.zeros((len(frameRange), 1))

    if quadrantRepeat % 2 == 0:
        ONpattern = patternA
    else:
        ONpattern = patternB

    for ind, frame in enumerate(frameRange):
        # assemble fly positions for frame
        spsData = np.ones((len(xPos[frame]), 1)).squeeze()
        spsColInds = np.array(xPos[frame].astype('int').squeeze())
        spsRowInds = np.array(yPos[frame].astype('int').squeeze())

        flyLocations = sps.coo_matrix((spsData, (spsRowInds, spsColInds)),
                                      shape=(imageSizePx[0], imageSizePx[1])).toarray()

        # check in wich pattern flies appear
        fliesInONpattern = (ONpattern + flyLocations) > 1
        fliesInLEDon[ind] = sum(sum(fliesInONpattern))
        fractionInLEDon[ind] = sum(sum(fliesInONpattern)) / sum(sum(flyLocations))

    subplt = quadFractionFig.add_subplot(1, numQuadRepeat, quadrantRepeat + 1)
    subplt.plot(time, fractionInLEDon, marker='.', linestyle='none', color='black', alpha=0.75)
    lightONindicator(subplt, 0, 0.5, stimSec, 0.5)

    sns.despine(right=True, offset=0)  # trim=True)
    sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': '.8', })
    subplt.set_title('block ' + str(quadrantRepeat + 1), fontsize=10)
    subplt.set_ylim((0, 1))

    if quadrantRepeat == 0:
        subplt.set_xlabel('time [s]')
        subplt.set_ylabel('fraction of detected\nflies in "ON-quadrants"')
    else:
        subplt.set_yticks([])
