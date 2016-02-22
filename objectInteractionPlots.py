"""
Plots that characterise the interaction of the fly with an object.
For example, the effect of the realative heading angle and the distance of the object on turns and walking velocity.
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

from sys import path

# Import custom plotting functions
path.insert(1, '/Users/hannah/Dropbox/code/plottingUtilities/')
from plottingUtilities import myAxisTheme


def modulationOfRuns(turnModfig, gammaFull, vRotFilt_ds, selectedRangeDist, selectedRangeDistTurn,objDistance):
    # Directional modulation of runs (Gomez-Marin and Louis, 2014)
    turnMod = turnModfig.add_subplot(111)
    turnModsc = turnMod.scatter(gammaFull[selectedRangeDist], vRotFilt_ds[selectedRangeDist], marker='o', s=15,
                                linewidths=0, c=objDistance[selectedRangeDist], cmap=plt.cm.Spectral, alpha=0.5)
    turnMod.scatter(gammaFull[selectedRangeDistTurn], vRotFilt_ds[selectedRangeDistTurn], marker='o', s=15,
                    linewidths=0.5, c=objDistance[selectedRangeDistTurn], cmap=plt.cm.Spectral, alpha=0.5)
    turnMod.set_xlim(-np.pi, np.pi)
    turnMod.set_ylim(-5, 5)
    turnMod.set_xlabel('relative heading [rad]\n(or local bearing)')
    turnMod.set_ylabel('instantaneous rot. velocity [rad/s]\n(filtered)')

    turnModcb = plt.colorbar(turnModsc)
    turnModcb.set_label('distance from object [mm]')
    myAxisTheme(turnMod)

    return turnMod


def plotResidencyInMiniarena(ax, xPosMA, yPosMA, numBins, arenaRad,colormap, titleString):
    ax.hexbin(xPosMA, yPosMA, gridsize=numBins, cmap=colormap)
    plt.xlabel('x'); plt.ylabel('y')
    ax.set_xlim(-arenaRad-5, arenaRad+5); ax.set_ylim(-arenaRad-5, arenaRad+5)
    ax.set_title(titleString)
    ax.set_aspect('equal')
    myAxisTheme(ax)


def plotPositionHistogram(ax, posMA, numBins, histRange, colormap, alphaval, plotOrientation):
    plt.hist(posMA, bins=numBins, range=histRange, color=colormap, alpha=alphaval, orientation=plotOrientation)

    if plotOrientation == 'vertical':
        ax.set_ylabel('count')
        ax.set_xlim(histRange)
    else:
        ax.set_ylim(histRange)
        # plt.yticks(np.arange(0, max(nCount)+1, 1.0))

    myAxisTheme(ax)


def residencyWithHistograms_splitOnWalking(xPosMAall, yPosMAall, movingall, arenaRad, numBins, vTransTH, figureTitle):
    histRange = (-arenaRad, arenaRad)

    gs = gridspec.GridSpec(3, 4, width_ratios=[5, 1, 5, 1], height_ratios=[0.05, 1, 5])

    hexplotfig = plt.figure(figsize=(15, 7.5))
    sns.set_style('ticks')

    ax1 = hexplotfig.add_subplot(gs[1, 0])
    plotPositionHistogram(ax1, xPosMAall[~movingall], numBins, histRange, 'grey', 0.4, 'vertical')

    ax2 = hexplotfig.add_subplot(gs[2, 1])
    plotPositionHistogram(ax2, yPosMAall[~movingall], numBins, histRange, 'grey', 0.4, 'horizontal')

    ax3 = hexplotfig.add_subplot(gs[2, 0])
    plotResidencyInMiniarena(ax3, xPosMAall[~movingall], yPosMAall[~movingall], numBins, arenaRad, 'Greys',
                             'Non moving (transl. velocity < ' + str(vTransTH) + ' mm/s)')

    ax4 = hexplotfig.add_subplot(gs[1, 2])
    plotPositionHistogram(ax4, xPosMAall[movingall], numBins, histRange, 'grey', 0.4, 'vertical')

    ax5 = hexplotfig.add_subplot(gs[2, 3])
    plotPositionHistogram(ax5, yPosMAall[movingall], numBins, histRange, 'grey', 0.4, 'horizontal')

    ax6 = hexplotfig.add_subplot(gs[2, 2])
    plotResidencyInMiniarena(ax6, xPosMAall[movingall], yPosMAall[movingall], numBins, arenaRad, 'Greys',
                             'Moving (transl. velocity > ' + str(vTransTH) + ' mm/s)')

    hexplotfig.suptitle(figureTitle, fontsize=14)

    hexplotfig.autofmt_xdate()
    hexplotfig.tight_layout()

    return hexplotfig


def residencyWithHistograms(xPosMAall, yPosMAall, movingall, arenaRad, numBins, vTransTH, figureTitle):
    histRange = (-arenaRad, arenaRad)

    gs = gridspec.GridSpec(3, 2, width_ratios=[5, 1], height_ratios=[0.05, 1, 5])

    hexplotfig = plt.figure(figsize=(7, 7.5))
    sns.set_style('ticks')

    ax1 = hexplotfig.add_subplot(gs[1, 0])
    plotPositionHistogram(ax1, xPosMAall[movingall], numBins, histRange, 'grey', 0.4, 'vertical')

    ax2 = hexplotfig.add_subplot(gs[2, 1])
    plotPositionHistogram(ax2, yPosMAall[movingall], numBins, histRange, 'grey', 0.4, 'horizontal')

    ax3 = hexplotfig.add_subplot(gs[2, 0])
    plotResidencyInMiniarena(ax3, xPosMAall[movingall], yPosMAall[movingall], numBins, arenaRad, 'Greys',
                             'Moving (transl. velocity > ' + str(vTransTH) + ' mm/s)')

    hexplotfig.suptitle(figureTitle, fontsize=14)

    hexplotfig.autofmt_xdate()
    hexplotfig.tight_layout()

    return hexplotfig

# Heading-object distance relationship .................................................................................

def headingDistanceHistogram(headingDistHistSplt, objdistToPlot, gammaToPlot, distEdges, angleEdges):
    n, xedges, yedges = np.histogram2d(objdistToPlot, gammaToPlot, bins=(distEdges, angleEdges))

    X, Y = np.meshgrid(yedges, xedges)

    ringArea = (np.pi*xedges[1:]**2) - (np.pi*xedges[:-1]**2)
    ringArea2D = np.reshape(np.repeat(ringArea, (len(yedges)-1)), (len(xedges)-1, len(yedges)-1), order='C')

    headingDistHistSplt.pcolormesh(Y, X, n/ringArea2D)
    headingDistHistSplt.set_xlim(min(xedges), max(xedges))
    headingDistHistSplt.set_ylim(min(yedges), max(yedges))
    headingDistHistSplt.set_xlabel('radial distance from object [mm]')
    headingDistHistSplt.set_ylabel('absolut angle relative to object [deg]')

    return headingDistHistSplt


def anglePerFlyHist(radHistSplt, flyIDs, gammaToPlot, flyIDallarray, angleEdges, angleBins):

    import matplotlib.colors as colors

    numFlies = len(flyIDs)
    flyCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=numFlies), cmap='Accent')

    for fly in range(numFlies):
        n, edges = np.histogram(gammaToPlot[flyIDallarray == flyIDs[fly]],
                                range=(min(angleEdges), max(angleEdges)), bins=angleBins, normed=True)
        edgeCenteres = edges[:-1]+np.mean(np.diff(edges))/2

        # alphaVal = min((1+len(gammaToPlot[flyIDallarray == flyIDs[fly]]))/(10.0*60*10), 1)
        radHistSplt.plot(n, edgeCenteres, color=flyCMap.to_rgba(fly))# , alpha=alphaVal)

    radHistSplt.set_xlabel('count')
    radHistSplt.set_ylim(min(angleEdges), max(angleEdges))

    return radHistSplt


def distancePerFlyHist(radHistSplt, flyIDs, objdistToPlot, flyIDallarray, distEdges, distBins):

    import matplotlib.colors as colors

    numFlies = len(flyIDs)
    flyCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=numFlies), cmap='Accent')

    for fly in range(numFlies):
        n, edges = np.histogram(objdistToPlot[flyIDallarray == flyIDs[fly]],
                                range=(min(distEdges), max(distEdges)), bins=distBins, normed=True)
        edgeCenteres = edges[:-1]+np.mean(np.diff(edges))/2
        ringArea = (np.pi*edges[1:]**2) - (np.pi*edges[:-1]**2)

        # alphaVal = min((1+len(objdistToPlot[flyIDallarray == flyIDs[fly]]))/(10.0*60*10), 1)

        radHistSplt.plot(edgeCenteres, n/ringArea, color=flyCMap.to_rgba(fly))# ,alpha=alphaVal)

    radHistSplt.set_ylabel('count (normed to area)')
    radHistSplt.set_xlim(min(distEdges), max(distEdges))
    radHistSplt.legend(flyIDs, ncol=4, loc='upper center', bbox_to_anchor=(0.85, 1), fontsize=8)

    return radHistSplt


def radDistAngleCombiPlot(distBins, angleBins, maxDist, flyIDs, flyIDarray, objDistance, gamma):

    distEdges = np.linspace(0, maxDist, distBins)
    angleEdges = 180.0/np.pi*np.linspace(0, np.pi, angleBins)

    gammaToPlot = 180.0/np.pi*abs(gamma)

    objdistToPlot = objDistance

    headingDistFig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=(1, 3), width_ratios=(3, 1))

    # Subplot1: per fly distance histogram
    radHistSplt = headingDistFig.add_subplot(gs[0])
    radHistSplt = distancePerFlyHist(radHistSplt, flyIDs, objdistToPlot, flyIDarray, distEdges, distBins)
    myAxisTheme(radHistSplt)

    # Subplot2:  2d distance/angle histogram
    headingDistHistSplt = headingDistFig.add_subplot(gs[2])
    headingDistHistSplt = headingDistanceHistogram(headingDistHistSplt, objdistToPlot, gammaToPlot, distEdges, angleEdges)
    myAxisTheme(headingDistHistSplt)

    # Subplot3: per fly angle histogram
    radHistSplt = headingDistFig.add_subplot(gs[3])
    radHistSplt = anglePerFlyHist(radHistSplt, flyIDs, gammaToPlot, flyIDarray, angleEdges, angleBins)
    myAxisTheme(radHistSplt)

    headingDistFig.tight_layout()

    return headingDistFig


# Curvature related plots ..............................................................................................

def curvatureControlPlot(curv, xPos, yPos, sf, ef, rangeLimits, colormap):
    # Cap extremes of curvature to gain dynamic range at values around zero
    curvTH = 0.3
    curvToPlot = np.copy(curv)
    curvToPlot[curv > curvTH] = curvTH
    curvToPlot[curv < -curvTH] = -curvTH

    curvControlFig = plt.figure(figsize=(10, 10))
    ax = curvControlFig.add_subplot(111)
    plt.scatter(xPos[sf:ef], yPos[sf:ef], s=25, c=curvToPlot[sf:ef], cmap=colormap, marker='o', edgecolors='none')
    plt.xlim(rangeLimits)
    plt.ylim(rangeLimits)
    ax.set_aspect('equal')

    myAxisTheme(ax)

    return curvControlFig


def curvatureVsHeading_DistanceScatterplot(curvature, gammaFull, objDist):
    curvScatterPlot = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1.4])

    ax0 = curvScatterPlot.add_subplot(gs[0])
    plt.scatter(gammaFull, curvature, c=objDist, s=60-objDist, marker='.', alpha=0.8,
                edgecolors='none', cmap='Spectral')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-0.3, 0.3)

    ax1 = curvScatterPlot.add_subplot(gs[1])
    plt.scatter(gammaFull, curvature, c=objDist, s=objDist, marker='.', alpha=0.8, edgecolors='none', cmap='Spectral')
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-0.15, 0.15)

    ax0.axvline(-120*(np.pi/180), linestyle='--', color='grey', alpha=0.8)
    ax0.axvline(120*(np.pi/180), linestyle='--', color='grey', alpha=0.8)
    ax1.axvline(-120*(np.pi/180), linestyle='--', color='grey', alpha=0.8)
    ax1.axvline(120*(np.pi/180), linestyle='--', color='grey', alpha=0.8)

    ax0.axhline(0, linestyle='-', color='black', alpha=0.5)
    ax1.axhline(0, linestyle='-', color='black', alpha=0.5)

    myAxisTheme(ax0)
    myAxisTheme(ax1)

    return


def curvatureVsHeading_DistanceBoxplot(curvature, gammaFull, objDist, nGammaBins, nDistBins, arenaRad, titleString):
    # Bin values w.r.t distance from object and subsequently w.r.t. heading angle
    # Then generate box plot with nGammaBins for each distance bin
    # nGammaBins = 18
    # nDistBins = 3
    gammaBins = np.linspace(-np.pi, np.pi, nGammaBins+1)
    gammaBins = gammaBins[1:]

    nBinOutOfFOV = (180-120)/(360/nGammaBins)
    nBinInFOV = nGammaBins - 2*nBinOutOfFOV

    distBins = np.linspace(0, arenaRad, nDistBins+1)
    distBins = distBins[1:]

    curvDigitized = []
    gammaFullDigitized = []

    cmap = plt.cm.get_cmap('Accent')
    colors = np.zeros((nDistBins, 4))
    outcolor = [0.5,  0.5,  0.5,  1]

    binInd = np.digitize(objDist, distBins)

    for i in range(nDistBins):
        curvDigitized.append(curvature[binInd == i])
        gammaFullDigitized.append(gammaFull[binInd == i])
        colors[i] = cmap(float(i)/nDistBins)

    curvBoxPlot=plt.figure(figsize=(8, 11))
    curvBoxPlot.suptitle(titleString, fontsize=13)

    for distLevel in range(nDistBins):
        curvDoubleDigitized = []
        curvDigitizedMedian = np.zeros(nGammaBins)
        binInd = np.digitize(gammaFullDigitized[distLevel], gammaBins)

        for i in range(nGammaBins):
            curvDoubleDigitized.append(curvDigitized[distLevel][binInd == i])
            curvDigitizedMedian[i] = np.median(curvDigitized[distLevel][binInd == i])

        outcolorList = np.reshape(np.repeat(outcolor, nBinOutOfFOV), (4, nBinOutOfFOV)).transpose()
        incolorList = np.reshape(np.repeat(colors[distLevel], nBinInFOV), (4, nBinInFOV)).transpose()
        colorList = np.vstack((outcolorList, incolorList, outcolorList))

        ax = curvBoxPlot.add_subplot(nDistBins, 1, distLevel+1)
        for g in range(nGammaBins):
            y = curvDoubleDigitized[g]
            # Add some random "jitter" to the x-axis
            x = np.random.normal(g+1, 0.06, size=len(y))
            plt.plot(x, y, '.', color=colorList[g], alpha=1, markersize=5)

        box = ax.boxplot(curvDoubleDigitized, notch=True, patch_artist=True)
        plt.setp(box['boxes'], alpha=0.6, edgecolor='black')
        plt.setp(box['whiskers'], color='black', linewidth=1, linestyle='-')
        plt.setp(box['caps'], color='black', linewidth=1, linestyle='-')
        for patch, color in zip(box['boxes'], colorList):
            patch.set_facecolor(color)
        ax.set_ylabel(str(int(distBins[distLevel]-np.mean(np.diff(distBins)))) + '-'
                      + str(int(distBins[distLevel])) + ' mm\n' + 'curvature [1/mm]')
        plt.xticks(range(1, len(gammaBins)+1),
                   np.round((gammaBins - np.mean(np.diff(gammaBins))/2)*180/np.pi, 2),
                   rotation=40)
        ax.plot(range(1, len(gammaBins)+1), curvDigitizedMedian, color='black', alpha=0.8)
        ax.set_ylim(-0.25, 0.25)
        ax.axhline(0, color='black', linewidth=1, linestyle='--')

        myAxisTheme(ax)

    ax.set_xlabel('heading relative to object [deg]')

    return curvBoxPlot
