"""
Plots that characterise the interaction of the fly with an object.
For example, the effect of the realative heading angle and the distance of the object on turns and walking velocity.
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as colors
import seaborn as sns
from os.path import sep
from os import getcwd
from sys import path

# Set path to analysis code directory
codeDir = sep.join(getcwd().split(sep)[:-2])
path.insert(1, codeDir)

from trajectoryAnalysis.trajectoryDerivedParams import cartesian2polar
from .basicPlotting import myAxisTheme


# Modulation of runs ..........................................................................................

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


# 2D (cartesian) residency histograms ........................................................................

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


# 1D (radial) residency histograms ...........................................................................

def oneDimResidencyWithVar_df(radResPlt, FODataframe, flyIDs, movementFilter, minLMdist, visState,
                              numBins, histRange, lineAlpha, plotLog, varstyle, fill, condLegend):
    numFlies = len(flyIDs)

    # normalisation factor for cirle area rings
    areaNormA = np.square(np.linspace(histRange[0], histRange[1], numBins))*np.pi
    areaNorm = areaNormA[1:]-areaNormA[:-1]

    # colormap for trials (visible object trials in colour, invisible object trials in grey shades)
    numInvTrials = sum(['invisible' in visState[trial] for trial in range(len(visState))])
    numVisTrials = len(visState)-numInvTrials

    visTrialCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=numVisTrials), cmap='Reds')
    invTrialCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=-2, vmax=numInvTrials), cmap='Greys')
    trialCMap = [visTrialCMap.to_rgba(visTrial) for visTrial in range(numVisTrials)]
    [trialCMap.append(invTrialCMap.to_rgba(invTrial)) for invTrial in range(numInvTrials)]

    legendhand = []

    for trial, cond in enumerate(visState):
        trialRadRes = np.zeros((numFlies, numBins-1))
        for fly in range(numFlies):
            querystring = '(trialtype=="' + cond + '") & (trial==' + str(trial+1) + ') & (' + movementFilter\
                          + ') & (flyID=="' + flyIDs[fly]+'") & (LMdist > '+str(minLMdist)+')'

            xPosMA = FODataframe.query(querystring)['xPosInMiniarena'].values.squeeze()
            yPosMA = FODataframe.query(querystring)['yPosInMiniarena'].values.squeeze()

            # transform trajectory to polar coordinates
            objDist, theta = cartesian2polar(xPosMA, yPosMA)

            radresidency, edges = np.histogram(objDist, bins=np.linspace(histRange[0], histRange[1], numBins))
            trialRadRes[fly, :] = radresidency/areaNorm

            jitterRange = 0.2*np.diff(histRange)/numBins

            if varstyle == 'dotplot':
                if plotLog:
                    toplot = np.log(radresidency/areaNorm)
                else:
                    toplot = radresidency/areaNorm

                radResPlt.plot(edges[:-1]+np.diff(edges)/2.0+np.random.uniform(-jitterRange, jitterRange), toplot,
                               color=trialCMap[trial], linestyle='none', marker='.', alpha=0.5)

        if plotLog:
            if varstyle == 'std':
                toplot = np.log(np.nanmean(trialRadRes, 0))
                var1 = np.log(np.nanmean(trialRadRes, 0) + np.nanstd(trialRadRes, 0))
                var2 = np.log(np.nanmean(trialRadRes, 0) - np.nanstd(trialRadRes, 0))
            elif varstyle == 'iqr':
                toplot = np.log(np.nanmedian(trialRadRes,0))
                [var1, var2] = np.log(np.nanpercentile(trialRadRes, [25, 75], axis=0))
            else:
                toplot = np.log(np.nanmean(trialRadRes, 0))

        else:
            if varstyle == 'std':
                toplot = np.nanmean(trialRadRes, 0)
                var1 = toplot + np.nanstd(trialRadRes, 0)
                var2 = toplot - np.nanstd(trialRadRes, 0)
            elif varstyle == 'iqr':
                toplot = np.nanmedian(trialRadRes, 0)
                [var1, var2] = np.nanpercentile(trialRadRes, [25, 75], axis=0)
            else:
                toplot = np.nanmean(trialRadRes,0)

        if varstyle != 'dotplot':
            lhand, = radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, toplot, color=trialCMap[trial],
                                    alpha=lineAlpha,linewidth=3)
            radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, var1, color=trialCMap[trial], alpha=lineAlpha, linewidth=1)
            radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, var2, color=trialCMap[trial], alpha=lineAlpha, linewidth=1)
            if fill:
                radResPlt.fill_between(edges[:-1]+np.diff(edges)/2.0, var1, var2, color=trialCMap[trial], alpha=0.2)
        else:
            lhand, = radResPlt.plot(edges[:-1]+np.diff(edges)/2.0, toplot, color=trialCMap[trial], alpha=lineAlpha,
                                    linewidth=3)

        legendhand.append(lhand)

    plt.legend(legendhand, condLegend, loc='best', fontsize=12)
    radResPlt.set_xlabel('object distance [mm]', fontsize=12)
    if plotLog:
        radResPlt.set_ylabel('log(area corrected residency [count/mm^2])', fontsize=12)
    else:
        radResPlt.set_ylabel('area corrected residency [count/mm^2]', fontsize=12)

    return radResPlt

# Distributions -----------------------------------------------------------------------------------------------

def lineHistogram(ax, histRange, yVals, xlab, ylab, lineCol, densityFlag, nBins):
    n, edges = np.histogram(yVals, normed=densityFlag, density=densityFlag, range=histRange, bins=nBins)
    edges = edges[:-1] + np.diff(edges)/2.0
    ax.plot(edges, n, color=lineCol, linewidth=1.5)

    ax.set_xlim(histRange)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    myAxisTheme(ax)

    return ax, n, edges


def headingLineHist(ax, flyCol, edges, normedhead, curralpha, histRange, xlab):
    shiftedges = edges[:-1]

    ax.plot(shiftedges, normedhead, color=flyCol, alpha=curralpha)
    ax.set_xlim(histRange)
    ax.set_xlabel(xlab)
    myAxisTheme(ax)

    return shiftedges

# Velo distributions figure ...................................................................................

def plotWalkingVelocityDistr(allFlies_df, flyIDs, flyCMap, histRangeVT, histRangeVR, numBins):
    walkingFig, axs = plt.subplots(2, 2, figsize=(10, 8))
    verString = ['Frequency', 'Count']
    xString = ['Translational velocity [mm/s]', 'Rotational velocity [deg/s]']
    numFlies = len(flyIDs)

    for ver, densityFlag in enumerate([True, False]):
        histDat = np.zeros((2, numFlies, numBins))

        for fly in range(numFlies):
            flyMov = allFlies_df.query('(flyID == "' + flyIDs[fly] + '")')['moving'].values.squeeze()
            flyVT = allFlies_df.query('(flyID == "' + flyIDs[fly] + '")')['transVelo'].values.squeeze()
            flyVR = allFlies_df.query('(flyID == "' + flyIDs[fly] + '")')['rotVelo'].values.squeeze()

            # translational velocity
            axs[ver, 0], n, edgevt = lineHistogram(axs[ver, 0], histRangeVT, flyVT[flyMov > 0], xString[0],
                                                  verString[ver], flyCMap.to_rgba(fly), densityFlag, numBins)
            histDat[0, fly, :] = n

            # rotational velocity
            axs[ver, 1], n, edgevr = lineHistogram(axs[ver, 1], histRangeVR, flyVR[flyMov > 0], xString[1],
                                                  verString[ver], flyCMap.to_rgba(fly), densityFlag, numBins)
            histDat[1, fly, :] = n

            edges = [edgevt, edgevr]

        for ind in range(2):
            axs[ver, ind].plot(edges[ind], np.nanmedian(histDat[ind, :, :], axis=0), color='k', linewidth=2)
            [var1, var2] = np.nanpercentile(histDat[ind, :, :], [25, 75], axis=0)
            axs[ver, ind].fill_between(edges[ind], var1, var2, color='k', alpha=0.2)
            axs[ver, 1].axvline(0, linestyle='dashed', color='grey', linewidth=2)

    return walkingFig


# Heading angle distribution plots (e.g. for stripe tracking) .................................................

def plotHeadingDistFromTimeseries(figax, headingts, densityFlag, rangevals, binvals, flycol, xlab, ylab, alphaval):

    nhead, edges = np.histogram(headingts, normed=densityFlag, density=densityFlag, range=rangevals, bins=binvals)

    if densityFlag:
        normFactor = nhead.sum()
    else:
        normFactor = 1.0

    figax.plot(edges[:-1]+np.diff(edges)/2, nhead/normFactor, color=flycol, alpha=alphaval)
    figax.set_xlim(rangevals)
    figax.set_xlabel(xlab)
    figax.set_ylabel(ylab)
    myAxisTheme(figax)

    return figax, nhead/normFactor, edges[:-1]+np.diff(edges)/2


def plotHeadingComparison(FOAllFlies_df, flyIDs, sceneName, minLMdist, titleString, flyCMap, densityFlag, plotIQR):
    halfBins = 18
    fullBins = 36
    alphaval = [1, 0.6]

    ystring = ['(moving)', '(standing)']

    headingfig, axs = plt.subplots(2, 2, figsize=(10, 8))

    numFlies = len(flyIDs)
    nhead_halfGamma = np.nan * np.ones((numFlies, halfBins, 2))
    nhead_fullGamma = np.nan * np.ones((numFlies, fullBins, 2))

    for movever in range(2):  # moving vs not moving
        legendlist = []

        for fly in range(numFlies):
            flyCol = flyCMap.to_rgba(fly)
            querystring = '(sceneName=="' + sceneName + '") & (flyID =="' + flyIDs[fly] + '")'
            flydf = FOAllFlies_df.query(querystring)
            if len(flydf) == 0: continue
            if flydf['trialtype'].values[0] == 'plane':
                flydf = flydf.query('LMdist > '+str(minLMdist))

            moving = flydf['moving'].values.squeeze()
            gamma = flydf['gamma'].values.squeeze()
            gammaFull = flydf['gammaFull'].values.squeeze()

            if sum(moving) <= 0.2 * len(moving):
                print('fly ' + str(flyIDs[fly]) + ' not moving')
                print(100.0 * sum(moving) / max(1, len(moving)))
                continue

            legendlist.append(flyIDs[fly])

            if movever == 0:
                gamma = gamma[moving > 0]
                gammaFull = gammaFull[moving > 0]
            else:
                gamma = gamma[moving == 0]
                gammaFull = gammaFull[moving == 0]

            # Half gamma  -  gammaPlt = axs[0,plotver]
            histRange = (0, np.pi)
            nhead, edges = np.histogram(gamma, normed=densityFlag, density=densityFlag, range=histRange,
                                        bins=halfBins)

            if densityFlag:
                axs[movever, 0].set_ylabel('frequency ' + ystring[movever])
                normFactor = nhead.sum()
            else:
                axs[movever, 0].set_ylabel('count ' + ystring[movever])
                normFactor = 1.0

            halfedges = headingLineHist(axs[movever, 0], flyCol, edges, nhead / normFactor,
                                        alphaval[movever], histRange, 'rel. heading')

            nhead_halfGamma[fly, :, movever] = nhead / normFactor

            # Full gamma  -  gammaFullPlt = axs[0,1]
            histRange = (-np.pi, np.pi)
            nhead, edges = np.histogram(gammaFull, normed=densityFlag, density=densityFlag, range=histRange,
                                        bins=fullBins)
            if densityFlag:
                normFactor = nhead.sum()
            else:
                normFactor = 1.0

            fulledges = headingLineHist(axs[movever, 1], flyCol, edges, nhead / normFactor,
                                        alphaval[movever], histRange, 'rel. heading (full)')

            nhead_fullGamma[fly, :, movever] = nhead / normFactor

        axs[movever, 0].plot(halfedges, np.nanmedian(nhead_halfGamma[:, :, movever], 0), color='k', linewidth=3)
        axs[movever, 1].plot(fulledges, np.nanmedian(nhead_fullGamma[:, :, movever], 0), color='k', linewidth=3)

        if (plotIQR):
            [var1, var2] = np.nanpercentile(nhead_halfGamma[:, :, movever], [25, 75], axis=0)
            axs[movever, 0].fill_between(halfedges, var1, var2, color='k', alpha=0.2)

            [var1, var2] = np.nanpercentile(nhead_fullGamma[:, :, movever], [25, 75], axis=0)
            axs[movever, 1].fill_between(fulledges, var1, var2, color='k', alpha=0.2)

    headingfig.suptitle(titleString, fontsize=13)
    axs[movever, 1].legend(legendlist, ncol=4)
    headingfig.tight_layout()

    return headingfig, nhead_fullGamma[:, :, 0]


def plot2DHeadingComparison(FOAllFlies_df, flyIDs, sceneName, minLMdist, titleString, flyCMap, densityFlag, plotIQR):
    numFlies = len(flyIDs)

    legendlist = []
    fullBins = 36
    minMov = 0.2
    histRange = (-np.pi, np.pi)

    nhead_angle = np.nan*np.ones((numFlies,fullBins))
    nhead_gamma = np.nan*np.ones((numFlies,fullBins))

    headingfig, axs = plt.subplots(1, 2, figsize=(10, 4))

    for fly in range(numFlies):
        querystring = '(sceneName=="' + sceneName + '") & (flyID =="' + flyIDs[fly] +  '")'

        flydf = FOAllFlies_df.query(querystring)
        if len(flydf) == 0: continue
        if flydf['trialtype'].values[0] == 'plane':
            flydf = flydf.query('LMdist > '+str(minLMdist))

        moving = flydf['moving'].values.squeeze()
        angle = flydf['headingAngle'].values.squeeze()
        gammaFull = flydf['gammaFull'].values.squeeze()

        if sum(moving)<=minMov*len(moving):
            print('fly '+str(flyIDs[fly])+' not moving: '+  str(100.0*sum(moving)/max(1,len(moving))))
            continue

        if densityFlag:
            ylab = 'frequency (when moving)'
        else:
            ylab = 'count (when moving)'

        # rel. heading
        axs[0], normgam, fulledges = plotHeadingDistFromTimeseries(axs[0], gammaFull[moving > 0], densityFlag,
                                                histRange, fullBins, flyCMap.to_rgba(fly),'rel. heading', ylab, 1)
        nhead_gamma[fly, :] = normgam

        # abs. heading
        axs[1], normang, fulledges = plotHeadingDistFromTimeseries(axs[1], angle[moving > 0], densityFlag,
                                                histRange, fullBins, flyCMap.to_rgba(fly),'abs. heading', ylab, 1)
        nhead_angle[fly,:] = normang

    headingfig.suptitle(titleString,fontsize=13)
    headingfig.tight_layout()

    axs[0].plot(fulledges,np.nanmedian(nhead_gamma[:,:],0),color='k',linewidth=3)
    axs[1].plot(fulledges,np.nanmedian(nhead_angle[:,:],0),color='k',linewidth=3)

    if(plotIQR):
        [var1,var2] = np.nanpercentile(nhead_gamma[:,:],[25,75],axis=0)
        axs[0].fill_between(fulledges, var1, var2, color='k', alpha=0.2)
        [var1,var2] = np.nanpercentile(nhead_angle[:,:],[25,75],axis=0)
        axs[1].fill_between(fulledges, var1, var2, color='k', alpha=0.2)
    return headingfig


# Heading-object distance relationship .................................................................................

# functions for heading vs. distance histogram ---------------
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


def anglePerFlyHist(radHistSplt, flyIDs, gammaToPlot, flyIDallarray, angleEdges, angleBins, timeTH):

    import matplotlib.colors as colors

    numFlies = len(flyIDs)
    flyCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=numFlies), cmap='Accent')

    for fly in range(numFlies):
        if sum(flyIDallarray == flyIDs[fly]) < timeTH:
            continue
        n, edges = np.histogram(gammaToPlot[flyIDallarray == flyIDs[fly]],
                                range=(min(angleEdges), max(angleEdges)), bins=angleBins, normed=True)
        edgeCenteres = edges[:-1]+np.mean(np.diff(edges))/2

        radHistSplt.plot(n, edgeCenteres, color=flyCMap.to_rgba(fly))

    radHistSplt.set_xlabel('count')
    radHistSplt.set_ylim(min(angleEdges), max(angleEdges))

    return radHistSplt


def distancePerFlyHist(radHistSplt, flyIDs, objdistToPlot, flyIDallarray, distEdges, distBins, timeTH):

    import matplotlib.colors as colors

    numFlies = len(flyIDs)
    flyCMap = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0, vmax=numFlies), cmap='Accent')

    for fly in range(numFlies):
        if sum(flyIDallarray == flyIDs[fly]) < timeTH:
            continue

        n, edges = np.histogram(objdistToPlot[flyIDallarray == flyIDs[fly]],
                                range=(min(distEdges), max(distEdges)), bins=distBins, normed=True)
        edgeCenteres = edges[:-1]+np.mean(np.diff(edges))/2
        ringArea = (np.pi*edges[1:]**2) - (np.pi*edges[:-1]**2)

        radHistSplt.plot(edgeCenteres, n/ringArea, color=flyCMap.to_rgba(fly))

    radHistSplt.set_ylabel('count (normed to area)')
    radHistSplt.set_xlim(min(distEdges), max(distEdges))
    flyIDsShort = []
    [flyIDsShort.append(flyIDs[fly][-3:]) for fly in range(numFlies)]
    radHistSplt.legend(flyIDsShort, ncol=4, loc='upper center', bbox_to_anchor=(0.85, 1), fontsize=8)

    return radHistSplt


def radDistAngleCombiPlot(distBins, angleBins, minDist, maxDist, flyIDs, flyIDarray, objDistance, gamma):

    distEdges = np.linspace(minDist, maxDist, distBins)
    angleEdges = np.linspace(-np.pi, np.pi, angleBins)
    timeTH = 0

    gammaToPlot = gamma

    objdistToPlot = objDistance

    headingDistFig = plt.figure(figsize=(9, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=(1, 3), width_ratios=(3, 1))

    # Subplot1: per fly distance histogram
    radHistSplt = headingDistFig.add_subplot(gs[0])
    radHistSplt = distancePerFlyHist(radHistSplt, flyIDs, objdistToPlot, flyIDarray, distEdges, distBins, timeTH)
    myAxisTheme(radHistSplt)

    # Subplot2:  2d distance/angle histogram
    headingDistHistSplt = headingDistFig.add_subplot(gs[2])
    headingDistHistSplt = headingDistanceHistogram(headingDistHistSplt, objdistToPlot, gammaToPlot, distEdges, angleEdges)
    myAxisTheme(headingDistHistSplt)

    # Subplot3: per fly angle histogram
    radHistSplt = headingDistFig.add_subplot(gs[3])
    radHistSplt = anglePerFlyHist(radHistSplt, flyIDs, gammaToPlot, flyIDarray, angleEdges, angleBins, timeTH)
    myAxisTheme(radHistSplt)

    headingDistFig.tight_layout()

    return headingDistFig


def radDistAngleCombiPlot_freeWalking(distBins, angleBins, minDist, maxDist, flyIDs, flyIDarrayIn, objDistance, gamma):

    distEdges = np.linspace(minDist, maxDist, distBins)
    angleEdges = np.linspace(-np.pi, np.pi, angleBins)
    timeTH = 0.25*10*60

    # filter for selected distance range
    inArea = np.logical_and(objDistance > minDist, objDistance < maxDist)
    gammaToPlot = gamma[inArea]
    objdistToPlot = objDistance[inArea]
    flyIDarray = flyIDarrayIn[inArea]

    headingDistFig = plt.figure(figsize=(9, 7))
    gs = gridspec.GridSpec(2, 2, height_ratios=(1, 3), width_ratios=(3, 1))

    # Subplot1: per fly distance histogram
    radHistSplt = headingDistFig.add_subplot(gs[0])
    radHistSplt = distancePerFlyHist(radHistSplt, flyIDs, objdistToPlot, flyIDarray, distEdges, distBins, timeTH)
    myAxisTheme(radHistSplt)

    # Subplot2:  2d distance/angle histogram
    headingDistHistSplt = headingDistFig.add_subplot(gs[2])
    headingDistHistSplt = headingDistanceHistogram(headingDistHistSplt, objdistToPlot, gammaToPlot, distEdges, angleEdges)
    myAxisTheme(headingDistHistSplt)

    # Subplot3: per fly angle histogram
    radHistSplt = headingDistFig.add_subplot(gs[3])
    radHistSplt = anglePerFlyHist(radHistSplt, flyIDs, gammaToPlot, flyIDarray, angleEdges, angleBins, timeTH)
    myAxisTheme(radHistSplt)

    headingDistFig.tight_layout()

    return headingDistFig

# functions related to heading vs. distance maps of steering parameter -----------------------

def relPolarCoordAverageMap(relPolMeanPlt, distEdges, angleEdges, valuesToMap, objDistance, gamma, colorMap, useMean,
                            maxValue, xlab, ylab):

    # bin valuesToMap by objectDistance value
    digitizedDist = np.digitize(objDistance, distEdges)

    # bin valuesToMap by objectDistance value
    digitizedAngle = np.digitize(gamma, angleEdges)

    meanVals = 1.0*np.zeros((len(angleEdges), len(distEdges)))

    for distBin in range(1, 1+len(distEdges)):
        for angleBin in range(1, 1+len(angleEdges)):
            sltPts = np.logical_and(digitizedDist == distBin, digitizedAngle == angleBin)
            if sum(sltPts) > 0:
                if useMean:
                    meanVals[angleBin-1, distBin-1, ] = np.mean(valuesToMap[sltPts])
                else:
                    # use median
                    meanVals[angleBin-1, distBin-1, ] = np.median(valuesToMap[sltPts])

    pc = relPolMeanPlt.pcolormesh(distEdges, angleEdges, meanVals, cmap=colorMap, vmin=-maxValue, vmax=maxValue)
    relPolMeanPlt.set_xlim(min(distEdges), max(distEdges))
    relPolMeanPlt.set_ylim(min(angleEdges), max(angleEdges))
    relPolMeanPlt.set_xlabel(xlab)
    relPolMeanPlt.set_ylabel(ylab)

    return relPolMeanPlt, meanVals, pc


def movementSplitPolarAverageMaps(relPolMeanPltAll, relPolMeanPltAp, relPolMeanPltDep, cbax, titleString, distEdges,
                                  angleEdges, app, dep, valuesToMap, objDistance, gamma, useMean, maxValue):

    cMap = 'RdBu'
    relPolMeanPltAp.set_title('Total')
    relPolMeanPltAll, meanVals, pc = relPolarCoordAverageMap(relPolMeanPltAll, distEdges, angleEdges,valuesToMap,
                                                             objDistance, gamma, cMap, useMean, maxValue,
                                                             'radial distance from object [mm]',
                                                             'heading angle relative to object [deg]')
    myAxisTheme(relPolMeanPltAll)

    relPolMeanPltAp.set_title('Approaches')
    relPolMeanPltAp, meanVals, pc = relPolarCoordAverageMap(relPolMeanPltAp, distEdges, angleEdges, valuesToMap[app],
                                                            objDistance[app], gamma[app], cMap, useMean, maxValue,
                                                            'radial distance from object [mm]', '')
    myAxisTheme(relPolMeanPltAp)

    relPolMeanPltDep.set_title('Departures')
    relPolMeanPltDep, meanVals, pc = relPolarCoordAverageMap(relPolMeanPltDep, distEdges, angleEdges, valuesToMap[dep],
                                                             objDistance[dep], gamma[dep], cMap, useMean, maxValue,
                                                             'radial distance from object [mm]', '')
    cb = plt.colorbar(pc, cax=cbax)
    cb.ax.set_ylabel(titleString, rotation=270, fontsize=12)
    myAxisTheme(relPolMeanPltDep)

    return relPolMeanPltAp, relPolMeanPltDep


def make4ValuePolCoordPlot(relPolMeanFig, objDistanceall, gammaall, gammaFullall, vT, polarCurv, d_theta, time,
                           distEdges, fullAngleEdges, maxVals):

    d_objDist = np.hstack((0, np.diff(np.convolve(objDistanceall, np.ones((7,))/7, mode='same'))))

    # Split data roughly into 'approaches' and 'departues'
    app = d_objDist < 0
    dep = d_objDist > 0
    useMean = False

    gs = gridspec.GridSpec(5, 4, width_ratios=[1, 1, 1, 0.05])

    # 1) Translational velocity
    titleString = 'Median translational velocity'
    _ = movementSplitPolarAverageMaps(relPolMeanFig.add_subplot(gs[0]), relPolMeanFig.add_subplot(gs[1]),
                                      relPolMeanFig.add_subplot(gs[2]), relPolMeanFig.add_subplot(gs[3]), titleString,
                                      distEdges, fullAngleEdges, app, dep, vT, objDistanceall, gammaFullall,
                                      useMean, maxVals[0])

    # 2) Translational acceleration
    d_vT = np.hstack((0, np.diff(np.convolve(vT, np.ones((5,))/5, mode='same'))/np.diff(time)))
    d_vT = np.convolve(d_vT, np.ones((5,))/5, mode='same')
    d_vT[abs(d_vT) > 2*np.std(abs(d_vT))] = np.sign(d_vT[abs(d_vT) > 2*np.std(abs(d_vT))])*2*np.std(abs(d_vT))

    titleString = 'Median translational acceleration'
    _ = movementSplitPolarAverageMaps(relPolMeanFig.add_subplot(gs[4]), relPolMeanFig.add_subplot(gs[5]),
                                      relPolMeanFig.add_subplot(gs[6]), relPolMeanFig.add_subplot(gs[7]), titleString,
                                      distEdges, fullAngleEdges, app, dep, d_vT, objDistanceall, gammaFullall,
                                      useMean, maxVals[1])

    # 3) Heading angle change
    d_gamma = np.hstack((0, np.diff(gammaall)/np.diff(time)))
    d_gamma = np.convolve(d_gamma, np.ones((7,))/7, mode='same')
    titleString = 'Median heading velocity'
    _ = movementSplitPolarAverageMaps(relPolMeanFig.add_subplot(gs[8]), relPolMeanFig.add_subplot(gs[9]),
                                      relPolMeanFig.add_subplot(gs[10]), relPolMeanFig.add_subplot(gs[11]), titleString,
                                      distEdges, fullAngleEdges, app, dep, d_gamma, objDistanceall, gammaFullall,
                                      useMean, maxVals[2])

    # 4) Corrected curvature
    turnSign = np.sign(polarCurv)
    turnSign[d_theta > 0] = np.sign(polarCurv[d_theta > 0])
    turnSign[d_theta < 0] = -np.sign(polarCurv[d_theta < 0])
    polarCurv = np.convolve(polarCurv, np.ones((5,))/5, mode='same')
    polarCurv[abs(polarCurv) > 2] = 2*np.sign(polarCurv[abs(polarCurv) > 2])
    curvmag = abs(np.convolve(polarCurv, np.ones((5,))/5, mode='same'))
    corCurv = turnSign*curvmag

    titleString = 'Median (corrected) curvature'
    _ = movementSplitPolarAverageMaps(relPolMeanFig.add_subplot(gs[12]), relPolMeanFig.add_subplot(gs[13]),
                                      relPolMeanFig.add_subplot(gs[14]), relPolMeanFig.add_subplot(gs[15]), titleString,
                                      distEdges, fullAngleEdges, app, dep, corCurv, objDistanceall, gammaFullall,
                                      useMean, maxVals[3])

    # 5) Coverage map
    titleString = 'Sampling coverage'
    nAll, xedAll, yedAll = np.histogram2d(objDistanceall, gammaFullall, bins=(distEdges, fullAngleEdges))
    nApr, xedApr, yedApr = np.histogram2d(objDistanceall[app], gammaFullall[app], bins=(distEdges, fullAngleEdges))
    nDep, xedDep, yedDep = np.histogram2d(objDistanceall[dep], gammaFullall[dep], bins=(distEdges, fullAngleEdges))

    allCov = relPolMeanFig.add_subplot(gs[16])
    X, Y = np.meshgrid(yedAll, xedAll)
    allCov.pcolormesh(Y, X, nAll, cmap='copper_r', vmin=0, vmax=np.max(nAll))
    allCov.set_ylim(min(yedAll), max(yedAll))
    allCov.set_xlim(min(xedAll), max(xedAll))
    allCov.set_title('Approaches')
    allCov.set_xlabel('radial distance from object [mm]')
    allCov.set_ylabel('heading angle relative to object [deg]')
    myAxisTheme(allCov)

    aprCov = relPolMeanFig.add_subplot(gs[17])
    X, Y = np.meshgrid(yedApr, xedApr)
    aprCov.pcolormesh(Y, X, nApr, cmap='copper_r', vmin=0, vmax=np.max(nAll))
    aprCov.set_ylim(min(yedApr), max(yedApr))
    aprCov.set_xlim(min(xedApr), max(xedApr))
    aprCov.set_title('Approaches')
    aprCov.set_xlabel('radial distance from object [mm]')
    myAxisTheme(aprCov)

    depCov = relPolMeanFig.add_subplot(gs[18])
    X, Y = np.meshgrid(yedDep, xedDep)
    pc = depCov.pcolormesh(Y, X, nDep, cmap='copper_r', vmin=0, vmax=np.max(nAll))
    depCov.set_ylim(min(yedDep), max(yedDep))
    depCov.set_xlim(min(xedDep), max(xedDep))
    depCov.set_title('Departures')
    aprCov.set_xlabel('radial distance from object [mm]')
    myAxisTheme(depCov)

    cb = plt.colorbar(pc, cax=relPolMeanFig.add_subplot(gs[19]))
    cb.ax.set_ylabel(titleString, rotation=270, fontsize=12)

    return relPolMeanFig
