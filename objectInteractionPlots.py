"""
Plots that characterise the interaction of the fly with an object.
For example, the effect of the realative heading angle and the distance of the object on turns and walking velocity.
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns

def myAxisTheme(myax):
    myax.get_xaxis().tick_bottom()
    myax.get_yaxis().tick_left()
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)


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


# 2D (cartesian) residency histograms ..................................................................................

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


# Heading angle distribution plots (e.g. for stripe tracking) ..........................................................

def plotHeadingComparison(FOAllFlies_df, flyIDs, sceneName, titleString, keyind_gam, keyind_gamful, keyind_mov, flyCMap,
                          densityFlag, plotIQR):
    legendlist = []
    halfBins = 18
    fullBins = 36

    headingfig = plt.figure(figsize=(10, 8))

    numFlies = len(flyIDs)
    nhead_halfGamma = np.nan*np.ones((numFlies, halfBins, 2))
    nhead_fullGamma = np.nan*np.ones((numFlies, fullBins, 2))

    for fly in range(numFlies):
        querystring = '(sceneName=="' + sceneName + '") & (flyID =="' + flyIDs[fly] + '")'

        gamma = FOAllFlies_df.query(querystring).iloc[:, keyind_gam:keyind_gam+1].squeeze()
        gammaFull = FOAllFlies_df.query(querystring).iloc[:, keyind_gamful:keyind_gamful+1].squeeze()

        moving = FOAllFlies_df.query(querystring).iloc[:, keyind_mov:keyind_mov+1].squeeze()

        if sum(moving) <= 0.2*len(moving):
            print('fly '+str(flyIDs[fly])+' not moving')
            print(100.0*sum(moving)/max(1, len(moving)))
            continue

        legendlist.append(flyIDs[fly])

        gammaPlt = headingfig.add_subplot(221)
        histRange = (0, np.pi)
        nhead, edges = np.histogram(gamma[moving > 0], normed=densityFlag, density=densityFlag, range=histRange,
                                    bins=halfBins)
        gammaPlt.set_xlabel('rel. heading')
        if densityFlag:
            gammaPlt.set_ylabel('frequency (when moving)')
            normFactor = nhead.sum()
        else:
            gammaPlt.set_ylabel('count (when moving)')
            normFactor = 1.0
        gammaPlt.plot(edges[:-1]+np.diff(edges)/2, nhead/normFactor, color=flyCMap.to_rgba(fly))
        gammaPlt.set_xlim(histRange)
        myAxisTheme(gammaPlt)

        nhead_halfGamma[fly, :, 0] = nhead/normFactor
        halfedges = edges[:-1]+np.diff(edges)/2

        gammaFullPlt = headingfig.add_subplot(222)
        histRange = (-np.pi, np.pi)
        nhead, edges = np.histogram(gammaFull[moving > 0], normed=densityFlag, density=densityFlag, range=histRange,
                                    bins=fullBins)
        if densityFlag:
            normFactor = nhead.sum()
        else:
            normFactor = 1.0
        gammaFullPlt.plot(edges[:-1]+np.diff(edges)/2,nhead/normFactor,color=flyCMap.to_rgba(fly))
        gammaFullPlt.set_xlim(histRange)
        gammaFullPlt.set_xlabel('rel. heading (full)')
        myAxisTheme(gammaFullPlt)

        nhead_fullGamma[fly, :, 0] = nhead/normFactor
        fulledges = edges[:-1]+np.diff(edges)/2

        gammaPlt2 = headingfig.add_subplot(223)
        histRange = (0, np.pi)
        nhead, edges = np.histogram(gamma[moving == 0], normed=densityFlag, density=densityFlag, range=histRange,
                                    bins=halfBins)
        gammaPlt2.set_xlabel('rel. heading')
        if densityFlag:
            gammaPlt2.set_ylabel('frequency (when standing)')
            normFactor = nhead.sum()
        else:
            gammaPlt2.set_ylabel('count (when standing)')
            normFactor = 1.0
        gammaPlt2.plot(edges[:-1]+np.diff(edges)/2, nhead/normFactor, color=flyCMap.to_rgba(fly), alpha=0.6)
        gammaPlt2.set_xlim(histRange)
        myAxisTheme(gammaPlt2)

        nhead_halfGamma[fly, :, 1] = nhead/normFactor

        gammaFullPlt2 = headingfig.add_subplot(224)
        histRange = (-np.pi, np.pi)
        nhead, edges = np.histogram(gammaFull[moving == 0], normed=densityFlag, density=densityFlag, range=histRange,
                                    bins=fullBins)
        gammaFullPlt2.plot(edges[:-1]+np.diff(edges)/2, nhead/normFactor, color=flyCMap.to_rgba(fly), alpha=0.6)
        gammaFullPlt2.set_xlim(histRange)
        gammaFullPlt2.set_xlabel('rel. heading (full)')
        myAxisTheme(gammaFullPlt2)

        nhead_fullGamma[fly, :, 1] = nhead/normFactor

    headingfig.suptitle(titleString,fontsize=13)
    gammaFullPlt2.legend(legendlist)
    headingfig.tight_layout()

    gammaPlt.plot(halfedges, np.nanmedian(nhead_halfGamma[:, :, 0], 0), color='k', linewidth=3)
    gammaFullPlt.plot(fulledges, np.nanmedian(nhead_fullGamma[:, :, 0], 0), color='k', linewidth=3)

    gammaPlt2.plot(halfedges, np.nanmedian(nhead_halfGamma[:, :, 1], 0), color='k', alpha=0.6, linewidth=3)
    gammaFullPlt2.plot(fulledges, np.nanmedian(nhead_fullGamma[:, :, 1], 0), color='k', alpha=0.6, linewidth=3)

    if(plotIQR):
        [var1, var2] = np.nanpercentile(nhead_halfGamma[:, :, 0], [25, 75], axis=0)
        gammaPlt.fill_between(halfedges, var1, var2, color='k', alpha=0.2)

        [var1, var2] = np.nanpercentile(nhead_fullGamma[:, :, 0], [25, 75], axis=0)
        gammaFullPlt.fill_between(fulledges, var1, var2, color='k', alpha=0.2)

        [var1, var2] = np.nanpercentile(nhead_halfGamma[:, :, 1], [25, 75], axis=0)
        gammaPlt2.fill_between(halfedges, var1, var2, color='k', alpha=0.2)

        [var1, var2] = np.nanpercentile(nhead_fullGamma[:, :, 1], [25, 75], axis=0)
        gammaFullPlt2.fill_between(fulledges, var1, var2, color='k', alpha=0.2)

    return headingfig, nhead_fullGamma[:, :, 0]


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
