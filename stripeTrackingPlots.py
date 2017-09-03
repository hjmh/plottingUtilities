"""
Specialised plotting functions for visualising stripe tracking assay data
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from basicPlotting import myAxisTheme


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


def plotHeadingComparison(sceneName, titleString, densityFlag, plotIQR, flyIDs, FOAllFlies_df,
                          keyind_gam, keyind_gamful, keyind_mov):
    legendlist = []
    halfBins = 18
    fullBins = 36

    numFlies = len(flyIDs)

    nhead_halfGamma = np.nan*np.ones((numFlies, halfBins, 2))
    nhead_fullGamma = np.nan*np.ones((numFlies, fullBins, 2))

    headingfig = plt.figure(figsize=(10, 8))

    cNorm  = colors.Normalize(vmin=0, vmax=numFlies)
    flyCMap = plt.cm.ScalarMappable(norm=cNorm, cmap='Accent')

    gammaPlt = headingfig.add_subplot(221)
    gammaFullPlt = headingfig.add_subplot(222)
    gammaPlt2 = headingfig.add_subplot(223)
    gammaFullPlt2 = headingfig.add_subplot(224)

    for fly in range(numFlies):
        querystring = '(sceneName=="' + sceneName + '") & (flyID =="' + flyIDs[fly] + '")'

        gamma = FOAllFlies_df.query(querystring).iloc[:, keyind_gam:keyind_gam+1].squeeze()
        gammaFull = FOAllFlies_df.query(querystring).iloc[:, keyind_gamful:keyind_gamful+1].squeeze()

        moving = FOAllFlies_df.query(querystring).iloc[:, keyind_mov:keyind_mov+1].squeeze()

        if sum(moving) <= 0.2*len(moving):
            print('fly ' + str(flyIDs[fly])+' not moving: ' + str(100.0*sum(moving)/max(1, len(moving))))
            continue

        legendlist.append(flyIDs[fly])

        if densityFlag:
            ylab = 'frequency (when moving)'
        else:
            ylab = 'count (when moving)'

        gammaPlt, normgam, halfedges = plotHeadingDistFromTimeseries(gammaPlt, gamma[moving > 0], densityFlag,
                                                                     (0, np.pi), halfBins, flyCMap.to_rgba(fly),
                                                                     'rel. heading', ylab, 1)
        nhead_halfGamma[fly, :, 0] = normgam

        gammaFullPlt, normgam, fulledges = plotHeadingDistFromTimeseries(gammaFullPlt, gammaFull[moving > 0],
                                                                         densityFlag, (-np.pi, np.pi), fullBins,
                                                                         flyCMap.to_rgba(fly), 'rel. heading (full)',
                                                                         ylab, 1)
        nhead_fullGamma[fly, :, 0] = normgam

        if densityFlag:
            ylab = 'frequency (when standing)'
        else:
            ylab = 'count (when standing)'

        gammaPlt2, normgam, halfedges = plotHeadingDistFromTimeseries(gammaPlt2, gamma[moving == 0], densityFlag,
                                                                     (0, np.pi), halfBins, flyCMap.to_rgba(fly),
                                                                     'rel. heading', ylab, 0.6)
        nhead_halfGamma[fly, :, 1] = normgam

        gammaFullPlt2, normgam, fulledges = plotHeadingDistFromTimeseries(gammaFullPlt2, gammaFull[moving == 0],
                                                                          densityFlag, (-np.pi, np.pi), fullBins,
                                                                          flyCMap.to_rgba(fly), 'rel. heading (full)',
                                                                          ylab, 1)
        nhead_fullGamma[fly, :, 1] = normgam

    headingfig.suptitle(titleString, fontsize=13)
    gammaFullPlt2.legend(legendlist)
    headingfig.tight_layout()

    gammaPlt.plot(halfedges, np.nanmedian(nhead_halfGamma[:, :, 0], 0), color='k', linewidth=3)
    gammaFullPlt.plot(fulledges, np.nanmedian(nhead_fullGamma[:, :, 0], 0), color='k', linewidth=3)

    gammaPlt2.plot(halfedges, np.nanmedian(nhead_halfGamma[:, :, 1], 0), color='k', alpha=0.6, linewidth=3)
    gammaFullPlt2.plot(fulledges, np.nanmedian(nhead_fullGamma[:, :, 1], 0), color='k', alpha=0.6, linewidth=3)

    if plotIQR:
        [var1, var2] = np.nanpercentile(nhead_halfGamma[:, :, 0], [25, 75], axis=0)
        gammaPlt.fill_between(halfedges, var1, var2, color='k', alpha=0.2)

        [var1, var2] = np.nanpercentile(nhead_fullGamma[:, :, 0], [25, 75], axis=0)
        gammaFullPlt.fill_between(fulledges, var1, var2, color='k', alpha=0.2)

        [var1, var2] = np.nanpercentile(nhead_halfGamma[:, :, 1], [25, 75], axis=0)
        gammaPlt2.fill_between(halfedges, var1, var2, color='k', alpha=0.2)

        [var1, var2] = np.nanpercentile(nhead_fullGamma[:, :, 1], [25, 75], axis=0)
        gammaFullPlt2.fill_between(fulledges, var1, var2, color='k', alpha=0.2)

    return headingfig
