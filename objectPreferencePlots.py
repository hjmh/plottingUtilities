"""
Plots visualising relative preferences for certain landmarks in a multi-landmark VR
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np

from os import getcwd
from os.path import sep
from sys import path

import matplotlib.pyplot as plt

# Set path to analysis code directory
codeDir = sep.join(getcwd().split(sep)[:-2])
path.insert(1, codeDir)

from basicPlotting import myAxisTheme


def countvisits(dist2Obj, visitRad):
    inside = (dist2Obj < visitRad).astype('int')
    time = np.linspace(0, 600, len(dist2Obj))

    entries = np.zeros(len(inside))
    entries[1:] = np.diff(inside) == 1

    exits = np.zeros(len(inside))
    exits[1:] = np.diff(inside) == -1

    # check if no entries and/or no exits
    if len(inside) == 0 or sum(np.diff(inside) == -1) < 1:
        visitT = entryTime = exitTime = np.nan
        return entries, exits, visitT, entryTime, exitTime

    entryTime = time[entries.astype('bool')]
    exitTime = time[exits.astype('bool')]

    if len(entryTime) == len(exitTime):
        visitT = exitTime - entryTime
    else:
        visitT = exitTime[0:min(sum(exits), sum(entries)).astype('int')] - entryTime[
                                                                           0:min(sum(exits), sum(entries)).astype(
                                                                               'int')]

    return entries, exits, visitT, entryTime, exitTime


def prettyBoxPlot(bpPlt, myBoxCols, boxalpha, linealpha, myObjVals, flyIDs, offsets, trialName, plotLabels):
    # myObjVals should be e.g. VisitCount[objtype], myBoxCols should be objBoxColors[objtype]

    mask = ~np.isnan(myObjVals)
    filt_myObjVals = [d[m] for d, m in zip(myObjVals.T, mask.T)]

    boxs = bpPlt.boxplot(filt_myObjVals, patch_artist=True)
    plt.setp(boxs['whiskers'], color='black', linestyle='-')
    plt.setp(boxs['medians'], color='black', linewidth=2)
    plt.setp(boxs['fliers'], color='grey', marker='+')

    jitter = np.random.normal(0, 0.012 * len(offsets), size=len(myObjVals[:, 0]))

    for ind, box in enumerate(boxs['boxes']):
        plt.setp(box, color=myBoxCols[ind], linewidth=1.5, alpha=boxalpha)
        boxprops = dict(linestyle='-', linewidth=1.5, color='grey')

        # Add some random "jitter" to the x-axis
        x = ind + jitter + offsets[ind]
        bpPlt.plot(x, myObjVals[:, ind], 'o', color=myBoxCols[ind], alpha=0.8)

    for fly in range(len(flyIDs)):
        if len(offsets) < 3:
            trialOffSets = np.vstack((0 + jitter[fly] + offsets[0], 1 + jitter[fly] + offsets[1]))
        elif len(offsets) == 3:
            trialOffSets = np.vstack((np.vstack((0 + jitter[fly] + offsets[0], 1 + jitter[fly] + offsets[1])),
                                      2 + jitter[fly] + offsets[2]))
        else:
            trialOffSets = np.arange(len(offsets)) + offsets + jitter[fly]

        bpPlt.plot(trialOffSets, myObjVals[fly, :], '-', color='grey', linewidth=0.5, alpha=linealpha)

        if plotLabels:
            bpPlt.text(len(offsets) - .91 + jitter[fly] + offsets[-1], myObjVals[fly, -1], flyIDs[fly])

    if len(offsets) < 3:
        plt.xticks(range(1, len(trialName)), [trialName[0], trialName[2]])
    else:
        plt.xticks(range(1, len(trialName) + 1), trialName)
    bpPlt.axhline(y=0, linewidth=1, color='grey', linestyle='dashed')
    bpPlt.set_ylim((-0.1 * np.nanmax(myObjVals), (0.1 * np.nanmax(myObjVals)) + np.nanmax(myObjVals)))
    myAxisTheme(bpPlt)

    return bpPlt


def simpleBoxPlot(bpPlt, myBoxCols, boxalpha, linealpha, myObjVals, flyIDs, trialName):
    boxs = bpPlt.boxplot(myObjVals, patch_artist=True)
    plt.setp(boxs['whiskers'], color='black', linestyle='-')
    plt.setp(boxs['medians'], color='black', linewidth=2)
    plt.setp(boxs['fliers'], color='grey', marker='+')

    for ind, box in enumerate(boxs['boxes']):
        plt.setp(box, color=myBoxCols[ind], linewidth=1.5, alpha=boxalpha)
        boxprops = dict(linestyle='-', linewidth=1.5, color='grey')

    for fly in range(len(flyIDs)):
        bpPlt.plot(np.arange(len(trialName)) + 1, myObjVals[fly, :], '-', color='grey', linewidth=0.5, alpha=linealpha)

    bpPlt.set_xticklabels(trialName)
    bpPlt.axhline(y=0, linewidth=1, color='grey', linestyle='dashed')
    bpPlt.set_ylim((-0.1 * np.nanmax(myObjVals), (0.1 * np.nanmax(myObjVals)) + np.nanmax(myObjVals)))
    myAxisTheme(bpPlt)

    return bpPlt


def diffCorrPlot(bpPlt, prePostDiffVals, flyIDs, dotcol):
    from scipy.stats.stats import pearsonr

    bpPlt.scatter(prePostDiffVals[:, 0], prePostDiffVals[:, 1], s=30, facecolor=dotcol)

    if np.sum(np.isnan(VisitLength[0])) == 0:
        for fly in range(len(flyIDs)):
            bpPlt.text(prePostDiffVals[fly, 0] + 0.03, prePostDiffVals[fly, 1] + 0.03, flyIDs[fly])

            # rsq, p = pearsonr(prePostDiffVals[:, 0], prePostDiffVals[:, 1])
    bpPlt.set_xlabel('Delta pre')
    bpPlt.set_ylabel('Delta post')
    bpPlt.axhline(y=0, linewidth=1, color='grey', linestyle='dashed')
    bpPlt.axvline(x=0, linewidth=1, color='grey', linestyle='dashed')
    minplt = -0.5 + np.nanmin(prePostDiffVals)
    maxplt = 0.5 + np.nanmax(prePostDiffVals)
    bpPlt.plot([minplt, maxplt], [minplt, maxplt], linewidth=1, color=dotcol, alpha=0.3)
    # bpPlt.set_title('correlation: PCC=' + str(round(rsq, 4)) + ', p=' + str(round(p, 4)))
    bpPlt.set_xlim((minplt, maxplt))
    bpPlt.set_ylim((minplt, maxplt))
    myAxisTheme(bpPlt)

    return bpPlt