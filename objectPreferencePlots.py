"""
Plots visualising relative preferences for certain landmarks in a multi-landmark VR
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import numpy as np
import matplotlib.pyplot as plt

def prettyBoxPlot(bpPlt, myBoxCols, objtype, myObjVals, flyIDs, offsets, plotLabels):
    # myObjVals should be e.g. VisitCount[objtype], myBoxCols should be objBoxColors[objtype]

    boxs = bpPlt.boxplot(myObjVals, patch_artist=True)
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
        else:
            trialOffSets = np.vstack((np.vstack((0 + jitter[fly] + offsets[0], 1 + jitter[fly] + offsets[1])),
                                      2 + jitter[fly] + offsets[2]))
        bpPlt.plot(trialOffSets, myObjVals[fly, :], '-', color='grey', linewidth=0.5, alpha=linealpha)

        if plotLabels:
            bpPlt.text(len(offsets) - .91 + jitter[fly] + offsets[-1], myObjVals[fly, -1], flyIDs[fly])

    if len(offsets) < 3:
        plt.xticks(range(1, len(trialName)), [trialName[0], trialName[2]])
    else:
        plt.xticks(range(1, len(trialName) + 1), trialName)
    bpPlt.axhline(y=0, linewidth=1, color='grey', linestyle='dashed')
    bpPlt.set_ylim((-0.1 * np.max(myObjVals), (0.1 * np.max(myObjVals)) + np.max(myObjVals)))
    myAxisTheme(bpPlt)

    return bpPlt


def diffCorrPlot(bpPlt, prePostDiffVals, flyIDs, dotcol):
    from scipy.stats.stats import pearsonr

    bpPlt.scatter(prePostDiffVals[:, 0], prePostDiffVals[:, 1], s=30, facecolor=dotcol)
    for fly in range(len(flyIDs)):
        bpPlt.text(prePostDiffVals[fly, 0] + 0.03, prePostDiffVals[fly, 1] + 0.03, flyIDs[fly])

    rsq, p = pearsonr(prePostDiffVals[:, 0], prePostDiffVals[:, 1])
    bpPlt.set_xlabel('Delta pre')
    bpPlt.set_ylabel('Delta post')
    bpPlt.axhline(y=0, linewidth=1, color='grey', linestyle='dashed')
    bpPlt.axvline(x=0, linewidth=1, color='grey', linestyle='dashed')
    minplt = -0.5 + np.min(prePostDiffVals)
    maxplt = 0.5 + np.max(prePostDiffVals)
    bpPlt.plot([minplt, maxplt], [minplt, maxplt], linewidth=1, color=dotcol, alpha=0.3)
    bpPlt.set_title('correlation: PCC=' + str(round(rsq, 4)) + ', p=' + str(round(p, 4)))
    bpPlt.set_xlim((minplt, maxplt))
    bpPlt.set_ylim((minplt, maxplt))
    myAxisTheme(bpPlt)

    return bpPlt