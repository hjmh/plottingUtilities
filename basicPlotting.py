"""
Frequently used themes, plots and subplots for visualising fly walking traces.
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import matplotlib.pyplot as plt
from os import mkdir
import seaborn as sns

sns.set_style('ticks')


def myAxisTheme(myax):
    myax.get_xaxis().tick_bottom()
    myax.get_yaxis().tick_left()
    myax.spines['top'].set_visible(False)
    myax.spines['right'].set_visible(False)
    
    
def timeAxisTheme(tax):
    tax.spines['top'].set_visible(False)
    tax.spines['right'].set_visible(False)
    tax.spines['left'].set_visible(False)
    tax.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelleft='off')


def niceScatterPlot(pltAx, xpts, ypts, xlimRange, ylimRange, colorVal, alphaVal):
    pltAx.scatter(xpts, ypts, color=colorVal, alpha=alphaVal)
    plt.xlim(xlimRange)
    plt.ylim(ylimRange)
    myAxisTheme(pltAx)

    return pltAx


def plotSparseMatrix(figsize, aspectRatio, matrixToPlot, titleString):

    fig = plt.figure(figsize=figsize)
    fig.set_canvas(plt.gcf().canvas)
    sns.set_style('ticks')
    ax = fig.add_subplot(111)

    ax.spy(matrixToPlot)
    ax.set_aspect(aspectRatio)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('top')
    ax.set_title(titleString)
    sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': '.8', })
    fig.tight_layout()

    return fig


def makeNestedPlotDirectory(rootAnalysisDir, subdir1, subdir2):
    try:
        mkdir(rootAnalysisDir + subdir1)
    except OSError:
        print('Analysis directory already exists.')
    try:
        mkdir(rootAnalysisDir + subdir1 + subdir2)
    except OSError:
        print('Plot directory already exists.')
