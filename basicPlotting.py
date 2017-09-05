"""
Frequently used themes, plots and subplots for visualising fly walking traces.
"""

__author__ = 'Hannah Haberkern, hjmhaberkern@gmail.com'

import matplotlib.pyplot as plt
from os import mkdir
import seaborn as sns

sns.set_style('ticks')

# custom colormap data based on Accent
#       colormaps can be generated from this data in the following way:
#       mycmap = clr.LinearSegmentedColormap('myAccent',mycdict, N=256, gamma=1.0)
#       mycmap = clr.LinearSegmentedColormap.from_list('myAccent',mycollist, N=256, gamma=1.0)
# The list data generates slightly lower dynamic range colormap, more similar to the original Accent cmap.

_myAccentList = [(0.49803921580314636, 0.78823530673980713, 0.49803921580314636, 1.0),
                 (0.7450980544090271, 0.68235296010971069, 0.83137255907058716, 1.0),
                 (0.99215686321258545, 0.75294119119644165, 0.52549022436141968, 1.0),
                 (1.0, 1.0, 0.60000002384185791, 1.0),
                 (0.21960784494876862, 0.42352941632270813, 0.69019609689712524, 1.0),
                 (0.94117647409439087, 0.0078431377187371254, 0.49803921580314636, 1.0),
                 (0.74901962280273438, 0.35686275362968445, 0.090196080505847931, 1.0),
                 (0.40000000596046448, 0.40000000596046448, 0.40000000596046448, 1.0)]

_myAccentDict = {
    'blue': [(0.0,   0.0,                    0.0),
             (0.125, 0.49803921580314636,    0.49803921580314636),
             (0.25,  0.83137255907058716,    0.83137255907058716),
             (0.375, 0.52549022436141968,    0.52549022436141968),
             (0.5,   0.60000002384185791,    0.60000002384185791),
             (0.625, 0.69019609689712524,    0.69019609689712524),
             (0.75,  0.49803921580314636,    0.49803921580314636),
             (0.875, 0.090196080505847931,   0.090196080505847931),
             (1.0,   0.40000000596046448,    0.40000000596046448),],
        
    'green':[(0.0,   0.0,                    0.0),
             (0.125, 0.78823530673980713,    0.78823530673980713),
             (0.25,  0.68235296010971069,    0.68235296010971069),
             (0.375, 0.75294119119644165,    0.75294119119644165),
             (0.5,   1.0,                    1.0),
             (0.625, 0.42352941632270813,    0.42352941632270813),
             (0.75,  0.0078431377187371254,  0.0078431377187371254),
             (0.875, 0.35686275362968445,    0.35686275362968445),
             (1.0,   0.40000000596046448,    0.40000000596046448),],
             
    'red':  [(0.0,   0.0,                    0.0),
             (0.125, 0.49803921580314636,    0.49803921580314636),
             (0.25,  0.7450980544090271,     0.7450980544090271),
             (0.375, 0.99215686321258545,    0.99215686321258545),
             (0.5,   1.0,                    1.0),
             (0.625, 0.21960784494876862,    0.21960784494876862),
             (0.75,  0.94117647409439087,    0.94117647409439087),
             (0.875, 0.74901962280273438,    0.74901962280273438),
             (1.0,   0.40000000596046448,    0.40000000596046448),],
}

# axis beautification

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

# some simple basic plots

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

# make directories for plots

def makeNestedPlotDirectory(rootAnalysisDir, subdir1, subdir2):
    try:
        mkdir(rootAnalysisDir + subdir1)
    except OSError:
        print('Analysis directory already exists.')
    try:
        mkdir(rootAnalysisDir + subdir1 + subdir2)
    except OSError:
        print('Plot directory already exists.')
