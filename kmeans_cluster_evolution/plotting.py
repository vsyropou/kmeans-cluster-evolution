
def SeabornPairGridWrapper(*args, **kwargs):
    # aprse args
    data = args[0]
    lowFunc = args[1]
    uppFunc = args[2]
    diaFunc = args[3]

    pairGridArgs = kwargs.pop('pairGridKwargs',{})
    mapLowerArgs = kwargs.pop('mapLowerKwargs',{})
    mapUpperArgs = kwargs.pop('mapUpperKwargs',{})
    mapDiagArgs  = kwargs.pop('mapDiagKwargs',{})

    # make pairgrid and plot
    import seaborn as sns
    plot = sns.PairGrid(data, **pairGridArgs)

    # fill grid
    plot.map_diag(diaFunc, **mapDiagArgs)
    plot.map_lower(lowFunc, **mapLowerArgs)
    plot.map_upper(uppFunc, **mapUpperArgs)

    return plot


def kMeansClusterEvolutionOnPairGrid(axes, allCentroids, **kwargs):

    clustColors = kwargs.pop('clustColors', {} )
    edgeColor   = kwargs.pop('edgeColor', 'yellow')
    markerStyle = kwargs.pop('markerStyle', 'X')
    alpha       = kwargs.pop('alpha', 1)
    nClusters   = kwargs.pop('nClusters', len(allCentroids[0]))

    ftureNames  = kwargs.pop('featureNames', [])
    saveSubPlot = kwargs.pop('saveSubPlot', False)
    subPlotSufx = kwargs.pop('subPlotSufx', '')

    saveall = kwargs.pop('saveall', True)

    if saveSubPlot:
        assert ftureNames, 'SubPlots will be overwritten if featureNames is not specified.'

    # plot cluster centroids
    gridRank = len(axes)
    upprAxes = [(i,j) for i in range(gridRank) for j in range(gridRank) if i<j]
    lowrAxes = [(i,j) for i in range(gridRank) for j in range(gridRank) if i>j]

    for itNum, centroids in enumerate(allCentroids):

        for clustIdx in range(nClusters):

            cntroid = centroids[clustIdx]

            for (rIdx,cIdx) in upprAxes + lowrAxes:

                axs = axes[rIdx][cIdx]
                #import pdb; pdb.set_trace()                
                cnx = cntroid[cIdx]
                cny = cntroid[rIdx]

                mrkCol = clustColors[clustIdx]
                edgCol = edgeColor if itNum + 1 == len(allCentroids) else mrkCol
                pltArgs = dict(c = mrkCol, edgecolor = edgCol, marker = 'X',
                               alpha = alpha, zorder = itNum + 1)

                axs.scatter(cnx,cny,**pltArgs)

        if saveall:
            axs.figure.savefig('cluster_evolution_iter_%s.pdf'%itNum)
                
