
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
                
                cnx = cntroid[cIdx]
                cny = cntroid[rIdx]

                mrkCol = clustColors[clustIdx]
                edgCol = mrkCol if itNum + 1 == len(allCentroids) else edgeColor

                pltArgs = dict(c = mrkCol, edgecolor = edgCol, marker = 'X',
                               alpha = alpha, zorder = itNum + 1)

                axs.scatter(cnx,cny,**pltArgs)


class PyPlotScatterWraper():
    def __init__(self, x, y, **kwargs):

        assert x and y, 'No input data.'

        self._x = x
        self._y = y
        
        self._scatterArgs  = kwargs.pop('scatterArgs', {})
        self._errorbarArgs = kwargs.pop('errorbarArgs', {})
        self._pltFigArgs   = kwargs.pop('pltFigureArgs', {})
        self._pltFuncArgs  = kwargs.pop('pltFuncArgs', {})

        self._xerr = self._errorbarArgs.pop('xerr', [])
        self._yerr = self._errorbarArgs.pop('yerr', [])

        self._tightLayout = kwargs.pop('pltTightLayout', True)

        self._fileName = kwargs.pop('fileName', '')


    def __call__(self):

        import matplotlib.pyplot as plt

        # make figure
        if 'figsize' not  in self._pltFigArgs.keys():
            self._pltFigArgs.update( {'figsize':(9,6)} )
        fig = plt.figure(**self._pltFigArgs)

        # make scatter
        plt.scatter(self._x,self._y, **self._scatterArgs)

        # make error bars
        if self._xerr:
            self._errorbarArgs.update( {'xerr' : self._xerr} )
        if self._yerr:
            self._errorbarArgs.update( {'yerr' : self._yerr} )
        if 'xerr' in self._errorbarArgs.keys() or 'yerr' in self._errorbarArgs.keys(): 
            plt.errorbar(self._x, self._y, **self._errorbarArgs)

        # miscelaneous
        # here we simply parse pyplot function names and arguments 
        if self._pltFuncArgs:

            tightLayout = self._pltFuncArgs.pop('tight_layout', False)
            for funcName, allArgs in self._pltFuncArgs.items():
                
                if hasattr(plt,funcName): # check befora calling
                    func = getattr(plt,funcName)

                    if type(allArgs) == bool: # function calls with no arguments
                        func()
                    elif type(allArgs) == dict:

                        args   = allArgs.pop('args', [])
                        kwargs = allArgs.pop('kwargs', {})

                        if args:
                            args = [args]
                            func(*args, **kwargs)
                        else:
                            func(**kwargs)
                    else:
                        print('Cannot parse "%s" argument properly.'%funcName)
                else:
                    print('matplotlibpyplot does not have "%s" attribute'%funcName)

                if tightLayout: # always at the end
                    plt.tight_layout()
        # save
        if self._fileName:
            print('saving file %s'%self._fileName)
            fig.savefig(self._fileName)

            
class PyPlotHistWraper():
## TODO:: All of it :-P
    
    def __init__(self):
        hist, edg = histogram(dat[nam], density=False, weights=weights)
        # hist = hist/float(hist.sum())

               
        xs = [edg[i] + (edg[i+1]-edg[i])/2. for i in range(len(hist)) ]
        er = [sqrt(i) for i in hist]

        for x,y,e in zip(xs,hist,er):
            print(x,y,sqrt(y),e)
            # print (edg)
            # print (xs)
            # print(hist)

            axs.errorbar(xs,hist,yerr=er, fmt='none', ecolor='black')
            import pdb; pdb.set_trace()
            # print(h)

            
def voronoiPlot(data,nClusters):
    print ("Voronoi plot is not completelly understood in the current implemetation. ")
    ## TODO:: Allow for more flexibility, with more rich data.
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as  pyplot
    import numpy as np
    
    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=nClusters, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    fig = pyplot.figure()
    pyplot.clf()
    pyplot.imshow(Z, interpolation='nearest',
                  extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                  cmap=pyplot.cm.Paired,
                  aspect='auto', origin='lower')

    pyplot.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    pyplot.scatter(centroids[:, 0], centroids[:, 1],
                   marker='x', s=169, linewidths=3,
                   color='w', zorder=10)
    # pyplot.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
    #              'Centroids are marked with white cross')
    pyplot.xlim(x_min, x_max)
    pyplot.ylim(y_min, y_max)
    pyplot.xticks(())
    pyplot.yticks(())

    print (x_min,x_max)
    print (y_min,y_max)
       
    pyplot.tight_layout()
    pyplot.show()

    
def histPointsFromAxis(ax, binCenters = True):

    bnHeight, bnEdges = [],[]
    for rect in ax.patches:
        ((x0, y0), (x1, y1)) = rect.get_bbox().get_points()
        bnHeight.append(y1-y0)
        bnEdges.append(x0) # left edge of each bin
    bnEdges.append(x1) # also get right edge of last bin

    if binCenters:
        bnCntrs = []
        for idx, edg in enumerate(bnEdges[:-1]): 
            low = bnEdges[idx]
            hig = bnEdges[idx+1]
            bnCntrs += [low + float(hig-low)/2. ] 
        bnEdges = bnCntrs

    return bnHeight, bnEdges
