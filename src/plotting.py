from skimage.transform import resize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import AlignedSOM

#HitHistogram
def HitHist(_m, _n, _weights, _idata, upscaling_factor=1000):
    hist = np.zeros(_m * _n)
    for vector in _idata: 
        position =np.argmin(np.sqrt(np.sum(np.power(_weights - vector, 2), axis=1)))
        hist[position] += 1
    
    hist_reshaped = hist.reshape(_m, _n)
    return resize(hist_reshaped, (upscaling_factor, upscaling_factor), mode='constant')

#U-Matrix - implementation
def UMatrix(_m, _n, _weights, _dim, upscaling_factor=1000):
    U = _weights.reshape(_m, _n, _dim)
    U = np.insert(U, np.arange(1, _n), values=0, axis=1)
    U = np.insert(U, np.arange(1, _m), values=0, axis=0)
    #calculate interpolation
    for i in range(U.shape[0]): 
        if i%2==0:
            for j in range(1,U.shape[1],2):
                U[i,j][0] = np.linalg.norm(U[i,j-1] - U[i,j+1], axis=-1)
        else:
            for j in range(U.shape[1]):
                if j%2==0: 
                    U[i,j][0] = np.linalg.norm(U[i-1,j] - U[i+1,j], axis=-1)
                else:      
                    U[i,j][0] = (np.linalg.norm(U[i-1,j-1] - U[i+1,j+1], axis=-1) + np.linalg.norm(U[i+1,j-1] - U[i-1,j+1], axis=-1))/(2*np.sqrt(2))

    U = np.sum(U, axis=2) #move from Vector to Scalar

    for i in range(0, U.shape[0], 2): #count new values
        for j in range(0, U.shape[1], 2):
            region = []
            if j>0: region.append(U[i][j-1]) #check left border
            if i>0: region.append(U[i-1][j]) #check bottom
            if j<U.shape[1]-1: region.append(U[i][j+1]) #check right border
            if i<U.shape[0]-1: region.append(U[i+1][j]) #check upper border

            U[i,j] = np.median(region)
    return resize(U, (upscaling_factor, upscaling_factor), mode='constant')
    #return U


#SDH - implementation
def SDH(_m, _n, _weights, _idata, factor=2, approach=0, upscaling_factor=1000):
    import heapq

    sdh_m = np.zeros( _m * _n)

    cs=0
    for i in range(factor): cs += factor-i

    for vector in _idata:
        dist = np.sqrt(np.sum(np.power(_weights - vector, 2), axis=1))
        c = heapq.nsmallest(factor, range(len(dist)), key=dist.__getitem__)
        if (approach==0): # normalized
            for j in range(factor):  sdh_m[c[j]] += (factor-j)/cs 
        if (approach==1):# based on distance
            for j in range(factor): sdh_m[c[j]] += 1.0/dist[c[j]] 
        if (approach==2): 
            dmin, dmax = min(dist[c]), max(dist[c])
            for j in range(factor): sdh_m[c[j]] += 1.0 - (dist[c[j]]-dmin)/(dmax-dmin)
    
    sdh_m = sdh_m.reshape(_m, _n)
    return resize(sdh_m, (upscaling_factor, upscaling_factor), mode='constant')

def plot_aligned_som(asom: AlignedSom, data:np.ndarray, visualization_function=SDH, num_plots=5, **kwargs):
    """Plot the aligned SOM

    Args:
        asom (AlignedSom): trained aligned SOM to plot
        data (np.ndarray): input data to use for the visualization
        visualization_function (Callable, optional): Which visualization to use. Options are: SDH, HitHist and UMatrix. Defaults to SDH.
        num_plots (int, optional): How many intermediary plots to show. Defaults to 5.
        value_range (tuple, optional): Value range of the histogram given as tuple of min and max values. Defaults to (0,5).
        kwargs: Additional arguments to pass to the visualization function
    
    Returns:
        matplotlib figure: Figure object_
    """
    assert num_plots <= asom.num_layers, "Number of plots must be less than or equal to the number of layers"
    
    # calculate the histograms
    visualizations = []
    for layer_weights in asom.get_layer_weights():
        layer_weights = np.reshape(layer_weights, (asom.dimension[0] * asom.dimension[1], data.shape[1]))
        if visualization_function == UMatrix:
            histogram = visualization_function(asom.dimension[0], asom.dimension[1], layer_weights, data.shape[1], **kwargs)
        else:
            histogram = visualization_function(asom.dimension[0], asom.dimension[1], layer_weights, data, **kwargs)
        visualizations.append(histogram)
    
    # decrease figure size to increase plotting speed for larger plots
    if num_plots > 32:
        figsize = (0.75*num_plots, 0.6125)
    if num_plots > 16:
        figsize = (1.5*num_plots, 1.25)
    elif num_plots > 8:
        figsize = (3*num_plots, 2.5)
    else:
        figsize=(6*num_plots,5)
    
    max_value = np.max(np.array(visualizations))
    
    # create the plot
    figure, axis = plt.subplots(1, num_plots, figsize=figsize)
    for i, vis_i in enumerate(np.linspace(0, asom.num_layers - 1, num_plots, dtype=int)):
        hp = sns.heatmap(visualizations[vis_i], ax=axis[i], vmin=0, vmax=max_value, cbar=False, cmap='viridis')
        hp.set(xticklabels=[])
        hp.set(yticklabels=[])
        axis[i].tick_params(left=False, bottom=False)
        hp.set(xlabel=f"Weight Feature 1: {asom.weights_by_layer[vis_i][0]}")
    plt.show()
    return figure