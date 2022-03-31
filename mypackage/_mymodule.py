import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
import pandas as pd
import hotspot
import base64
from io import BytesIO


def automatic_proportion_threshold(
    st_adata, 
    kind_threshold='primary',
    output_file='threshold.html',
    ct_list=None):
    """
    Function to compute automatic threshold on cell type proportion values.
    For further reference check [Lopez22].

    Parameters
    ----------
    st_adata
        Spatial sequencing dataset with proportions in obsm['proportions'] and spatial location in
        obsm['location']
    kind_threshold
        Which threshold value to use. Supported are 'primary', 'secondary', 'min_value'.
        'min_value' uses the minimum of primary and secondary threshold for each cell type.
    output_file
        File where html output is stored. Defaults to 'threshold.html'
    ct_list
        Celltypes to use. Defaults to all celltypes.

    Returns
    -------
    ct_thresholds
        Dictionary containing all threshold values.
    
    """

    ct_thresholds = {}
    nominal_threshold = {}

    html = "<h2>Automatic thresholding</h2>"

    for name_ct in ct_list:
        array = st_adata.obsm["proportions"][name_ct]
        vmax = np.quantile(array.values, 0.99)

        # get characteristic values
        quantiles, stack = _form_stacked_quantiles(array.values)
        index, z_values = _get_autocorrelations(st_adata, stack, quantiles)
        z_values, smoothed, derivative, sign_2nd, _ = _smooth_get_critical_points(index, z_values, s=0.1)
        ipoints = index[np.where(sign_2nd[:-1] != sign_2nd[1:])[0]]
        nom_map = index[np.argmin(derivative)]
        
        #PLOT 1 shows proportions in spatial dimensions without thresholding
        def plot_proportions_xy(ax, threshold):
            fig = plt.figure(figsize=(20, 5))
            fig.suptitle(name_ct+": critical points")
            
            _prettify_axis(ax1, False)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("Spatial1")
            plt.ylabel("Spatial2")
            
            plt.scatter(
                st_adata.obsm["location"][:, 0], st_adata.obsm["location"][:, 1], 
                c=array * (array.values > threshold), s=8, vmax=vmax)
            plt.title("name_ct, main threshold: t={:0.3f}".format(threshold))
            plt.tight_layout()

            return ax

        ax1 = plt.subplot(141)
        ax1 = plot_proportions_xy(ax1, 0)

        ax4 = plt.subplot(144)
        ax4 = plot_proportions_xy(ax1, nom_map)

        #plot characteristic plots
        def characteristic_plot(ax):
            ymax = np.max(z_values)
            _prettify_axis(ax)
            plt.plot(index, z_values, label="noisy data")
            plt.plot(index, smoothed, label="fitted")
            plt.plot(index, ymax * sign_2nd, label="sign 2st derivative")
            # identify points
            plt.vlines(ipoints, ymin=0, ymax=np.max(z_values), color="red", linestyle="--", label="secondary thresholds")
            # nominal mapping
            plt.axvline(nom_map, c="red", label="main threshold")
            plt.ylabel("Autocorrelation")
            plt.xlabel("proportions value")
            plt.title("Autocorrelation study")
            plt.legend()

            return ax

        ax2 = plt.subplot(142)
        ax2 = characteristic_plot(ax2)
        
        # plot on top of histogram
        ax3 = plt.subplot(143)
        _prettify_axis(ax3)
        n, _, _ = plt.hist(array.values)
        plt.vlines(ipoints, ymin=0, ymax=np.max(n), color="red", linestyle="--", label="secondary thresholds")
        # nominal mapping
        plt.axvline(nom_map, c="red", label="main threshold")
        plt.xlabel("proportions value")
        plt.title("Cell type frequency histogram")
        plt.legend()
        
        # add thresholds to dict
        nominal_threshold[name_ct] = nom_map
        ct_thresholds[name_ct] = ipoints
        
        # DUMP TO HTML
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded)

    # write HTML
    with open(output_file,'w') as f:
        f.write(html)


def _prettify_axis(ax, all_=False):
    if not all_:
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    else:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)

def _form_stacked_quantiles(data, N=100):
    quantiles = np.quantile(data, np.linspace(0, 1, N, endpoint=False))
    return quantiles, np.vstack([_flatten(data, q) for q in quantiles])

def _flatten(x, threshold):
    return (x > threshold) * x

def _smooth_get_critical_points(x, noisy_data, k=5, s=0.1):
    f = splrep(x, noisy_data,k=5, s=1)
    smoothed = splev(x,f)
    derivative = splev(x,f,der=1)
    sign_2nd = splev(x,f,der=2) > 0
    curvature = splev(x,f,der=3)
    return noisy_data, smoothed, derivative, sign_2nd, curvature

def _get_autocorrelations(st_adata, stacked_quantiles, quantiles):
    # form dataframes
    loc = pd.DataFrame(data=st_adata.obsm["location"], index=st_adata.obs.index)
    df = pd.DataFrame(data=stacked_quantiles, columns=st_adata.obs.index, index=quantiles)
    # run hotspot
    hs = hotspot.Hotspot(df, model='none', latent=loc,)
    hs.create_knn_graph(
        weighted_graph=True, n_neighbors=10,
    )
    hs_results = hs.compute_autocorrelations(jobs=1)
    return hs_results.index.values, hs_results["Z"].values
