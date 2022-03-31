import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep, splev
from scipy.stats import ks_2samp
import pandas as pd
import hotspot
import base64
from io import BytesIO
from scipy.spatial.distance import pdist, squareform
import cmap2d
import gseapy
import torch
from adjustText import adjust_text
from statsmodels.stats.multitest import multipletests
from sklearn.mixture import GaussianMixture


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
        fig = plt.figure(figsize=(20, 5))
        fig.suptitle(name_ct+": critical points")

        array = st_adata.obsm["proportions"][name_ct]
        vmax = np.quantile(array.values, 0.99)

        # get characteristic values
        quantiles, stack = _form_stacked_quantiles(array.values)
        index, z_values = _get_autocorrelations(st_adata, stack, quantiles)
        z_values, smoothed, derivative, sign_2nd, _ = _smooth_get_critical_points(index, z_values, s=0.1)
        ipoints = index[np.where(sign_2nd[:-1] != sign_2nd[1:])[0][0]]
        nom_map = index[np.argmin(derivative)]
        
        #PLOT 1 shows proportions in spatial dimensions without thresholding
        def plot_proportions_xy(ax, threshold):            
            _prettify_axis(ax, False)
            plt.xticks([])
            plt.yticks([])
            plt.xlabel("Spatial1")
            plt.ylabel("Spatial2")
            plt.scatter(
                st_adata.obsm["location"][:, 0], st_adata.obsm["location"][:, 1], 
                c=array * (array.values > threshold), s=14, vmax=vmax)
            plt.title("name_ct, main threshold: t={:0.3f}".format(threshold))
            plt.tight_layout()

            return ax

        ax1 = plt.subplot(141)
        ax1 = plot_proportions_xy(ax1, 0)

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

        ax4 = plt.subplot(144)
        ax4 = plot_proportions_xy(ax4, nom_map)
        
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
    
    return ct_thresholds, nominal_threshold

def explore_gamma_space(
    st_adata,
    sc_adata,
    st_model,
    sc_model,
    ct_thresholds=None,
    output_file='sPCA.html',
    ct_list=None):
    """
    Function to compute automatic threshold on cell type proportion values.
    For further reference check [Lopez22].

    Parameters
    ----------
    st_adata
        Spatial sequencing dataset with proportions in obsm['proportions']
    sc_adata
        Single cell sequencing dataset used for training CondSCVI
    st_model
        Trained destVI model
    sc_model
        Trained CondSCVI model
    ct_threshold
        List with threshold values for cell type proportions
    output_file
        File where html output is stored. Defaults to 'sPCA.html'
    ct_list
        Celltypes to use. Defaults to all celltypes.

    Returns
    -------
    ct_thresholds
        Dictionary containing all threshold values.
    
    """
    html = "<h1>sPCA analysis</h1>"

    if ct_thresholds is None:
        ct_thresholds = {ct: 0 for ct in ct_list}
    tri_coords = [[-1,-1], [-1,1], [1, 0]]
    tri_colors = [(1,0,0), (0,1,0), (0,0,1)]

    gamma = st_model.get_gamma(return_numpy=True)

    for name_ct in ct_list:
        filter_ = st_adata.obsm["proportions"][name_ct].values > ct_thresholds[name_ct]
        locations = st_adata.obsm["location"][filter_]
        proportions = st_adata.obsm["proportions"][name_ct].values[filter_]
        ct_index = np.where(name_ct == st_model.cell_type_mapping)[0][0]
        data = gamma[:, :, ct_index][filter_]
        vec = _get_spatial_components(locations, proportions, data)
        # project data onto them
        projection = np.dot(data - np.mean(data, 0), vec)

        # create the colormap
        cmap = cmap2d.TernaryColorMap(tri_coords, tri_colors)

        # apply colormap to spatial data
        color = np.vstack([cmap(projection[i]) for i in range(projection.shape[0])])
        
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(name_ct)
        fig.tight_layout(rect=[0, 0.1, 1, 0.2])
        ax1 = plt.subplot(132)
        _prettify_axis(ax1)
        plt.scatter(projection[:, 0], projection[:, 1],c=color, marker="X")
        # variance and explained variance
        total_var = np.sum(np.diag(np.cov(data.T)))
        explained_var = 100 * np.diag(np.cov(projection.T)) / total_var 
        plt.xlabel("SpatialPC1 ({:.1f}% explained var)".format(explained_var[0]))
        plt.ylabel("SpatialPC2 ({:.1f}% explained var)".format(explained_var[1]))
        plt.title("Projection of the spatial data")

        ax3 = plt.subplot(131)
        _prettify_axis(ax3, False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Spatial1")
        plt.ylabel("Spatial2")
        plt.scatter(st_adata.obsm["location"][:, 0], st_adata.obsm["location"][:, 1], alpha=0.1, s=7, c="blue")
        plt.scatter(st_adata.obsm["location"][filter_, 0], st_adata.obsm["location"][filter_, 1], 
                    c=color, s=7)
        plt.title("Spatial transcriptome coloring")

        # go back to the single-cell data and find gene correlated with the axis
        sc_adata_slice = sc_adata[sc_adata.obs[
            sc_model.registry_['setup_args']['labels_key']] == name_ct]
        normalized_counts = sc_adata_slice.X.A
        sc_latent = sc_model.get_latent_representation(sc_adata_slice)
        sc_projection = np.dot(sc_latent - np.mean(sc_latent,0), vec)

        # show the colormap for single-cell data
        color = np.vstack([cmap(sc_projection[i]) for i in range(sc_projection.shape[0])])
        ax2 = plt.subplot(133)
        _prettify_axis(ax2)
        plt.scatter(sc_projection[:, 0], sc_projection[:, 1],c=color)
        # variance and explained variance
        total_var = np.sum(np.diag(np.cov(sc_latent.T)))
        explained_var = 100 * np.diag(np.cov(sc_projection.T)) / total_var 
        plt.xlabel("SpatialPC1 ({:.1f}% explained var)".format(explained_var[0]))
        plt.ylabel("SpatialPC2 ({:.1f}% explained var)".format(explained_var[1]))
        plt.title("Projection of the scRNA-seq data")
        plt.tight_layout()

        # DUMP TO HTML
        tmpfile = BytesIO()
        plt.savefig(tmpfile, format='png')
        encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
        html += '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
        
        # calculate correlations, and for each axis:
        # (A) display top 50 genes + AND - (B) for each gene set, get GSEA 
        for d in [0, 1]:
            html += f"<h4>Genes associated with SpatialPC{d+1}</h4>"
            r = _vcorrcoef(normalized_counts.T, sc_projection[:, d])
            for mode in ["Positively", "Negatively"]:
                ranking = np.argsort(r)
                if mode == "Positively":
                    ranking = ranking[::-1]
                gl = list(st_adata.var.index[ranking[:50]])
                enr = gseapy.enrichr(gene_list=gl, description='pathway', 
                                    gene_sets='BioPlanet_2019', outdir='test', no_plot=True)
                html += f"<h5> {mode} </h5>"
                html += "<p>" + ", ".join(gl) + "</p>"
                text_signatures = enr.results.head(10)["Term"].values
                for i in range(10):
                    if enr.results.iloc[i]["Adjusted P-value"] < 0.01:
                        text_signatures[i] += "*"
                
                html += "<p>" + ", ".join(text_signatures) + "</p>"            
    # write HTML
    with open(output_file,'w') as f:
        f.write(html)

def de_genes(st_adata, st_model, mask, ct, mask2=None, interesting_genes=None, de_results=None):
    # get statistics
    if mask2 is None:
        mask2 = ~mask

    if de_results is None:
        avg_library_size = np.mean(np.sum(st_adata.layers["counts"], axis=1).flatten())
        exp_px_o = st_model.module.px_o.detach().exp().cpu().numpy()
        imputations = st_model.get_scale_for_ct(ct).values
        mean = avg_library_size * imputations

        concentration = torch.tensor(avg_library_size * imputations / exp_px_o)
        rate = torch.tensor(1. / exp_px_o)

        # slice conditions
        N_mask, N_unmask = (10, 10)

        def simulation(mask_, N_mask_):
            # generate 
            simulated = torch.distributions.Gamma(
                concentration=concentration[mask_], rate = rate).sample((N_mask_,)).cpu().numpy()
            simulated = np.log(simulated + 1)
            simulated = simulated.reshape((-1, simulated.shape[-1]))
            return simulated

        simulated_case = simulation(mask, N_mask)
        simulated_control = simulation(mask2, N_unmask)

        de = np.array([ks_2samp(simulated_case[:, gene], 
                    simulated_control[:, gene], 
                    alternative='two-sided', mode="asymp") for gene in range(simulated_control.shape[1])])
        lfc = np.log2(1+mean[mask]).mean(0) - np.log2(1+mean[mask2]).mean(0)
        res = pd.DataFrame(data=np.vstack([lfc, de[:, 0], de[:, 1]]), columns=st_adata.var.index, 
                        index=["log2FC", "pval", "score"]).T

    corr_p_vals = multipletests(res['pval'], method='fdr_bh')
    min_score = np.min(res['score'][corr_p_vals[0]])
    plt.figure(figsize=(5, 5))
    # plot DE genes
    mask_de = (res['score'] > min_score) * (np.abs(lfc) > _get_delta(lfc))
    de_scatter = plt.scatter(lfc[mask_de], res['score'][mask_de], s=10, c="r")
    nde_scatter = plt.scatter(lfc[~mask_de], res['score'][~mask_de], s=10, c="black")
    plt.xlabel("log2 fold-change \n{:s}".format(ct))
    plt.ylabel("score")
    plt.legend((de_scatter, nde_scatter), ("DE genes", "Other genes"), frameon=True)
    if interesting_genes is None:
        interesting_genes = ["Mmp2", "Dcn", "Col3a1", "Pcolce", "Col4a1", "Sparc", "Mxra8", "Lgals1"]
    texts = []
    for i, gene in enumerate(interesting_genes):
        ind = np.where(st_adata.var.index == gene)[0]
        x_coord, y_coord = lfc[ind], res['score'][ind]
        plt.scatter(x_coord, y_coord, c="r", s=10)
        texts += [plt.text(x_coord, y_coord, gene, fontsize=12)]
    adjust_text(texts, lfc, res['score'].values, arrowprops=dict(arrowstyle="-", color='blue'))

    return res

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
    # create Anndata and run hotspot
    adata = ad.AnnData(stacked_quantiles.T)
    adata.obs_names = st_adata.obs.index
    adata.var_names = [str(i) for i in quantiles]
    adata.obsm["spatial"] = st_adata.obsm["location"]
    hs = hotspot.Hotspot(adata, model='none', latent_obsm_key="spatial")
    hs.create_knn_graph(
        weighted_graph=True, n_neighbors=10,
    )
    hs_results = hs.compute_autocorrelations(jobs=1)
    index = np.array([float(i) for i in hs_results.index.values])
    return index, hs_results["Z"].values

def _get_laplacian(s, pi):
    N = s.shape[0]
    dist_table = pdist(s)
    bandwidth = np.median(dist_table)
    sigma=(0.5 * bandwidth**2)
    
    l2_square = squareform(dist_table)**2
    D = np.exp(- l2_square / sigma) * np.dot(pi, pi.T)
    L = -D
    sum_D = np.sum(D, axis=1)
    for i in range(N):
            L[i, i] = sum_D[i]
    return L

def _get_spatial_components(locations, proportions, data):
    # find top two spatial principal vectors
    # form laplacian
    L = _get_laplacian(locations, proportions)
    # center data
    transla_ = data.copy()
    transla_ -= np.mean(transla_, axis=0)
    # get eigenvectors
    A = np.dot(transla_.T, np.dot(L, transla_))
    w, v = np.linalg.eig(A)
    # don't forget to sort them...
    idx = np.argsort(w)[::-1]
    vec = v[:, idx][:, :2]
    return vec

def _vcorrcoef(X,y):
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r

def _get_delta(lfc):
    return np.max(np.abs(GaussianMixture(n_components=3).fit(lfc.reshape(-1, 1)).means_))

