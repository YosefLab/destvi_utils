import anndata as ad
import hotspot
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splrep
from scipy.spatial.distance import pdist, squareform
from sklearn.mixture import GaussianMixture


def _prettify_axis(ax, spatial=False):
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")
    if spatial:
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("Spatial1")
        plt.ylabel("Spatial2")


def _form_stacked_quantiles(data, N=100):
    quantiles = np.quantile(data, np.linspace(0, 1, N, endpoint=False))
    return quantiles, np.vstack([_flatten(data, q) for q in quantiles])


def _flatten(x, threshold):
    return (x > threshold) * x


def _smooth_get_critical_points(x, noisy_data, k=5, s=0.1):
    f = splrep(x, noisy_data, k=5, s=1)
    smoothed = splev(x, f)
    derivative = splev(x, f, der=1)
    sign_2nd = splev(x, f, der=2) > 0
    curvature = splev(x, f, der=3)
    return noisy_data, smoothed, derivative, sign_2nd, curvature


def _get_autocorrelations(st_adata, stacked_quantiles, quantiles):
    # create Anndata and run hotspot
    adata = ad.AnnData(stacked_quantiles.T)
    adata.obs_names = st_adata.obs.index
    adata.var_names = [str(i) for i in quantiles]
    adata.obsm["spatial"] = st_adata.obsm["spatial"]
    hs = hotspot.Hotspot(adata, model="none", latent_obsm_key="spatial")
    hs.create_knn_graph(
        weighted_graph=True,
        n_neighbors=10,
    )
    hs_results = hs.compute_autocorrelations(jobs=1)
    index = np.array([float(i) for i in hs_results.index.values])
    return index, hs_results["Z"].values


def _get_laplacian(s, pi):
    N = s.shape[0]
    dist_table = pdist(s)
    bandwidth = np.median(dist_table)
    sigma = 0.5 * bandwidth**2

    l2_square = squareform(dist_table) ** 2
    D = np.exp(-l2_square / sigma) * np.dot(pi, pi.T)
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


def _vcorrcoef(X, y):
    Xm = np.reshape(np.mean(X, axis=1), (X.shape[0], 1))
    ym = np.mean(y)
    r_num = np.sum((X - Xm) * (y - ym), axis=1)
    r_den = np.sqrt(np.sum((X - Xm) ** 2, axis=1) * np.sum((y - ym) ** 2))
    r = r_num / r_den
    return r


def _get_delta(lfc):
    return np.max(
        np.abs(GaussianMixture(n_components=3).fit(np.array(lfc).reshape(-1, 1)).means_)
    )
