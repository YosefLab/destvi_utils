import base64
import logging
from argparse import ArgumentError
from io import BytesIO

import cmap2d
import gseapy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from adjustText import adjust_text
from IPython.core.display import HTML, display
from scipy.sparse import issparse
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import multipletests

from . import _utils


def automatic_proportion_threshold(
    st_adata, kind_threshold="primary", output_file=None, ct_list=None
):
    """
    Function to compute automatic threshold on cell type proportion values.
    For further reference check [Lopez22]_.

    Parameters
    ----------
    st_adata
        Spatial sequencing dataset with proportions in obsm['proportions'] and spatial location in
        obsm['spatial']
    kind_threshold
        Which threshold value to use. Supported are 'primary', 'secondary'.
        'min_value' uses the minimum of primary and secondary threshold for each cell type.
    output_file
        File where html output is stored. None means displaying the results and not storing them.
        Defaults to None.
    ct_list
        Celltypes to use. Defaults to all celltypes.

    Returns
    -------
    ct_thresholds
        Dictionary containing all threshold values.

    """
    if "proportions" not in st_adata.obsm:
        raise ValueError(
            'Please provide cell type proportions in st_adata.obsm["proportions"] and restart.'
        )
    if "spatial" not in st_adata.obsm:
        raise ValueError(
            'Please provide cell type locations in st_adata.obsm["spatial"] and restart.'
        )

    if ct_list is None:
        ct_list = list(st_adata.obsm["proportions"].columns)
    ct_thresholds = {}

    html = "<h2>Automatic thresholding</h2>"

    for name_ct in ct_list:
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(
            name_ct + ": critical points", fontsize="x-large", fontweight="semibold"
        )

        array = st_adata.obsm["proportions"][name_ct]
        vmax = np.quantile(array.values, 0.99)

        # get characteristic values
        quantiles, stack = _utils._form_stacked_quantiles(array.values)
        index, z_values = _utils._get_autocorrelations(st_adata, stack, quantiles)
        (
            z_values,
            smoothed,
            derivative,
            sign_2nd,
            _,
        ) = _utils._smooth_get_critical_points(index, z_values, s=0.1)
        ipoints = index[np.where(sign_2nd[:-1] != sign_2nd[1:])[0]]
        nom_map = index[np.argmin(derivative)]

        # add thresholds to dict
        if kind_threshold == "primary":
            ct_thresholds[name_ct] = nom_map
        elif kind_threshold == "secondary":
            ct_thresholds[name_ct] = ipoints[0]
        else:
            raise ArgumentError(
                'Kind threshold {} is not defined. Use "secondary" or "primary"'.format(
                    kind_threshold
                )
            )

        # PLOT 1 shows proportions in spatial dimensions without thresholding
        def plot_proportions_xy(ax, threshold):
            _utils._prettify_axis(ax, spatial=True)
            plt.scatter(
                st_adata.obsm["spatial"][:, 0],
                st_adata.obsm["spatial"][:, 1],
                c=array * (array.values > threshold),
                s=14,
                vmax=vmax,
                cmap="Reds",
            )
            plt.colorbar()
            plt.title("name_ct, threshold: t={:0.3f}".format(threshold))
            plt.tight_layout(rect=[0, 0.03, 1, 0.9])

            return ax

        ax1 = plt.subplot(131)
        _utils._prettify_axis(ax1)
        ax1 = plot_proportions_xy(ax1, 0)

        # plot on top of histogram
        ax2 = plt.subplot(132)
        _utils._prettify_axis(ax2)
        n, _, _ = plt.hist(array.values)
        plt.vlines(
            ipoints,
            ymin=0,
            ymax=np.max(n),
            color="red",
            linestyle="--",
            label="secondary thresholds",
        )
        # nominal mapping
        plt.axvline(nom_map, c="red", label="main threshold")
        plt.xlabel("proportions value")
        plt.title("Cell type frequency histogram")
        plt.legend()

        ax3 = plt.subplot(133)
        ax3 = plot_proportions_xy(ax3, ct_thresholds[name_ct])

        if output_file is not None:
            tmpfile = BytesIO()
            plt.savefig(tmpfile, format="png")
            encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
            html += "<img src='data:image/png;base64,{}'>".format(encoded)
            plt.close()
        else:
            plt.show()

    # dump+write to HTML
    if output_file is not None:
        logging.info(
            "Saving output to {}. Set output_file=None to display results.".format(
                output_file
            )
        )
        with open(output_file, "w") as f:
            f.write(html)

    return ct_thresholds


def explore_gamma_space(
    st_model,
    sc_model,
    st_adata=None,
    ct_thresholds=None,
    output_file=None,
    ct_list=None,
):
    """
    Function to compute automatic threshold on cell type proportion values.
    For further reference check [Lopez22]_.

    Parameters
    ----------
    st_model
        Trained destVI model
    sc_model
        Trained CondSCVI model
    st_adata
        Spatial sequencing dataset with proportions in obsm['proportions']. Otherwise uses data in st_model.
    ct_threshold
        List with threshold values for cell type proportions
    output_file
        File where html output is stored. None means displaying the results and not storing them.
        Defaults to None.
    ct_list
        Celltypes to use. Defaults to all celltypes.

    """
    html = "<h1>sPCA analysis</h1>"

    if st_adata is None:
        st_adata = st_model.adata
        st_adata.obsm["proportions"] = st_model.get_proportions()
    else:
        if "proportions" not in st_adata.obsm:
            raise ValueError(
                'Please provide cell type proportions in st_adata.obsm["proportions"] and restart.'
            )

    if "spatial" not in st_adata.obsm:
        raise ValueError(
            'Please provide cell type locations in st_adata.obsm["spatial"] and restart.'
        )

    sc_adata = sc_model.adata

    if ct_list is None:
        ct_list = list(st_adata.obsm["proportions"].columns)
    if ct_thresholds is None:
        ct_thresholds = {ct: 0 for ct in ct_list}

    tri_coords = [[-1, -1], [-1, 1], [1, 0]]
    tri_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    gamma = st_model.get_gamma(return_numpy=True)

    for name_ct in ct_list:
        filter_ = st_adata.obsm["proportions"][name_ct].values > ct_thresholds[name_ct]
        locations = st_adata.obsm["spatial"][filter_]
        proportions = st_adata.obsm["proportions"][name_ct].values[filter_]
        ct_index = np.where(name_ct == st_model.cell_type_mapping)[0][0]
        data = gamma[:, :, ct_index][filter_]
        vec = _utils._get_spatial_components(locations, proportions, data)
        # project data onto them
        projection = np.dot(data - np.mean(data, 0), vec)

        # create the colormap
        cmap = cmap2d.TernaryColorMap(tri_coords, tri_colors)

        # apply colormap to spatial data
        color = np.vstack([cmap(projection[i]) for i in range(projection.shape[0])])

        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(name_ct, fontsize="x-large", fontweight="semibold")
        ax1 = plt.subplot(132)
        _utils._prettify_axis(ax1)
        plt.scatter(projection[:, 0], projection[:, 1], c=color, marker="X")
        # variance and explained variance
        total_var = np.sum(np.diag(np.cov(data.T)))
        explained_var = 100 * np.diag(np.cov(projection.T)) / total_var
        plt.xlabel("SpatialPC1 ({:.1f}% explained var)".format(explained_var[0]))
        plt.ylabel("SpatialPC2 ({:.1f}% explained var)".format(explained_var[1]))
        plt.title("Projection of the spatial data")

        ax3 = plt.subplot(131)
        _utils._prettify_axis(ax3, spatial=True)
        plt.scatter(
            st_adata.obsm["spatial"][:, 0],
            st_adata.obsm["spatial"][:, 1],
            alpha=0.1,
            s=7,
            c="blue",
        )
        plt.scatter(
            st_adata.obsm["spatial"][filter_, 0],
            st_adata.obsm["spatial"][filter_, 1],
            c=color,
            s=7,
        )
        plt.title("Spatial transcriptome coloring")

        # go back to the single-cell data and find gene correlated with the axis
        sc_adata_slice = sc_adata[
            sc_adata.obs[sc_model.registry_["setup_args"]["labels_key"]] == name_ct
        ].copy()
        is_sparse = issparse(sc_adata_slice.X)
        normalized_counts = sc_adata_slice.X.A if is_sparse else sc_adata_slice.X
        indices_ct = np.where(
            sc_adata.obs[sc_model.registry_["setup_args"]["labels_key"]] == name_ct
        )

        sc_latent = sc_model.get_latent_representation(indices=indices_ct)
        sc_projection = np.dot(sc_latent - np.mean(sc_latent, 0), vec)

        # show the colormap for single-cell data
        color = np.vstack(
            [cmap(sc_projection[i]) for i in range(sc_projection.shape[0])]
        )
        ax2 = plt.subplot(133)
        _utils._prettify_axis(ax2)
        plt.scatter(sc_projection[:, 0], sc_projection[:, 1], c=color)
        # variance and explained variance
        total_var = np.sum(np.diag(np.cov(sc_latent.T)))
        explained_var = 100 * np.diag(np.cov(sc_projection.T)) / total_var
        plt.xlabel("SpatialPC1 ({:.1f}% explained var)".format(explained_var[0]))
        plt.ylabel("SpatialPC2 ({:.1f}% explained var)".format(explained_var[1]))
        plt.title("Projection of the scRNA-seq data")
        plt.tight_layout(rect=[0, 0.03, 1, 0.9])

        # DUMP TO HTML
        if output_file is not None:
            tmpfile = BytesIO()
            plt.savefig(tmpfile, format="png")
            encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
            html += "<img src='data:image/png;base64,{}'>".format(encoded)

        # calculate correlations, and for each axis:
        # (A) display top 50 genes + AND - (B) for each gene set, get GSEA
        for d in [0, 1]:
            if output_file is not None:
                html += f"<h4>Genes associated with SpatialPC{d+1}</h4>"
            else:
                print("Genes associated with SpatialPC", d + 1)
            r = _utils._vcorrcoef(normalized_counts.T, sc_projection[:, d])
            for mode in ["Positively", "Negatively"]:
                ranking = np.argsort(r)
                if mode == "Positively":
                    ranking = ranking[::-1]
                gl = list(st_adata.var.index[ranking[:50]])
                enr = gseapy.enrichr(
                    gene_list=gl,
                    description="pathway",
                    gene_sets="BioPlanet_2019",
                    outdir="test",
                    no_plot=True,
                )
                text_signatures = enr.results.head(10)["Term"].values
                for i in range(len(text_signatures)):
                    if enr.results.iloc[i]["Adjusted P-value"] < 0.01:
                        text_signatures[i] += "*"
                signatures = ", ".join(text_signatures)
                genes = ", ".join(gl)

                if output_file is not None:
                    html += f"<h5> {mode} </h5>"
                    html += "<p>" + genes + "</p>"
                    html += "<p>" + signatures + "</p>"
                else:
                    print(mode)
                    print(genes)
                    print(signatures)
        plt.close(fig)

    # write HTML
    if output_file is not None:
        logging.info(
            "Saving output to {}. Set output_file=None to display results.".format(
                output_file
            )
        )
        with open(output_file, "w") as f:
            f.write(html)
    else:
        display(HTML(html))


def de_genes(
    st_model, mask, ct, threshold=0.0, st_adata=None, mask2=None, key=None, N_sample=10
):
    """
    Function to compute differential expressed genes from generative model.
    For further reference check [Lopez22]_.

    Parameters
    ----------
    st_adata
        Spatial sequencing dataset with proportions in obsm['proportions']. If not provided uses data in st_model.
    st_model
        Trained destVI model
    mask
        Mask for subsetting the spots to condition 1 in differential expression.
    mask2
        Mask for subsetting the spots to condition 2 in differential expression (reference). If none, inverse of mask.
    ct
        Cell type for which differential expression is computed
    threshold
        Proportion threshold to subset to spots with this amount of cell type proportion
    key
        Key to store values in st_adata.uns[key]. If None returns pandas dataframe with DE results. Defaults to None
    N_sample
        N_samples drawn from generative model to simulate expression values.

    Returns
    -------
    res
        If key is None. Pandas dataframe containing results of differential expression.
        Dataframe columns are "log2FC", "pval", "score".
        If key is provided. mask, mask2 and de_results are stored in st_adata.uns[key]. Dictionary keys are
        "mask_active", "mask_rest", "de_results".

    """

    # get statistics
    if mask2 is None:
        mask2 = ~mask

    if st_adata is None:
        st_adata = st_model.adata
        st_adata.obsm["proportions"] = st_model.get_proportions()
    else:
        if "proportions" not in st_adata.obsm:
            raise ValueError(
                'Please provide cell type proportions in st_adata.obsm["proportions"] and restart.'
            )

    if st_model.registry_["setup_args"]["layer"]:
        expression = st_adata.layers[st_model.registry_["setup_args"]["layer"]]
    else:
        expression = st_adata.X

    mask = np.logical_and(mask, st_adata.obsm["proportions"][ct] > threshold)
    mask2 = np.logical_and(mask2, st_adata.obsm["proportions"][ct] > threshold)

    avg_library_size = np.mean(np.sum(expression, axis=1).flatten())
    exp_px_o = st_model.module.px_o.detach().exp().cpu().numpy()
    imputations = st_model.get_scale_for_ct(ct).values
    mean = avg_library_size * imputations

    concentration = torch.tensor(avg_library_size * imputations / exp_px_o)
    rate = torch.tensor(1.0 / exp_px_o)

    # slice conditions
    N_mask = N_unmask = N_sample

    def simulation(mask_, N_mask_):
        # generate
        simulated = (
            torch.distributions.Gamma(concentration=concentration[mask_], rate=rate)
            .sample((N_mask_,))
            .cpu()
            .numpy()
        )
        simulated = np.log(simulated + 1)
        simulated = simulated.reshape((-1, simulated.shape[-1]))
        return simulated

    simulated_case = simulation(mask, N_mask)
    simulated_control = simulation(mask2, N_unmask)

    de = np.array(
        [
            ks_2samp(
                simulated_case[:, gene],
                simulated_control[:, gene],
                alternative="two-sided",
                mode="asymp",
            )
            for gene in range(simulated_control.shape[1])
        ]
    )
    lfc = np.log2(1 + mean[mask]).mean(0) - np.log2(1 + mean[mask2]).mean(0)
    res = pd.DataFrame(
        data=np.vstack([lfc, de[:, 0], de[:, 1]]),
        columns=st_adata.var.index,
        index=["log2FC", "score", "pval"],
    ).T

    # Store results in st_adata
    if key is not None:
        st_adata.uns[key] = {}
        st_adata.uns[key]["de_results"] = res.sort_values(by="score", ascending=False)
        st_adata.uns[key]["mask_active"] = mask
        st_adata.uns[key]["mask_rest"] = mask2
        return st_adata
    else:
        return res


def plot_de_genes(st_adata, key, output_file=None, interesting_genes=None):
    """
    Function to plot results of differential expressed genes in a Volcano plot.
    For further reference check [Lopez22]_.

    Parameters
    ----------
    st_adata
        Spatial sequencing dataset with proportions in obsm['proportions']. If not provided uses data in st_model.
    key
        Key under which results of DE comparison are stored
    output_file
        File where picture is stored. None means displaying the results and not storing them.
        Defaults to None.
    interesting_genes
        Label dots in scatter plots with corresponding gene name. Uses first two genes if None.
    """
    if "spatial" not in st_adata.obsm:
        raise ValueError(
            'Please provide locations in st_adata.obsm["spatial"] and restart.'
        )
    if key not in st_adata.uns:
        raise ValueError(
            "DE results are not stored with given key. Please run de_genes function with given key."
        )
    matching_genes = np.array([i in st_adata.var_names for i in interesting_genes])
    if not matching_genes.all():
        missing_genes = np.array(interesting_genes)[~matching_genes]
        raise ValueError(
            "{} are not in st_adata.var_names. Remove these genes from interesting_genes.".format(
                missing_genes
            )
        )

    locations = st_adata.obsm["spatial"]
    res = st_adata.uns[key]["de_results"]
    mask_active = st_adata.uns[key]["mask_active"]
    mask_rest = st_adata.uns[key]["mask_rest"]

    corr_p_vals = multipletests(res["pval"], method="fdr_bh")
    min_score = np.min(res["score"][corr_p_vals[0]])
    plt.figure(figsize=(10, 5))

    # plot DE genes
    ax1 = plt.subplot(122)
    ax1.text(
        -0.1,
        1.05,
        "B",
        transform=ax1.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    # plot DE genes
    mask_de = (res["score"] > min_score) * (
        np.abs(res["log2FC"]) > _utils._get_delta(res["log2FC"])
    )
    plt.scatter(res["log2FC"][mask_de], res["score"][mask_de], s=10, c="r")
    plt.scatter(res["log2FC"][~mask_de], res["score"][~mask_de], s=10, c="black")
    plt.xlabel("log2 fold-change")
    plt.ylabel("score")
    plt.grid(False)
    if interesting_genes is not None:
        texts = []
        for gene in interesting_genes:
            x_coord, y_coord = res.loc[gene, "log2FC"], res.loc[gene, "score"]
            plt.scatter(x_coord, y_coord, c="r", s=10)
            texts += [plt.text(x_coord, y_coord, gene, fontsize=12)]
        adjust_text(
            texts,
            res["log2FC"].values,
            res["score"].values,
            arrowprops=dict(arrowstyle="-", color="blue"),
        )

    ax2 = plt.subplot(121)
    ax2.text(
        -0.1,
        1.05,
        "A",
        transform=ax2.transAxes,
        fontsize=16,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax2.scatter(
        locations[mask_active][:, 0], locations[mask_active][:, 1], s=5, label="active"
    )
    ax2.scatter(
        locations[mask_rest][:, 0], locations[mask_rest][:, 1], s=5, label="rest"
    )
    plt.legend()
    _utils._prettify_axis(ax2, spatial=True)

    plt.tight_layout()
    if output_file is not None:
        logging.info(
            "Saving output to {}. Set output_file=None to display results.".format(
                output_file
            )
        )
        plt.savefig(output_file, dpi=300)
        plt.close()
    else:
        plt.show()
