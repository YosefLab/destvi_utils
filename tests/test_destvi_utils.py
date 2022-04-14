import numpy as np
from scvi.data import synthetic_iid
from scvi.model import CondSCVI, DestVI

import destvi_utils


def test_destvi():
    # Step1 learn CondSCVI
    n_latent = 2
    n_labels = 5
    n_layers = 2
    dataset = synthetic_iid(n_labels=n_labels)
    dataset.obsm["spatial"] = np.random.randn(dataset.n_obs, 2)
    dataset.obs["overclustering_vamp"] = list(range(dataset.n_obs))
    CondSCVI.setup_anndata(dataset, labels_key="labels")
    sc_model = CondSCVI(dataset, n_latent=n_latent, n_layers=n_layers)
    sc_model.train(1, train_size=1)

    DestVI.setup_anndata(dataset, layer=None)
    spatial_model = DestVI.from_rna_model(dataset, sc_model, vamp_prior_p=dataset.n_obs)
    spatial_model.train(max_epochs=1)
    assert not np.isnan(spatial_model.history["elbo_train"].values[0][0])
    dataset.obsm["proportions"] = spatial_model.get_proportions()
    assert spatial_model.get_gamma(return_numpy=True).shape == (
        dataset.n_obs,
        n_latent,
        n_labels,
    )
    assert spatial_model.get_scale_for_ct("label_0", np.arange(50)).shape == (
        50,
        dataset.n_vars,
    )
    assert not np.isnan(
        np.fromiter(
            destvi_utils.automatic_proportion_threshold(
                dataset, kind_threshold="primary"
            ).values(),
            dtype=float,
        )
    ).any()
    assert not np.isnan(
        np.fromiter(
            destvi_utils.automatic_proportion_threshold(
                dataset, kind_threshold="secondary"
            ).values(),
            dtype=float,
        )
    ).any()
    destvi_utils.explore_gamma_space(spatial_model, sc_model)
    ct = dataset.obsm["proportions"].columns[0]
    destvi_utils.de_genes(
        spatial_model,
        mask=dataset.obs["overclustering_vamp"] < 10,
        key="disease",
        ct=ct,
    )
    assert not np.isnan(dataset.uns["disease"]["de_results"].values).any()
    destvi_utils.plot_de_genes(
        dataset, key="disease", interesting_genes=dataset.var_names[0:2]
    )
