import anndata as ad
import matplotlib.pyplot as plt
import numpy as np


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

        ax1 = plt.subplot(141)
        ax1 = plot_proportions_xy(ax1, 0)

        ax4 = plt.subplot(144)
        ax4 = plot_proportions_xy(ax1, nom_map)

        # get characteristic values
        quantiles, stack = form_stacked_quantiles(array.values)
        index, z_values = get_autocorrelations(st_adata, stack, quantiles)
        z_values, smoothed, derivative, sign_2nd, _ = smooth_get_critical_points(index, z_values, s=0.1)

        #plot characteristic plots
        ymax = np.max(z_values)
        ax2 = plt.subplot(142)
        _prettify_axis(ax2)
        plt.plot(index, z_values, label="noisy data")
        plt.plot(index, smoothed, label="fitted")
        plt.plot(index, ymax * sign_2nd, label="sign 2st derivative")
        # identify points
        ipoints = index[np.where(sign_2nd[:-1] != sign_2nd[1:])[0]]
        plt.vlines(ipoints, ymin=0, ymax=np.max(z_values), color="red", linestyle="--", label="secondary thresholds")
        # nominal mapping
        nom_map = index[np.argmin(derivative)]
        plt.axvline(nom_map, c="red", label="main threshold")
        plt.ylabel("Autocorrelation")
        plt.xlabel("proportions value")
        plt.title("Autocorrelation study")
        plt.legend()
        
        # plot on top of histogram
        ax3 = plt.subplot(143)
        _prettify_axis(ax3)
        n, bins, patches = plt.hist(array.values)
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
    return quantiles, np.vstack([flatten(data, q) for q in quantiles])

def _flatten(x, threshold):
    return (x > threshold) * x



class MyModule(BaseModuleClass):
    """
    Skeleton Variational auto-encoder model.

    Here we implement a basic version of scVI's underlying VAE [Lopez18]_.
    This implementation is for instructional purposes only.

    Parameters
    ----------
    n_input
        Number of input genes
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    dropout_rate
        Dropout rate for neural networks
    """

    def __init__(
        self,
        n_input: int,
        library_log_means: np.ndarray,
        library_log_vars: np.ndarray,
        n_batch: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.n_latent = n_latent
        self.n_batch = n_batch
        # this is needed to comply with some requirement of the VAEMixin class
        self.latent_distribution = "normal"

        self.register_buffer(
            "library_log_means", torch.from_numpy(library_log_means).float()
        )
        self.register_buffer(
            "library_log_vars", torch.from_numpy(library_log_vars).float()
        )

        # setup the parameters of your generative model, as well as your inference model
        self.px_r = torch.nn.Parameter(torch.randn(n_input))
        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        self.z_encoder = Encoder(
            n_input,
            n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input,
            1,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        self.decoder = DecoderSCVI(
            n_latent,
            n_input,
            n_layers=n_layers,
            n_hidden=n_hidden,
        )

    def _get_inference_input(self, tensors):
        """Parse the dictionary to get appropriate args"""
        x = tensors[REGISTRY_KEYS.X_KEY]

        input_dict = dict(x=x)
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]

        input_dict = {
            "z": z,
            "library": library,
        }
        return input_dict

    @auto_move_data
    def inference(self, x):
        """
        High level inference method.

        Runs the inference (encoder) model.
        """
        # log the input to the variational distribution for numerical stability
        x_ = torch.log(1 + x)
        # get variational parameters via the encoder networks
        qz_m, qz_v, z = self.z_encoder(x_)
        ql_m, ql_v, library = self.l_encoder(x_)

        outputs = dict(z=z, qz_m=qz_m, qz_v=qz_v, ql_m=ql_m, ql_v=ql_v, library=library)
        return outputs

    @auto_move_data
    def generative(self, z, library):
        """Runs the generative model."""

        # form the parameters of the ZINB likelihood
        px_scale, _, px_rate, px_dropout = self.decoder("gene", z, library)
        px_r = torch.exp(self.px_r)

        return dict(
            px_scale=px_scale, px_r=px_r, px_rate=px_rate, px_dropout=px_dropout
        )

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        qz_m = inference_outputs["qz_m"]
        qz_v = inference_outputs["qz_v"]
        ql_m = inference_outputs["ql_m"]
        ql_v = inference_outputs["ql_v"]
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]

        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)

        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(
            dim=1
        )

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )

        kl_divergence_l = kl(
            Normal(ql_m, torch.sqrt(ql_v)),
            Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
        ).sum(dim=1)

        reconst_loss = (
            -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
            .log_prob(x)
            .sum(dim=-1)
        )

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = dict(
            kl_divergence_l=kl_divergence_l, kl_divergence_z=kl_divergence_z
        )
        kl_global = torch.tensor(0.0)
        return LossRecorder(loss, reconst_loss, kl_local, kl_global)

    @torch.no_grad()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> np.ndarray:
        r"""
        Generate observation samples from the posterior predictive distribution.

        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.

        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale scamples to

        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = dict(n_samples=n_samples)
        _, generative_outputs, = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        px_r = generative_outputs["px_r"]
        px_rate = generative_outputs["px_rate"]
        px_dropout = generative_outputs["px_dropout"]

        dist = ZeroInflatedNegativeBinomial(
            mu=px_rate, theta=px_r, zi_logits=px_dropout
        )

        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    @torch.no_grad()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            z = inference_outputs["z"]
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.reconstruction_loss

            # Log-probabilities
            n_batch = self.library_log_means.shape[1]
            local_library_log_means = F.linear(
                one_hot(batch_index, n_batch), self.library_log_means
            )
            local_library_log_vars = F.linear(
                one_hot(batch_index, n_batch), self.library_log_vars
            )
            p_l = (
                Normal(local_library_log_means, local_library_log_vars.sqrt())
                .log_prob(library)
                .sum(dim=-1)
            )

            p_z = (
                Normal(torch.zeros_like(qz_m), torch.ones_like(qz_v))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = Normal(qz_m, qz_v.sqrt()).log_prob(z).sum(dim=-1)
            q_l_x = Normal(ql_m, ql_v.sqrt()).log_prob(library).sum(dim=-1)

            to_sum[:, i] = p_z + p_l + p_x_zl - q_z_x - q_l_x

        batch_log_lkl = torch.logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl
