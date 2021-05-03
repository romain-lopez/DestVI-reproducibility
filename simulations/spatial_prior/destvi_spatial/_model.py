"""Strategy

1. Align code with destvi, make notebook run
2. Remove all duplicate code
3. Push
"""


import logging
from typing import Optional, OrderedDict

import numpy as np
import torch
from anndata import AnnData
from scvi.data import register_tensor_from_anndata
from scvi.dataloaders import DataSplitter
from scvi.model import CondSCVI, DestVI
from scvi.train import TrainingPlan, TrainRunner
from torch.optim.lr_scheduler import ReduceLROnPlateau

from ._module import HSTDeconv

logger = logging.getLogger(__name__)


class CustomTrainingPlan(TrainingPlan):
    def __init__(self, vae_model, n_obs_training, myparameters=None, **kwargs):
        super().__init__(vae_model, n_obs_training, **kwargs)
        self.myparameters = myparameters

    def configure_optimizers(self):
        if self.myparameters is not None:
            logger.info("Training all parameters")
            params = self.myparameters
        else:
            logger.info("Training subsample of all parameters")
            params = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer = torch.optim.Adam(
            params, lr=self.lr, eps=0.01, weight_decay=self.weight_decay
        )
        config = {"optimizer": optimizer}
        if self.reduce_lr_on_plateau:
            scheduler = ReduceLROnPlateau(
                optimizer,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                threshold_mode="abs",
                verbose=True,
            )
            config.update(
                {
                    "lr_scheduler": scheduler,
                    "monitor": self.lr_scheduler_metric,
                },
            )
        return config


class DestVISpatial(DestVI):
    """
    Hierarchical DEconvolution of Spatial Transcriptomics data (DestVI).

    Parameters
    ----------
    adata
        AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    st_adata
        spatial transcriptomics AnnData object that has been registered via :func:`~scvi.data.setup_anndata`.
    state_dict
        state_dict from the CondSCVI model
    use_gpu
        Use the GPU or not.
    **model_kwargs
        Keyword args for :class:`~scvi.modules.VAEC`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.data.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.external.CondSCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    """

    def __init__(
        self,
        st_adata: AnnData,
        cell_type_mapping: np.ndarray,
        decoder_state_dict: OrderedDict,
        px_decoder_state_dict: OrderedDict,
        px_r: np.ndarray,
        n_hidden: int,
        n_latent: int,
        n_layers: int,
        use_gpu: bool = True,
        spatial_prior: bool = False,
        **module_kwargs,
    ):
        st_adata.obs["_indices"] = np.arange(st_adata.n_obs)
        register_tensor_from_anndata(st_adata, "ind_x", "obs", "_indices")
        register_tensor_from_anndata(st_adata, "x_n", "obsm", "x_n")
        register_tensor_from_anndata(st_adata, "ind_n", "obsm", "ind_n")
        super(DestVI, self).__init__(st_adata)
        self.module = HSTDeconv(
            n_spots=st_adata.n_obs,
            n_labels=cell_type_mapping.shape[0],
            decoder_state_dict=decoder_state_dict,
            px_decoder_state_dict=px_decoder_state_dict,
            px_r=px_r,
            n_genes=st_adata.n_vars,
            n_latent=n_latent,
            n_layers=n_layers,
            n_hidden=n_hidden,
            spatial_prior=spatial_prior,
            **module_kwargs,
        )
        self.cell_type_mapping = cell_type_mapping
        self._model_summary_string = "DestVI Model"
        self.init_params_ = self._get_init_params(locals())


    @classmethod
    def from_rna_model(
        cls,
        st_adata: AnnData,
        sc_model: CondSCVI,
        vamp_prior_p: int = 50,
        **module_kwargs,
    ):
        """
        Alternate constructor for exploiting a pre-trained model on a RNA-seq dataset.

        Parameters
        ----------
        st_adata
            registed anndata object
        sc_model
            trained CondSCVI model
        vamp_prior_p
            number of mixture parameter for VampPrior calculations
        **model_kwargs
            Keyword args for :class:`~scvi.model.DestVI`
        """
        decoder_state_dict = sc_model.module.decoder.state_dict()
        px_decoder_state_dict = sc_model.module.px_decoder.state_dict()
        px_r = sc_model.module.px_r.detach().cpu().numpy()
        mapping = sc_model.scvi_setup_dict_["categorical_mappings"]["_scvi_labels"][
            "mapping"
        ]
        if vamp_prior_p is None:
            mean_vprior = None
            var_vprior = None
        else:
            mean_vprior, var_vprior = sc_model.get_vamp_prior(
                sc_model.adata, p=vamp_prior_p
            )

        return cls(
            st_adata,
            mapping,
            decoder_state_dict,
            px_decoder_state_dict,
            px_r,
            sc_model.module.n_hidden,
            sc_model.module.n_latent,
            sc_model.module.n_layers,
            mean_vprior=mean_vprior,
            var_vprior=var_vprior,
            **module_kwargs,
        )

    @property
    def _plan_class(self):
        return CustomTrainingPlan

    def get_metric(self):
        dl = self._train_dl
        with torch.no_grad():
            rloss_all = []
            rloss = []
            for tensors in dl:
                _, outs_gen, outs_loss = self.module.forward(tensors)
                rloss.append(outs_loss.reconstruction_loss.detach().cpu())
                rloss_all.append(outs_loss.reconstruction_loss_all.detach().cpu())
        rloss = torch.cat(rloss, 0).mean(0)
        rloss_all = torch.cat(rloss_all, 0).mean(0)
        return rloss.item(), rloss_all.numpy()

    def train(
        self,
        max_epochs: Optional[int] = None,
        lr: float = 0.005,
        use_gpu: Optional[bool] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        n_epochs_kl_warmup: int = 50,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        update_dict = {
            "lr": lr,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()

        data_splitter = DataSplitter(
            self.adata,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        self._train_dl = data_splitter()[0]
        training_plan = CustomTrainingPlan(
            self.module, len(data_splitter.train_idx), **plan_kwargs
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()
