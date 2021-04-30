import logging
from typing import List, Optional, OrderedDict

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import isspmatrix
from scvi.data import register_tensor_from_anndata
from scvi.dataloaders import AnnDataLoader
from scvi.external.condscvi._model import CondSCVI
from scvi.lightning import TrainingPlan
from scvi.model.base import BaseModelClass
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from . import HSTDeconv

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


class DestVISpatial(BaseModelClass):
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
        sc_state_dict: List[OrderedDict],
        n_latent: int,
        n_layers: int,
        n_hidden: int,
        use_gpu: bool = True,
        spatial_prior: bool = False,
        **module_kwargs,
    ):
        st_adata.obs["_indices"] = np.arange(st_adata.n_obs)
        register_tensor_from_anndata(st_adata, "ind_x", "obs", "_indices")
        register_tensor_from_anndata(st_adata, "x_n", "obsm", "x_n")
        register_tensor_from_anndata(st_adata, "ind_n", "obsm", "ind_n")
        super(DestVI, self).__init__(st_adata, use_gpu=use_gpu)
        self.module = HSTDeconv(
            n_spots=st_adata.n_obs,
            n_labels=cell_type_mapping.shape[0],
            sc_state_dict=sc_state_dict,
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
        use_gpu: bool = True,
        **model_kwargs,
    ):
        """
        Alternate constructor for exploiting a pre-trained model on RNA-seq data.

        Parameters
        ----------
        st_adata
            registed anndata object
        sc_model
            trained RNADeconv model
        use_gpu
            Use the GPU or not.
        **model_kwargs
            Keyword args for :class:`~scvi.external.DestVI`
        """
        state_dict = (
            sc_model.module.decoder.state_dict(),
            sc_model.module.px_decoder.state_dict(),
            sc_model.module.px_r.detach().cpu().numpy(),
        )

        return cls(
            st_adata,
            sc_model.scvi_setup_dict_["categorical_mappings"]["_scvi_labels"][
                "mapping"
            ],
            state_dict,
            sc_model.module.n_latent,
            sc_model.module.n_layers,
            sc_model.module.n_hidden,
            use_gpu=use_gpu,
            **model_kwargs,
        )

    def construct_loaders(
        self,
        use_gpu: Optional[bool] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        train_indices=None,
        test_indices=None,
    ):
        from scvi import settings
        from scvi.lightning import Trainer

        if use_gpu is None:
            use_gpu = self.use_gpu
        else:
            use_gpu = use_gpu and torch.cuda.is_available()
        gpus = 1 if use_gpu else None
        pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and use_gpu) else False
        )

        if train_indices is None:
            train_dl, val_dl, test_dl = self._train_test_val_split(
                self.adata,
                train_size=train_size,
                validation_size=validation_size,
                pin_memory=pin_memory,
                batch_size=batch_size,
            )
        else:
            dl_kwargs = dict(
                pin_memory=pin_memory,
                batch_size=batch_size,
            )
            logging.info("Using custom train val split")
            train_dl = self._make_scvi_dl(
                self.adata, indices=train_indices, shuffle=True, **dl_kwargs
            )
            test_dl = self._make_scvi_dl(
                self.adata, indices=test_indices, shuffle=False, **dl_kwargs
            )
            val_dl = self._make_scvi_dl(
                self.adata, indices=[], shuffle=False, **dl_kwargs
            )
        self.train_indices_ = train_dl.indices
        self.test_indices_ = test_dl.indices
        self.validation_indices_ = val_dl.indices

        self._train_dl = train_dl
        self._test_dl = test_dl
        self._val_dl = val_dl


    def train(
        self,
        max_epochs: Optional[int] = None,
        use_gpu: Optional[bool] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        plan_kwargs: Optional[dict] = None,
        plan_class: Optional[None] = None,
        train_indices=None,
        test_indices=None,
        **kwargs,
    ):
        from scvi import settings
        from scvi.lightning import Trainer

        if use_gpu is None:
            use_gpu = self.use_gpu
        else:
            use_gpu = use_gpu and torch.cuda.is_available()
        gpus = 1 if use_gpu else None
        pin_memory = (
            True if (settings.dl_pin_memory_gpu_training and use_gpu) else False
        )

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

        self.trainer = Trainer(
            max_epochs=max_epochs,
            gpus=gpus,
            **kwargs,
        )

        if train_indices is None:
            train_dl, val_dl, test_dl = self._train_test_val_split(
                self.adata,
                train_size=train_size,
                validation_size=validation_size,
                pin_memory=pin_memory,
                batch_size=batch_size,
            )
        else:
            dl_kwargs = dict(
                pin_memory=pin_memory,
                batch_size=batch_size,
            )
            logging.info("Using custom train val split")
            train_dl = self._make_scvi_dl(
                self.adata, indices=train_indices, shuffle=True, **dl_kwargs
            )
            test_dl = self._make_scvi_dl(
                self.adata, indices=test_indices, shuffle=False, **dl_kwargs
            )
            val_dl = self._make_scvi_dl(
                self.adata, indices=[], shuffle=False, **dl_kwargs
            )
        self.train_indices_ = train_dl.indices
        self.test_indices_ = test_dl.indices
        self.validation_indices_ = val_dl.indices

        self._train_dl = train_dl
        self._test_dl = test_dl
        self._val_dl = val_dl

        if plan_class is None:
            plan_class = self._plan_class

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        self._pl_task = plan_class(self.module, len(self.train_indices_), **plan_kwargs)

        if train_size == 1.0:
            # circumvent the empty data loader problem if all dataset used for training
            self.trainer.fit(self._pl_task, train_dl)
        else:
            self.trainer.fit(self._pl_task, train_dl, val_dl)
        try:
            self.history_ = self.trainer.logger.history
        except AttributeError:
            self.history_ = None
        self.module.eval()
        if use_gpu:
            self.module.cuda()
        self.is_trained_ = True

    @property
    def _plan_class(self):
        return CustomTrainingPlan

    @property
    def _data_loader_cls(self):
        return AnnDataLoader

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


    def get_proportions(self, dataset=None, keep_noise=False) -> np.ndarray:
        """
        Returns the estimated cell type proportion for the spatial data.

        Shape is n_cells x n_labels OR n_cells x (n_labels + 1) if keep_noise.

        Parameters
        ----------
        keep_noise
            whether to account for the noise term as a standalone cell type in the proportion estimate.
        """
        column_names = self.cell_type_mapping
        if keep_noise:
            column_names = np.append(column_names, "noise_term")

        if self.module.amortization in ["both", "proportion"]:
            data = dataset.X
            if isspmatrix(data):
                data = data.A
            dl = DataLoader(
                TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=128
            )  # create your dataloader
            prop_ = []
            for tensors in dl:
                prop_local = self.module.get_proportions(x=tensors[0])
                prop_ += [prop_local.cpu()]
            data = np.array(torch.cat(prop_))
        else:
            data = self.module.get_proportions(keep_noise=keep_noise)
        return pd.DataFrame(
            data=data,
            columns=column_names,
            index=self.adata.obs.index,
        )

    def get_gamma(self, dataset=None) -> np.ndarray:
        """
        Returns the estimated cell-type specific latent space for the spatial data.

        Shape is n_cells x n_latent x n_labels.
        """
        if self.module.amortization in ["both", "latent"]:
            data = dataset.X
            if isspmatrix(dataset.X):
                data = dataset.X.A
            dl = DataLoader(
                TensorDataset(torch.tensor(data, dtype=torch.float32)), batch_size=128
            )  # create your dataloader
            gamma_ = []
            for tensors in dl:
                gamma_local = self.module.get_gamma(x=tensors[0])
                gamma_ += [gamma_local.cpu()]
            data = np.array(torch.cat(gamma_, dim=-1))
        else:
            data = self.module.get_gamma()
        return np.transpose(data, (2, 0, 1))

    @torch.no_grad()
    def get_scale_for_ct(
        self,
        x: Optional[np.ndarray] = None,
        ind_x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""
        Return the scaled parameter of the NB for every cell in queried cell types.

        Parameters
        ----------
        x
            gene expression data
        ind_x
            indices
        y
            cell types

        Returns
        -------
        gene_expression
        """
        if self.is_trained_ is False:
            raise RuntimeError("Please train the model first.")

        dl = DataLoader(
            TensorDataset(
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(ind_x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
            ),
            batch_size=128,
        )  # create your dataloader
        scale = []
        for tensors in dl:
            px_scale = self.module.get_ct_specific_expression(
                tensors[0], tensors[1], tensors[2]
            )
            scale += [px_scale.cpu()]
        return np.array(torch.cat(scale))
