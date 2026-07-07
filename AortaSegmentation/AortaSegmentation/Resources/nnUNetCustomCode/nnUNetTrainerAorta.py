import re

import torch
from torch import nn
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, isfile, save_json, load_json

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainer_onlyMirror01
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.topology_losses import AortaTopologyLoss
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

PATIENT_ID_PATTERN = re.compile(r"PT\d+")


def patient_id(case_identifier: str) -> str:
    """Extracts the PT### patient number so multiple scans of the same
    patient (e.g. PT011_M and PT011_M_F4) are always kept together."""
    match = PATIENT_ID_PATTERN.search(case_identifier)
    if match is None:
        raise ValueError(f"Could not extract patient id (PT###) from case identifier: {case_identifier}")
    return match.group(0)


class _PerScaleLoss(nn.Module):
    """
    Deep-supervision wrapper that applies topology losses only at the
    highest-resolution scale (index 0) and uses only the base loss for
    downsampled scales — avoids computing expensive Hessian on small crops.
    """

    def __init__(self, full_res_loss: nn.Module, base_loss: nn.Module, weight_factors):
        super().__init__()
        assert any(w != 0 for w in weight_factors)
        self.full_res_loss = full_res_loss
        self.base_loss = base_loss
        self.weight_factors = tuple(weight_factors)

    def forward(self, *args):
        assert all(isinstance(i, (tuple, list)) for i in args)
        total = None
        for i, inputs in enumerate(zip(*args)):
            w = self.weight_factors[i]
            if w == 0.0:
                continue
            loss_fn = self.full_res_loss if i == 0 else self.base_loss
            term = w * loss_fn(*inputs)
            total = term if total is None else total + term
        return total


class nnUNetTrainerAorta(nnUNetTrainer_onlyMirror01):
    """
    nnUNet trainer for aorta MRI segmentation.

    Changes vs. default trainer:
    - Mirrors only on axes 0 and 1 (anatomically consistent for aorta)
    - Adds clDice loss (connectivity / topology)
    - Adds Hessian shape loss (penalises bulges and spurious branch junctions)
    Both topology terms are applied only at full resolution in deep supervision.

    Weights can be tuned via the class attributes below.
    """

    WEIGHT_CLDICE: float = 0.2
    CLDICE_ITERS: int = 5

    def _build_loss(self):
        assert not self.label_manager.has_regions, \
            "nnUNetTrainerAorta expects a single binary foreground label, not regions."

        base_loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {},
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss,
        )

        if self._do_i_compile():
            base_loss.dc = torch.compile(base_loss.dc)

        full_res_loss = AortaTopologyLoss(
            base_loss=base_loss,
            weight_cldice=self.WEIGHT_CLDICE,
            cldice_iters=self.CLDICE_ITERS,
        )

        if not self.enable_deep_supervision:
            return full_res_loss

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        if self.is_ddp and not self._do_i_compile():
            weights[-1] = 1e-6
        else:
            weights[-1] = 0
        weights = weights / weights.sum()

        return _PerScaleLoss(full_res_loss, base_loss, weights)

    def do_split(self):
        """
        Same as nnUNetTrainer.do_split(), except the 5-fold split is generated with GroupKFold on
        the PT### patient number instead of plain KFold on case identifiers. This guarantees that
        cases sharing the same patient number (e.g. PT011_M and PT011_M_F4, a follow-up scan of the
        same patient) always land together in either train or val, never split across the two -
        preventing patient-level data leakage between folds.
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        if self.fold == "all":
            case_identifiers = self.dataset_class.get_identifiers(self.preprocessed_dataset_folder)
            tr_keys = case_identifiers
            val_keys = tr_keys
        else:
            splits_file = join(self.preprocessed_dataset_folder_base, "splits_final.json")
            dataset = self.dataset_class(self.preprocessed_dataset_folder,
                                         identifiers=None,
                                         folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage)
            if not isfile(splits_file):
                self.print_to_log_file("Creating new 5-fold cross-validation split (grouped by PT### patient id)...")
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                groups = [patient_id(k) for k in all_keys_sorted]
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5, groups=groups)
                save_json(splits, splits_file)
            else:
                self.print_to_log_file("Using splits from existing split file:", splits_file)
                splits = load_json(splits_file)
                self.print_to_log_file(f"The split file contains {len(splits)} splits.")

            self.print_to_log_file("Desired fold for training: %d" % self.fold)
            if self.fold < len(splits):
                tr_keys = splits[self.fold]['train']
                val_keys = splits[self.fold]['val']
                self.print_to_log_file("This split has %d training and %d validation cases."
                                       % (len(tr_keys), len(val_keys)))
            else:
                raise RuntimeError(
                    f"You requested fold {self.fold} but the split file only contains {len(splits)} folds. "
                    f"Delete splits_final.json to regenerate it, or pick a fold in range."
                )

            if any([i in val_keys for i in tr_keys]):
                self.print_to_log_file('WARNING: Some validation cases are also in the training set. Please check the '
                                       'splits.json or ignore if this is intentional.')
            tr_patients = {patient_id(k) for k in tr_keys}
            val_patients = {patient_id(k) for k in val_keys}
            overlap = tr_patients & val_patients
            if overlap:
                raise RuntimeError(
                    f"Patient-level leakage detected between train and val for fold {self.fold}: "
                    f"patients {sorted(overlap)} have cases in both sets. Delete splits_final.json "
                    f"(it may predate the grouped-split logic) and rerun to regenerate it."
                )

            self.print_to_log_file(f"--- fold {self.fold} case assignment ---")
            self.print_to_log_file(f"TRAINING cases ({len(tr_keys)}):")
            for k in sorted(tr_keys):
                self.print_to_log_file(f"    {k}")
            self.print_to_log_file(f"VALIDATION cases ({len(val_keys)}):")
            for k in sorted(val_keys):
                self.print_to_log_file(f"    {k}")
            self.print_to_log_file(f"VALIDATION patients ({len(val_patients)}): {sorted(val_patients)}")
            self.print_to_log_file("--- end fold case assignment ---")
        return tr_keys, val_keys


class nnUNetTrainerAorta10Epochs(nnUNetTrainerAorta):
    """Quick sanity-check trainer — runs for 10 epochs only."""
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 10
