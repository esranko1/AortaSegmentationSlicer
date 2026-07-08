"""
Standalone nnU-Net inference for AortaSegmentation.

Runs as its own process (launched via PythonSlicer by AortaSegmentationLogic),
not imported into Slicer directly. Keeping it out-of-process means the multi-minute
inference run doesn't block Slicer's Qt event loop - the caller streams this script's
stdout back into the module's status log instead, pumping the UI after each line.

Usage:
    PythonSlicer run_inference.py --input-dir <dir> --output-dir <dir> --model <model_dir>

Expects input-dir to contain exactly one case as "<case_id>_0000.nii.gz" (nnU-Net's
single-channel inference naming convention) and writes "<case_id>.nii.gz" to output-dir.
"""

import argparse
import shutil
import sys
from pathlib import Path


def ensure_custom_trainer(bundled_dir: Path) -> None:
    """Copies the custom nnUNetTrainerAorta trainer (and its topology-loss dependency)
    into the installed nnunetv2 package. The model was trained with this custom trainer;
    nnUNetPredictor looks trainers up by class name inside nnunetv2's own package tree,
    so a plain `pip install nnunetv2` does not include it."""
    import nnunetv2

    nnunetv2_dir = Path(nnunetv2.__file__).parent
    shutil.copy2(bundled_dir / "topology_losses.py", nnunetv2_dir / "training" / "loss" / "topology_losses.py")
    shutil.copy2(
        bundled_dir / "nnUNetTrainerAorta.py",
        nnunetv2_dir / "training" / "nnUNetTrainer" / "nnUNetTrainerAorta.py",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()

    print("Loading nnU-Net...", flush=True)
    bundledDir = Path(__file__).parent.parent / "Resources" / "nnUNetCustomCode"
    ensure_custom_trainer(bundledDir)

    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    modelFolder = args.model
    foldDirs = sorted(p.name for p in modelFolder.iterdir() if p.is_dir() and p.name.startswith("fold_"))
    useFolds = tuple(int(name.split("_")[1]) for name in foldDirs) if foldDirs else (0,)
    checkpointName = (
        "checkpoint_final.pth"
        if (modelFolder / foldDirs[0] / "checkpoint_final.pth").exists()
        else "checkpoint_best.pth"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=True,
        verbose_preprocessing=False,
    )
    predictor.initialize_from_trained_model_folder(str(modelFolder), use_folds=useFolds, checkpoint_name=checkpointName)

    print("Running segmentation...", flush=True)
    # predict_from_files_sequential (not predict_from_files): nnU-Net's regular
    # predict_from_files always spawns worker processes via
    # multiprocessing.get_context("spawn").Pool(...), even when num_processes=1. Inside
    # Slicer's embedded Python on Windows, sys.executable resolves to Slicer itself, so
    # a spawned "worker" actually relaunches the full Slicer application, which then
    # crashes trying to boot as a module host. predict_from_files_sequential runs
    # entirely in this process instead - no multiprocessing, no spawn. Since this script
    # only ever segments one volume at a time, there's no parallelism to lose anyway.
    predictor.predict_from_files_sequential(
        str(args.input_dir),
        str(args.output_dir),
        save_probabilities=False,
        overwrite=True,
    )
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
