import logging
import shutil
from pathlib import Path
from typing import Optional

import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLSegmentationNode

# Trained nnU-Net v2 (3d_fullres, 5 folds), single-channel MRI, foreground="aorta".
# Package contains only dataset.json/plans.json/dataset_fingerprint.json plus
# fold_0..fold_4/checkpoint_final.pth (validation dumps and checkpoint_best.pth are
# stripped out to keep the download small). Mean validation Dice (fold 0): 0.90.
MODEL_DOWNLOAD_URL = "https://github.com/esranko1/AortaSegmentationSlicer/releases/download/model-v1/aorta_nnunet_model.zip"
MODEL_SHA256 = "1337ef986d706da71cf2b1c4eb4aba924bfd1f6780fbb8529c08d79183dda3fd"


#
# AortaSegmentation
#


class AortaSegmentation(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Aorta Segmentation")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Segmentation")]
        self.parent.dependencies = []
        self.parent.contributors = ["Eszter Sranko"]
        self.parent.helpText = _("""
Segments the aorta from a single MRI volume using a pretrained nnU-Net model.
Select an input volume and an output segmentation, then click Apply.
On first use the module downloads its Python dependencies (PyTorch, nnU-Net) and
the trained model weights, which requires an internet connection.
""")
        self.parent.acknowledgementText = _("")


#
# AortaSegmentationParameterNode
#


@parameterNodeWrapper
class AortaSegmentationParameterNode:
    """
    inputVolume - the MRI volume to segment.
    outputSegmentation - the segmentation node the result is written into.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    outputSegmentation: vtkMRMLSegmentationNode


#
# AortaSegmentationWidget
#


class AortaSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        ScriptedLoadableModuleWidget.setup(self)

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/AortaSegmentation.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        self.logic = AortaSegmentationLogic()

        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        self.initializeParameterNode()

    def cleanup(self) -> None:
        self.removeObservers()

    def enter(self) -> None:
        self.initializeParameterNode()

    def exit(self) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        self.setParameterNode(self.logic.getParameterNode())

        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[AortaSegmentationParameterNode]) -> None:
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.outputSegmentation:
            self.ui.applyButton.toolTip = _("Run aorta segmentation")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select an input volume and an output segmentation")
            self.ui.applyButton.enabled = False

    def _setStatus(self, message: str) -> None:
        self.ui.statusLabel.text = message
        slicer.app.processEvents()

    def onApplyButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            self.logic.process(
                self.ui.inputSelector.currentNode(),
                self.ui.outputSelector.currentNode(),
                statusCallback=self._setStatus,
            )
        self._setStatus("")


#
# AortaSegmentationLogic
#


class AortaSegmentationLogic(ScriptedLoadableModuleLogic):
    """Implements the actual computation. Can be used without the GUI widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return AortaSegmentationParameterNode(super().getParameterNode())

    def _ensureDependencies(self):
        """Installs PyTorch and nnU-Net into Slicer's Python environment if missing."""
        try:
            import torch
            import nnunetv2  # noqa: F401
        except ImportError:
            if not slicer.util.confirmOkCancelDisplay(
                _(
                    "This module requires PyTorch and nnU-Net. They will be downloaded and "
                    "installed into Slicer's Python environment now (one-time, several hundred MB, "
                    "requires internet access)."
                ),
                _("Install dependencies"),
            ):
                raise RuntimeError(_("Dependency installation was cancelled by the user."))
            self._installTorch()
            slicer.util.pip_install("nnunetv2")
            import torch
            import nnunetv2  # noqa: F401

    def _installTorch(self) -> None:
        """Installs PyTorch, requesting a CUDA build if an NVIDIA GPU is present.
        A plain `pip install torch` can silently resolve a CPU-only wheel even on a
        machine with a capable GPU (observed on a machine with an RTX 3090): inference
        then runs correctly but far slower, with no obvious indication why. Checking for
        `nvidia-smi` and requesting the matching CUDA wheel avoids that trap."""
        if shutil.which("nvidia-smi") is not None:
            slicer.util.pip_install("torch --index-url https://download.pytorch.org/whl/cu121")
        else:
            slicer.util.pip_install("torch")

    def _ensureModel(self) -> Path:
        """Downloads and caches the trained nnU-Net model folder, returns its path."""
        modelDir = Path(slicer.app.cachePath) / "AortaSegmentation" / "model"
        if not (modelDir / "dataset.json").exists():
            modelDir.mkdir(parents=True, exist_ok=True)
            zipPath = modelDir.parent / "model.zip"
            logging.info(f"Downloading aorta segmentation model from {MODEL_DOWNLOAD_URL}")
            slicer.util.downloadFile(MODEL_DOWNLOAD_URL, str(zipPath), checksum=f"SHA256:{MODEL_SHA256}")
            slicer.util.extractArchive(str(zipPath), str(modelDir))
            zipPath.unlink()
        return modelDir

    def _runInferenceSubprocess(self, inputDir: Path, outputDir: Path, modelFolder: Path, status) -> None:
        """Runs Scripts/run_inference.py as a separate process (via Slicer's own Python)
        instead of calling nnU-Net in-process. Streaming its stdout back line-by-line -
        each line passed to status(), which pumps Qt's event loop - is what keeps Slicer
        responsive during the multi-minute run instead of appearing to hang; the same
        pattern SlicerTotalSegmentator uses."""
        pythonSlicerExecutablePath = shutil.which("PythonSlicer")
        if not pythonSlicerExecutablePath:
            raise RuntimeError(_("PythonSlicer executable not found"))

        scriptPath = Path(__file__).parent / "Scripts" / "run_inference.py"
        cmd = [
            pythonSlicerExecutablePath,
            str(scriptPath),
            "--input-dir", str(inputDir),
            "--output-dir", str(outputDir),
            "--model", str(modelFolder),
        ]

        proc = slicer.util.launchConsoleProcess(cmd)
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            status(line.rstrip())
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(_("Inference subprocess failed with exit code {code}").format(code=proc.returncode))

    def process(
        self,
        inputVolume: vtkMRMLScalarVolumeNode,
        outputSegmentation: vtkMRMLSegmentationNode,
        statusCallback=None,
    ) -> None:
        """
        Runs aorta segmentation on a single MRI volume and writes the result into
        outputSegmentation (as a segmentation, not a labelmap/volume).
        """

        if not inputVolume or not outputSegmentation:
            raise ValueError("Input volume or output segmentation is invalid")

        import time

        def status(message: str) -> None:
            logging.info(message)
            if statusCallback:
                statusCallback(message)

        startTime = time.time()
        status(_("Checking dependencies..."))
        self._ensureDependencies()

        status(_("Checking model weights..."))
        modelFolder = self._ensureModel()

        tempDir = Path(slicer.util.tempDirectory())
        inputDir = tempDir / "input"
        outputDir = tempDir / "output"
        inputDir.mkdir()
        outputDir.mkdir()

        try:
            caseId = "case"
            inputFile = inputDir / f"{caseId}_0000.nii.gz"
            # saveNode() handles the Slicer RAS -> NIfTI LPS conversion; exporting the
            # voxel array directly would silently flip the volume vs. what nnU-Net expects.
            slicer.util.saveNode(inputVolume, str(inputFile))

            status(_("Running aorta segmentation (this can take a few minutes)..."))
            self._runInferenceSubprocess(inputDir, outputDir, modelFolder, status)

            status(_("Loading result..."))
            outputFile = outputDir / f"{caseId}.nii.gz"
            labelmapVolumeNode = slicer.util.loadLabelVolume(str(outputFile))
            try:
                slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
                    labelmapVolumeNode, outputSegmentation
                )
                segmentation = outputSegmentation.GetSegmentation()
                if segmentation.GetNumberOfSegments() > 0:
                    segmentation.GetNthSegment(0).SetName("Aorta")
                outputSegmentation.CreateClosedSurfaceRepresentation()
            finally:
                slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        finally:
            shutil.rmtree(tempDir, ignore_errors=True)

        status(_("Done."))
        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime - startTime:.2f} seconds")


#
# AortaSegmentationTest
#


class AortaSegmentationTest(ScriptedLoadableModuleTest):
    """
    This is the test case for the scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_AortaSegmentation1()

    def test_AortaSegmentation1(self):
        """Smoke test: checks that the parameter node and Apply-button gating work as
        expected. A full run of process() needs a real MRI volume and the downloaded
        model, so it is not exercised here."""

        self.delayDisplay("Starting the test")

        logic = AortaSegmentationLogic()
        parameterNode = logic.getParameterNode()
        self.assertIsNone(parameterNode.inputVolume)
        self.assertIsNone(parameterNode.outputSegmentation)

        self.delayDisplay("Test passed")
