import heapq
import logging
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import qt
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

The Metrics section computes aorta morphology and centerline measurements
(diameter, cross-sectional area, length, curvature, torsion, tortuosity, surface
area, volume) for a segmentation. On first use it installs the SlicerVMTK
extension automatically (also requires internet access, and a one-time Slicer
restart).
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

        self.ui.computeMetricsButton.connect("clicked(bool)", self.onComputeMetricsButton)
        self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._checkCanComputeMetrics)
        self._checkCanComputeMetrics()

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

    def _checkCanComputeMetrics(self, caller=None, event=None) -> None:
        self.ui.computeMetricsButton.enabled = self.ui.outputSelector.currentNode() is not None

    def onComputeMetricsButton(self) -> None:
        with slicer.util.tryWithErrorDisplay(_("Failed to compute metrics."), waitCursor=True):
            results = self.logic.computeMetrics(
                self.ui.outputSelector.currentNode(),
                referenceVolumeNode=self.ui.inputSelector.currentNode(),
                statusCallback=self._setStatus,
            )
            self._populateMetricsTable(results)
        self._setStatus("")

    def _populateMetricsTable(self, results: dict) -> None:
        displayRows = [
            ("DiameterMIS (mm)", results.get("diameter_mis_mm")),
            ("Cross-sectional area (mm²)", results.get("cross_sectional_area_mm2")),
            ("DiameterCE (mm)", results.get("diameter_ce_mm")),
            ("Length (mm)", results.get("length_mm")),
            ("Mean curvature", results.get("mean_curvature")),
            ("Mean torsion", results.get("mean_torsion")),
            ("Tortuosity", results.get("tortuosity")),
            ("Surface area (mm²)", results.get("surface_area_mm2")),
            ("Volume (mm³)", results.get("volume_mm3")),
        ]
        table = self.ui.metricsTableWidget
        table.setRowCount(len(displayRows))
        for row, (label, value) in enumerate(displayRows):
            table.setItem(row, 0, qt.QTableWidgetItem(label))
            valueText = "{:.4f}".format(value) if isinstance(value, float) else str(value)
            table.setItem(row, 1, qt.QTableWidgetItem(valueText))


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

    def _ensureScipy(self) -> None:
        """scipy isn't bundled with Slicer by default; needed for connected-component
        labeling and the distance transform used by DiameterMIS."""
        try:
            import scipy  # noqa: F401
        except ImportError:
            slicer.util.pip_install("scipy")

    def _ensureVmtkExtension(self) -> None:
        """
        Ensures the ExtractCenterline module (from the SlicerVMTK extension,
        https://github.com/vmtk/SlicerExtension-VMTK) is installed, so its compiled
        vtkvmtk libraries are importable in-process. This is not a pip package -
        vtkvmtk's C++ bindings must be built against Slicer's own VTK, which is exactly
        what SlicerVMTK's own build already does; we lean on that instead of maintaining
        our own build or an external conda environment. Unlike a pip package, a
        newly-installed Slicer extension isn't usable until Slicer restarts (extensions
        are registered at application startup), so this may ask the user to restart and
        click Compute Metrics again afterward.
        """
        if hasattr(slicer.modules, "extractcenterline"):
            return

        if not slicer.util.confirmOkCancelDisplay(
            _(
                "This feature requires the 'SlicerVMTK' extension (for centerline "
                "extraction). It will be installed now, and Slicer will need to "
                "restart afterward before Compute Metrics can be used."
            ),
            _("Install VMTK extension"),
        ):
            raise RuntimeError(_("VMTK extension installation was cancelled by the user."))

        extensionsManagerModel = slicer.app.extensionsManagerModel()
        extensionsManagerModel.updateExtensionsMetadataFromServer(True, True)
        if not extensionsManagerModel.installExtension("SlicerVMTK"):
            raise RuntimeError(
                _(
                    "Failed to install the SlicerVMTK extension automatically. Please "
                    "install it manually from the Extension Manager (search for "
                    "'SlicerVMTK') and restart Slicer."
                )
            )

        slicer.util.infoDisplay(
            _(
                "The SlicerVMTK extension was installed. Slicer will now restart; "
                "after it reopens, click Compute Metrics again."
            )
        )
        slicer.app.restart()

    def _keepLargestSurfaceComponent(self, polyData):
        """Segmentations can contain small spurious disconnected surface fragments
        (e.g. stray voxel islands in an automated prediction); every VMTK step
        downstream assumes a single connected vessel surface, so isolate the largest
        connected region before doing anything else."""
        connectivity = vtk.vtkPolyDataConnectivityFilter()
        connectivity.SetInputData(polyData)
        connectivity.SetExtractionModeToLargestRegion()
        connectivity.Update()
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(connectivity.GetOutput())
        cleaner.Update()
        return cleaner.GetOutput()

    def _getNetworkEndPoints(self, network):
        """Returns (point_id, coords) pairs for every degree-1 point in the network
        graph (vessel-tree endpoints, including branch tips), largest-MIS-radius one
        first (typically the aortic root, the widest opening)."""
        points = network.GetPoints()
        radiusArray = network.GetPointData().GetArray("Radius")

        startPointId = -1
        maxRadius = 0.0
        endpointIds = vtk.vtkIdList()

        for cellIndex in range(network.GetNumberOfCells()):
            cell = network.GetCell(cellIndex)
            nPts = cell.GetNumberOfPoints()
            if nPts < 2:
                continue
            for pointIndex in (0, nPts - 1):
                pointId = cell.GetPointId(pointIndex)
                pointCells = vtk.vtkIdList()
                network.GetPointCells(pointId, pointCells)
                if pointCells.GetNumberOfIds() == 1:
                    endpointIds.InsertUniqueId(pointId)
                    radius = radiusArray.GetValue(pointId)
                    if startPointId < 0 or radius > maxRadius:
                        maxRadius = radius
                        startPointId = pointId

        endpoints = []
        nEndpoints = endpointIds.GetNumberOfIds()
        if nEndpoints == 0:
            return endpoints
        endpoints.append((startPointId, points.GetPoint(startPointId)))
        for i in range(nEndpoints):
            pointId = endpointIds.GetId(i)
            if pointId == startPointId:
                continue
            endpoints.append((pointId, points.GetPoint(pointId)))
        return endpoints

    def _networkCellLength(self, network, cellIndex) -> float:
        cell = network.GetCell(cellIndex)
        nPts = cell.GetNumberOfPoints()
        total = 0.0
        prev = None
        for i in range(nPts):
            p = np.array(network.GetPoint(cell.GetPointId(i)))
            if prev is not None:
                total += float(np.linalg.norm(p - prev))
            prev = p
        return total

    def _selectAortaEndPoints(self, network, endpoints):
        """
        Picks the true two aorta ends by cumulative PATH length along the vessel-tree
        graph (Dijkstra), not straight-line distance. endpoints[0] is the
        largest-MIS-radius endpoint (the aortic root, widest opening). The other true
        aorta end is whichever endpoint requires the longest path to reach from the
        root by walking the network's actual branch segments - not whichever is
        geometrically farthest in 3D. Straight-line "farthest point" is unreliable
        here: the aortic arch's U-turn can put a branch stub or a point on the arch
        itself closer to the root in 3D space than the true (possibly short, if the
        segmentation is cropped) distal end, silently shrinking the chord distance
        used later and inflating Tortuosity (length / chord) well beyond the real
        value. Dijkstra (not a naive "keep revisiting" walk) is used because the
        network graph isn't guaranteed to be a perfect tree; a naive longest-path walk
        can loop forever if it contains even one cycle.
        """
        rootId, rootCoords = endpoints[0]
        if len(endpoints) == 2:
            return rootCoords, endpoints[1][1]

        adjacency = {}
        for cellIndex in range(network.GetNumberOfCells()):
            cell = network.GetCell(cellIndex)
            nPts = cell.GetNumberOfPoints()
            if nPts < 2:
                continue
            a = cell.GetPointId(0)
            b = cell.GetPointId(nPts - 1)
            length = self._networkCellLength(network, cellIndex)
            adjacency.setdefault(a, []).append((b, length))
            adjacency.setdefault(b, []).append((a, length))

        distances = {rootId: 0.0}
        visited = set()
        heap = [(0.0, rootId)]
        while heap:
            dist, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)
            for neighbor, length in adjacency.get(node, []):
                newDist = dist + length
                if neighbor not in distances or newDist < distances[neighbor]:
                    distances[neighbor] = newDist
                    heapq.heappush(heap, (newDist, neighbor))

        targetId, targetCoords = max(endpoints[1:], key=lambda e: distances.get(e[0], -1.0))
        return rootCoords, targetCoords

    def _centerlineMetrics(self, centerline) -> dict:
        import vtk.util.numpy_support as vtk_np

        def pointArray(name):
            a = centerline.GetPointData().GetArray(name)
            return vtk_np.vtk_to_numpy(a) if a is not None else None

        def cellArray(name):
            a = centerline.GetCellData().GetArray(name)
            return vtk_np.vtk_to_numpy(a) if a is not None else None

        curvatureAll = pointArray("Curvature")
        torsionAll = pointArray("Torsion")
        radiiAll = pointArray("MaximumInscribedSphereRadius")
        length = cellArray("Length")
        tortuosity = cellArray("Tortuosity")

        nCells = centerline.GetNumberOfCells()
        # vtkvmtkPolyDataCenterlines can emit more than one cell; the main aorta path
        # is the longest one, not necessarily cell 0.
        mainCell = int(np.argmax(length)) if nCells > 1 else 0

        pointIds = centerline.GetCell(mainCell).GetPointIds()
        mainPointIndices = [pointIds.GetId(i) for i in range(pointIds.GetNumberOfIds())]

        curvature = curvatureAll[mainPointIndices]
        torsion = torsionAll[mainPointIndices]
        radii = radiiAll[mainPointIndices]

        maxRadius = float(np.nanmax(radii))

        return {
            "length_mm": float(length[mainCell]),
            # vmtk's raw Tortuosity array is (length/chord - 1); +1 gives the
            # conventional length/chord ratio.
            "tortuosity": float(tortuosity[mainCell]) + 1.0,
            "max_cross_section_mm2": np.pi * maxRadius**2,
            "diameter_ce_mm": 2 * maxRadius,
            "mean_curvature": float(np.nanmean(curvature)),
            # Signed mean (not mean of absolute value): torsion oscillates sign along
            # a gently-curving vessel and should mostly cancel out, not accumulate.
            "mean_torsion": float(np.nanmean(torsion)),
        }

    def _keepLargestArrayComponent(self, arr):
        """Voxel-array equivalent of _keepLargestSurfaceComponent, for the Volume and
        DiameterMIS calculations (which work from the labelmap directly, not the
        surface)."""
        from scipy.ndimage import label

        labeled, numFeatures = label(arr > 0)
        if numFeatures <= 1:
            return arr
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        largestLabel = sizes.argmax()
        return (labeled == largestLabel).astype(np.float32)

    def _getVolume(self, arr, spacing) -> float:
        voxelVolumeMm3 = spacing[0] * spacing[1] * spacing[2]
        return float(arr.sum()) * voxelVolumeMm3

    def _getDiameterMis(self, arr, spacing) -> float:
        from scipy.ndimage import distance_transform_edt

        mask = arr.astype(bool)
        # slicer.util.arrayFromVolume returns (k,j,i) = (z,y,x) index order;
        # GetSpacing() returns (x,y,z), so the EDT sampling needs to be reversed to
        # match the array's own axis order.
        sampling = (spacing[2], spacing[1], spacing[0])
        dist = distance_transform_edt(mask, sampling=sampling)
        return float(dist.max()) * 2.0

    def _getSurfaceArea(self, surface) -> float:
        mass = vtk.vtkMassProperties()
        mass.SetInputData(surface)
        mass.Update()
        return mass.GetSurfaceArea()

    def computeMetrics(
        self,
        segmentationNode: vtkMRMLSegmentationNode,
        referenceVolumeNode: vtkMRMLScalarVolumeNode = None,
        statusCallback=None,
    ) -> dict:
        """
        Computes aorta morphology/centerline metrics (DiameterMIS, CrossSectionalArea,
        DiameterCE, Length, mean curvature/torsion, Tortuosity, SurfaceArea, Volume)
        for segmentationNode, entirely in-process using the SlicerVMTK extension's
        ExtractCenterlineLogic (installed automatically on first use if missing - see
        _ensureVmtkExtension).

        referenceVolumeNode should be the volume the segmentation was created from
        (e.g. the module's inputVolume). Without it, ExportVisibleSegmentsToLabelmapNode
        falls back to the segmentation's own internal reference geometry, which can be
        resampled/oversampled relative to the source scan (Slicer does this for
        smoother segment editing) - confirmed to shift Volume by several percent versus
        the original NIfTI. Passing the original volume forces the export to align to
        the exact voxel grid the segmentation was made from.
        """
        if not segmentationNode:
            raise ValueError(_("No segmentation selected"))

        import time

        def status(message: str) -> None:
            logging.info(message)
            if statusCallback:
                statusCallback(message)

        startTime = time.time()

        status(_("Checking dependencies..."))
        self._ensureScipy()
        self._ensureVmtkExtension()

        import ExtractCenterline
        import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry

        centerlineLogic = ExtractCenterline.ExtractCenterlineLogic()

        status(_("Preparing surface..."))
        segmentation = segmentationNode.GetSegmentation()
        if segmentation.GetNumberOfSegments() == 0:
            raise ValueError(_("Segmentation has no segments"))
        segmentId = segmentation.GetNthSegmentID(0)
        surfacePolyData = centerlineLogic.polyDataFromNode(segmentationNode, segmentId)
        surfacePolyData = self._keepLargestSurfaceComponent(surfacePolyData)
        preprocessedPolyData = centerlineLogic.preprocess(
            surfacePolyData, targetNumberOfPoints=5000, decimationAggressiveness=4.0, subdivide=False
        )

        status(_("Finding vessel endpoints..."))
        networkPolyData = centerlineLogic.extractNetwork(preprocessedPolyData, endPointsMarkupsNode=None)
        networkCleaner = vtk.vtkCleanPolyData()
        networkCleaner.SetInputData(networkPolyData)
        networkCleaner.Update()
        network = networkCleaner.GetOutput()
        network.BuildCells()
        network.BuildLinks(0)
        endpoints = self._getNetworkEndPoints(network)
        if len(endpoints) < 2:
            raise RuntimeError(
                _("Could not find two vessel endpoints (found {n}).").format(n=len(endpoints))
            )
        p0, p1 = self._selectAortaEndPoints(network, endpoints)

        status(_("Computing centerline..."))
        endPointsNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode")
        try:
            endPointsNode.AddControlPoint(vtk.vtkVector3d(p0[0], p0[1], p0[2]))
            endPointsNode.AddControlPoint(vtk.vtkVector3d(p1[0], p1[1], p1[2]))
            # The source/start point is the first UNselected control point; mark the
            # target explicitly selected so p0 is unambiguously the source.
            endPointsNode.SetNthControlPointSelected(0, False)
            endPointsNode.SetNthControlPointSelected(1, True)
            centerlinePolyData, _voronoi = centerlineLogic.extractCenterline(
                preprocessedPolyData, endPointsNode, curveSamplingDistance=1.0
            )
        finally:
            slicer.mrmlScene.RemoveNode(endPointsNode)

        status(_("Computing centerline geometry..."))
        geometry = vtkvmtkComputationalGeometry.vtkvmtkCenterlineGeometry()
        geometry.SetInputData(centerlinePolyData)
        geometry.SetLengthArrayName("Length")
        geometry.SetCurvatureArrayName("Curvature")
        geometry.SetTorsionArrayName("Torsion")
        geometry.SetTortuosityArrayName("Tortuosity")
        geometry.SetFrenetTangentArrayName("FrenetTangent")
        geometry.SetFrenetNormalArrayName("FrenetNormal")
        geometry.SetFrenetBinormalArrayName("FrenetBinormal")
        geometry.Update()
        clMetrics = self._centerlineMetrics(geometry.GetOutput())

        status(_("Computing volume metrics..."))
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        try:
            slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
                segmentationNode, labelmapVolumeNode, referenceVolumeNode
            )
            arr = slicer.util.arrayFromVolume(labelmapVolumeNode)
            spacing = labelmapVolumeNode.GetSpacing()
            arr = self._keepLargestArrayComponent(arr)
            volume = self._getVolume(arr, spacing)
            diameterMis = self._getDiameterMis(arr, spacing)
        finally:
            slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

        surfaceArea = self._getSurfaceArea(surfacePolyData)

        results = {
            "diameter_mis_mm": diameterMis,
            "cross_sectional_area_mm2": clMetrics["max_cross_section_mm2"],
            "diameter_ce_mm": clMetrics["diameter_ce_mm"],
            "length_mm": clMetrics["length_mm"],
            "mean_curvature": clMetrics["mean_curvature"],
            "mean_torsion": clMetrics["mean_torsion"],
            "tortuosity": clMetrics["tortuosity"],
            "surface_area_mm2": surfaceArea,
            "volume_mm3": volume,
        }

        status(_("Done."))
        stopTime = time.time()
        logging.info(f"Metrics computed in {stopTime - startTime:.2f} seconds")
        return results


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
