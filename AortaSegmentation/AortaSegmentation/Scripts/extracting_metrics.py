"""
Standalone aorta morphology/centerline metrics for AortaSegmentation.

Runs as its own process (launched via AortaSegmentationLogic.computeMetrics using the
user-configured VMTK conda environment's python.exe), not imported into Slicer directly.
Slicer's embedded Python cannot host VMTK (vtkvmtk's compiled bindings must match a
specific VTK build, which a plain pip install into Slicer's Python can't guarantee),
so this script runs out-of-process against a separate conda environment that has
`vmtk` installed, the same way run_inference.py runs nnU-Net out-of-process.

Usage:
    Single case:
        python extracting_metrics.py --seg <segmentation.nii.gz> --out-json <results.json> [--out-dir <dir>]
        Writes the computed metrics to --out-json and, alongside them, four .vtp files
        (surface, centerline, vessel-tree endpoints, chosen source/target points) into
        --out-dir (defaults to --seg's own directory) for visual inspection.

    Batch (e.g. a whole labelsTr folder):
        python extracting_metrics.py --seg-dir <folder> --out-csv <results.csv> [--write-vtp]
        Processes every .nii.gz in the folder and writes one row per case to
        --out-csv. Per-case failures are caught and recorded (with the error message)
        rather than stopping the batch. The CSV is rewritten after every case, so
        rerunning the same command resumes rather than reprocessing already-completed
        cases. .vtp visualization files are skipped by default in batch mode (pass
        --write-vtp to also generate them for every case).
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt, label


def _require_vtk():
    try:
        import vtk
        import vtk.util.numpy_support as vtk_np
        return vtk, vtk_np
    except ImportError:
        sys.exit("vtk not found. Install with: pip install vtk")


def _require_vmtk():
    try:
        import vmtk.vmtkscripts as vmtkscripts
        return vmtkscripts
    except ImportError:
        sys.exit("vmtk not found. Install with: conda install -c vmtk vmtk")


def _require_vtkvmtk():
    try:
        from vmtk import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry
        from vmtk import vtkvmtkMiscPython as vtkvmtkMisc
        return vtkvmtkComputationalGeometry, vtkvmtkMisc
    except ImportError:
        sys.exit("vtkvmtk python bindings not found. Install with: conda install -c vmtk vmtk")


# ---------------------------------------------------------------------------
# Read NIfTI (VTK's own reader, no SimpleITK needed)
# ---------------------------------------------------------------------------

def read_nifti(nifti_path):
    vtk, vtk_np = _require_vtk()

    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(nifti_path))
    reader.Update()
    image = reader.GetOutput()
    qform = reader.GetQFormMatrix()

    nx, ny, nz = image.GetDimensions()
    spacing = image.GetSpacing()
    scalars = image.GetPointData().GetScalars()
    arr = vtk_np.vtk_to_numpy(scalars).reshape(nz, ny, nx).astype(np.float32)

    return image, arr, spacing, qform


def keep_largest_component(image, arr):
    """
    The segmentation mask can contain small spurious disconnected voxel
    islands (common in nnU-Net predictions). Marching cubes turns each one
    into its own disconnected surface piece mixed into the same mesh, which
    breaks every VMTK step downstream. Keep only the largest connected
    component of the mask.
    """
    _, vtk_np = _require_vtk()

    labeled, num_features = label(arr > 0)
    if num_features <= 1:
        return image, arr

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    largest_label = sizes.argmax()
    cleaned = (labeled == largest_label).astype(np.float32)

    print('DEBUG: mask had {} connected components; kept the largest ({} of {} voxels)'.format(
        num_features, int(cleaned.sum()), int((arr > 0).sum())), flush=True)

    vtk_arr = vtk_np.numpy_to_vtk(cleaned.ravel(order='C'), deep=True)
    image.GetPointData().SetScalars(vtk_arr)
    return image, cleaned


def get_surface_mesh(image, qform):
    vtk, _ = _require_vtk()

    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(image)
    mc.SetValue(0, 0.5)
    mc.Update()

    if qform is None:
        return mc.GetOutput()

    transform = vtk.vtkTransform()
    transform.SetMatrix(qform)
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(mc.GetOutputPort())
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    return transform_filter.GetOutput()


def smooth_surface(surface_poly_data, iterations=20, pass_band=0.05):
    """
    Windowed-sinc smoothing to remove marching-cubes staircase artifacts.
    Slicer's own segmentation-to-closed-surface conversion applies smoothing
    by default before the real VMTK extension ever sees a surface; raw
    marching cubes here with no equivalent smoothing pass produces a visibly
    jaggier mesh. Confirmed by comparing this pipeline's output against the
    real VMTK Slicer extension on the same manually-segmented mask: without
    this step, SurfaceArea reads too high and DiameterCE/CrossSectionalArea/
    Curvature/Torsion/Tortuosity all read too far in the direction a jagged
    (rather than smooth) vessel wall would push them.
    """
    vtk, _ = _require_vtk()
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(surface_poly_data)
    smoother.SetNumberOfIterations(iterations)
    smoother.SetPassBand(pass_band)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


# ---------------------------------------------------------------------------
# Volume / surface area / DiameterMIS
# ---------------------------------------------------------------------------

def get_volume(arr, spacing):
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    return float(arr.sum()) * voxel_volume_mm3


def get_surface_area(surface):
    vtk, _ = _require_vtk()
    mass = vtk.vtkMassProperties()
    mass.SetInputData(surface)
    mass.Update()
    return mass.GetSurfaceArea()


def get_diameter_mis(arr, spacing):
    mask = arr.astype(bool)
    sampling = (spacing[2], spacing[1], spacing[0])
    dist = distance_transform_edt(mask, sampling=sampling)
    return float(dist.max()) * 2.0


# ---------------------------------------------------------------------------
# Centreline extraction
#
# Endpoint discovery is ported from 3D Slicer's SlicerExtension-VMTK
# ExtractCenterlineLogic: extract the vessel tree's topology via
# vtkvmtkPolyDataNetworkExtraction and read off the true endpoints as the
# degree-1 points of that graph. Robust to branches, no manual tuning.
#
# For the actual centerline computation: punching a hole at each chosen
# aorta end and growing the clip radius works (confirmed directly via
# vtkFeatureEdges -- real boundary loops do appear). But vmtkSurfaceCapper +
# vmtkBoundaryReferenceSystems (the script-level VMTK wrappers) were found
# to report zero profiles on this mesh regardless of hole size, so we
# bypass them: find each boundary loop's centroid ourselves with plain VTK
# (vtkFeatureEdges + vtkPolyDataConnectivityFilter), then cap with the
# low-level vtkvmtkCapPolyData (confirmed to execute correctly) and hand
# the loop centroids to vmtkCenterlines as explicit source/target points.
# ---------------------------------------------------------------------------

def preprocess_surface(surface_poly_data, target_number_of_points=5000):
    """
    Decimate to ~target_number_of_points (matches Slicer's ExtractCenterline
    default), then clean, triangulate and add consistent outward normals.
    vtkDecimatePro with PreserveTopologyOn is used (not vtkQuadricDecimation)
    because it guarantees the mesh stays manifold.
    """
    vtk, _ = _require_vtk()

    n_input_points = surface_poly_data.GetNumberOfPoints()
    reduction_factor = (n_input_points - target_number_of_points) / n_input_points
    if reduction_factor > 0.0:
        decimator = vtk.vtkDecimatePro()
        decimator.SetInputData(surface_poly_data)
        decimator.SetTargetReduction(reduction_factor)
        decimator.PreserveTopologyOn()
        decimator.BoundaryVertexDeletionOff()
        decimator.SplittingOff()
        decimator.Update()
        surface_poly_data = decimator.GetOutput()
        print('DEBUG: decimated surface from {} to {} points'.format(
            n_input_points, surface_poly_data.GetNumberOfPoints()), flush=True)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(surface_poly_data)
    cleaner.Update()

    triangulator = vtk.vtkTriangleFilter()
    triangulator.SetInputData(cleaner.GetOutput())
    triangulator.PassLinesOff()
    triangulator.PassVertsOff()
    triangulator.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(triangulator.GetOutput())
    normals.SetAutoOrientNormals(1)
    normals.SetFlipNormals(0)
    normals.SetConsistency(1)
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


def open_surface_at_point(poly_data, hole_position):
    """Cut a single-cell hole in poly_data at the point closest to hole_position (in place)."""
    vtk, _ = _require_vtk()

    locator = vtk.vtkPointLocator()
    locator.SetDataSet(poly_data)
    locator.BuildLocator()
    hole_point_id = locator.FindClosestPoint(hole_position)
    if hole_point_id < 0:
        raise ValueError('open_surface_at_point failed: empty input polydata')

    poly_data.BuildLinks()
    cell_ids = vtk.vtkIdList()
    poly_data.GetPointCells(hole_point_id, cell_ids)
    if cell_ids.GetNumberOfIds() > 0:
        poly_data.DeleteCell(cell_ids.GetId(0))
        poly_data.RemoveDeletedCells()


def extract_network(surface):
    """
    Cut a single seed hole at a bounding-box corner (arbitrary, just gives
    the algorithm somewhere to start) and run VMTK's network extraction to
    recover the vessel tree's topology and per-point MIS radius.
    """
    vtk, _ = _require_vtk()
    _, vtkvmtkMisc = _require_vtkvmtk()

    network_surface = vtk.vtkPolyData()
    network_surface.DeepCopy(surface)

    bounds = network_surface.GetBounds()
    start_position = [bounds[0], bounds[2], bounds[4]]
    open_surface_at_point(network_surface, start_position)

    network_extraction = vtkvmtkMisc.vtkvmtkPolyDataNetworkExtraction()
    network_extraction.SetInputData(network_surface)
    network_extraction.SetAdvancementRatio(1.05)
    network_extraction.SetRadiusArrayName('Radius')
    network_extraction.SetTopologyArrayName('Topology')
    network_extraction.SetMarksArrayName('Marks')
    network_extraction.Update()
    return network_extraction.GetOutput()


def build_network_graph(network_poly_data):
    """
    Clean the raw network extraction output and build cells/links so it's
    ready for topology queries (endpoint detection, graph traversal).
    """
    vtk, _ = _require_vtk()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(network_poly_data)
    cleaner.Update()
    network = cleaner.GetOutput()
    network.BuildCells()
    network.BuildLinks(0)
    return network


def get_network_end_points(network):
    """
    Return every endpoint of the vessel tree as (point_id, coords) pairs --
    points belonging to exactly one cell, i.e. degree-1 points of the
    centerline graph. Covers branch tips too. The point with the largest MIS
    radius (the widest opening, typically the aortic root) is returned
    first. `network` must already be built via build_network_graph.
    """
    vtk, _ = _require_vtk()

    points = network.GetPoints()
    radius_array = network.GetPointData().GetArray('Radius')

    start_point_id = -1
    max_radius = 0.0
    endpoint_ids = vtk.vtkIdList()

    for cell_index in range(network.GetNumberOfCells()):
        cell = network.GetCell(cell_index)
        n_pts = cell.GetNumberOfPoints()
        if n_pts < 2:
            continue
        for point_index in (0, n_pts - 1):
            point_id = cell.GetPointId(point_index)
            point_cells = vtk.vtkIdList()
            network.GetPointCells(point_id, point_cells)
            if point_cells.GetNumberOfIds() == 1:
                endpoint_ids.InsertUniqueId(point_id)
                radius = radius_array.GetValue(point_id)
                if start_point_id < 0 or radius > max_radius:
                    max_radius = radius
                    start_point_id = point_id

    endpoints = []
    n_endpoints = endpoint_ids.GetNumberOfIds()
    if n_endpoints == 0:
        return endpoints
    endpoints.append((start_point_id, points.GetPoint(start_point_id)))
    for i in range(n_endpoints):
        point_id = endpoint_ids.GetId(i)
        if point_id == start_point_id:
            continue
        endpoints.append((point_id, points.GetPoint(point_id)))
    return endpoints


def _network_cell_length(network, cell_index):
    cell = network.GetCell(cell_index)
    n_pts = cell.GetNumberOfPoints()
    total = 0.0
    prev = None
    for i in range(n_pts):
        p = np.array(network.GetPoint(cell.GetPointId(i)))
        if prev is not None:
            total += float(np.linalg.norm(p - prev))
        prev = p
    return total


def select_aorta_end_points(network, endpoints):
    """
    Pick the true two aorta ends by cumulative PATH length along the
    vessel-tree graph, not straight-line distance. endpoints[0] is the
    largest-MIS-radius endpoint (get_network_end_points puts it first) --
    typically the aortic root, the widest opening. The other true aorta end
    is whichever endpoint requires the LONGEST path to reach from the root
    by walking the network's actual branch segments -- not whichever is
    geometrically farthest in 3D. Straight-line "farthest point" is
    unreliable here: the aortic arch's U-turn can put a branch stub or a
    point on the arch itself closer to the root in 3D space than the true
    (possibly short, if the segmentation is cropped) distal end, silently
    shrinking the chord distance used later and inflating Tortuosity
    (length / chord) well beyond the real value.
    """
    root_id, root_coords = endpoints[0]
    if len(endpoints) == 2:
        return root_coords, endpoints[1][1]

    # Undirected weighted graph: nodes are cell endpoints (branch points and
    # tree endpoints), edges are network cells weighted by arc length.
    adjacency = {}
    for cell_index in range(network.GetNumberOfCells()):
        cell = network.GetCell(cell_index)
        n_pts = cell.GetNumberOfPoints()
        if n_pts < 2:
            continue
        a = cell.GetPointId(0)
        b = cell.GetPointId(n_pts - 1)
        length = _network_cell_length(network, cell_index)
        adjacency.setdefault(a, []).append((b, length))
        adjacency.setdefault(b, []).append((a, length))

    # The network is a tree (no loops), so a plain traversal accumulating
    # path length from the root is enough -- no need for real Dijkstra.
    distances = {root_id: 0.0}
    stack = [root_id]
    while stack:
        node = stack.pop()
        for neighbor, length in adjacency.get(node, []):
            new_dist = distances[node] + length
            if neighbor not in distances or new_dist > distances[neighbor]:
                distances[neighbor] = new_dist
                stack.append(neighbor)

    print('DEBUG: path length from root to each endpoint: {}'.format(
        [(pid, distances.get(pid)) for pid, _ in endpoints[1:]]), flush=True)

    target_id, target_coords = max(endpoints[1:], key=lambda e: distances.get(e[0], -1.0))
    return root_coords, target_coords


def clip_surface_at_points(surface, points, radius):
    vtk, _ = _require_vtk()
    result = surface
    for p in points:
        sphere = vtk.vtkSphere()
        sphere.SetCenter(p)
        sphere.SetRadius(radius)
        clipper = vtk.vtkClipPolyData()
        clipper.SetInputData(result)
        clipper.SetClipFunction(sphere)
        clipper.SetInsideOut(False)
        clipper.Update()
        result = clipper.GetOutput()
    return result


def get_boundary_loop_centroids(poly_data):
    """
    Find each closed boundary loop's centroid directly with plain VTK.
    Bypasses vmtkBoundaryReferenceSystems, which reported zero profiles on
    this mesh even when clip_surface_at_points had genuinely created open
    boundaries (confirmed via vtkFeatureEdges directly).
    """
    vtk, _ = _require_vtk()

    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputData(poly_data)
    feature_edges.BoundaryEdgesOn()
    feature_edges.NonManifoldEdgesOff()
    feature_edges.FeatureEdgesOff()
    feature_edges.ManifoldEdgesOff()
    feature_edges.Update()
    boundary_edges = feature_edges.GetOutput()

    if boundary_edges.GetNumberOfPoints() == 0:
        return []

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(boundary_edges)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    labeled = connectivity.GetOutput()
    n_regions = connectivity.GetNumberOfExtractedRegions()

    region_ids = labeled.GetPointData().GetArray('RegionId')
    points = labeled.GetPoints()
    n_points = labeled.GetNumberOfPoints()

    centroids = []
    for region in range(n_regions):
        coords = [points.GetPoint(i) for i in range(n_points)
                  if int(region_ids.GetTuple1(i)) == region]
        if coords:
            centroids.append(tuple(np.mean(coords, axis=0)))
    return centroids


def open_and_cap(vtkvmtkComputationalGeometry, surface, points,
                  radii=(5.0, 10.0, 15.0, 20.0, 25.0, 30.0)):
    """
    Punch holes at `points` and grow the clip radius until exactly two
    clean boundary loops appear (checked directly via vtkFeatureEdges +
    vtkPolyDataConnectivityFilter). Then cap with vtkvmtkCapPolyData.
    """
    for radius in radii:
        opened = clip_surface_at_points(surface, points, radius=radius)
        centroids = get_boundary_loop_centroids(opened)
        print('DEBUG: clip radius={} -> {} boundary loops: {}'.format(
            radius, len(centroids), centroids), flush=True)
        if len(centroids) >= 2:
            capper = vtkvmtkComputationalGeometry.vtkvmtkCapPolyData()
            capper.SetInputData(opened)
            capper.SetDisplacement(0.0)
            capper.SetInPlaneDisplacement(0.0)
            capper.Update()
            return capper.GetOutput(), centroids
    raise RuntimeError(
        'Could not open two clean boundary loops at points {} after trying radii {}.'.format(
            points, radii))


def get_centerline(surface):
    vmtkscripts = _require_vmtk()
    vtkvmtkComputationalGeometry, _ = _require_vtkvmtk()

    network = build_network_graph(extract_network(surface))
    endpoints = get_network_end_points(network)
    if len(endpoints) < 2:
        raise RuntimeError(
            'Network extraction found fewer than two endpoints (n={}).'.format(len(endpoints)))
    print('DEBUG: {} vessel-tree endpoints found (topology-based): {}'.format(
        len(endpoints), [coords for _, coords in endpoints]), flush=True)

    p0, p1 = select_aorta_end_points(network, endpoints)
    print('DEBUG: aorta ends selected (path-length based): p0={} p1={}'.format(p0, p1), flush=True)

    cl_surface, centroids = open_and_cap(vtkvmtkComputationalGeometry, surface, [p0, p1])

    def closest_centroid(target):
        return min(centroids, key=lambda r: np.linalg.norm(np.array(r) - np.array(target)))

    source_point = closest_centroid(p0)
    target_point = closest_centroid(p1)
    print('DEBUG: cap centroids selected: source={} target={}'.format(source_point, target_point), flush=True)

    cl = vmtkscripts.vmtkCenterlines()
    cl.Surface = cl_surface
    cl.SeedSelectorName = 'pointlist'
    cl.SourcePoints = list(source_point)
    cl.TargetPoints = list(target_point)
    cl.AppendEndPoints = 1
    cl.Resampling = 1
    cl.ResamplingStepLength = 1.0
    cl.Execute()

    centerlines = cl.Centerlines
    print('DEBUG: centerline computed: {} points, {} cells'.format(
        centerlines.GetNumberOfPoints(), centerlines.GetNumberOfCells()), flush=True)

    endpoint_coords = [coords for _, coords in endpoints]
    return centerlines, endpoint_coords, source_point, target_point


def get_centerline_geometry(centerline):
    vmtkscripts = _require_vmtk()

    geom = vmtkscripts.vmtkCenterlineGeometry()
    geom.Centerlines = centerline
    geom.LineSmoothing = 1
    geom.Execute()
    return geom.Centerlines


def get_centerline_metrics(centerline):
    _, vtk_np = _require_vtk()

    def point_array(name):
        vtk_arr = centerline.GetPointData().GetArray(name)
        return vtk_np.vtk_to_numpy(vtk_arr) if vtk_arr is not None else None

    def cell_array(name):
        vtk_arr = centerline.GetCellData().GetArray(name)
        return vtk_np.vtk_to_numpy(vtk_arr) if vtk_arr is not None else None

    curvature_all = point_array('Curvature')
    torsion_all = point_array('Torsion')
    radii_all = point_array('MaximumInscribedSphereRadius')
    length = cell_array('Length')
    tortuosity = cell_array('Tortuosity')

    n_cells = centerline.GetNumberOfCells()
    main_cell = int(np.argmax(length)) if n_cells > 1 else 0
    if n_cells > 1:
        print('DEBUG: centerline has {} cells (lengths={}); using cell {} as the main aorta path'.format(
            n_cells, list(length), main_cell), flush=True)

    point_ids = centerline.GetCell(main_cell).GetPointIds()
    main_point_indices = [point_ids.GetId(i) for i in range(point_ids.GetNumberOfIds())]

    curvature = curvature_all[main_point_indices]
    torsion = torsion_all[main_point_indices]
    radii = radii_all[main_point_indices]

    max_radius = float(np.nanmax(radii))

    return {
        'length_mm':             float(length[main_cell]),
        'tortuosity':            float(tortuosity[main_cell]) + 1.0,
        'max_cross_section_mm2': np.pi * max_radius ** 2,
        'diameter_ce_mm':        2 * max_radius,
        'mean_curvature':        float(np.nanmean(curvature)),
        'max_curvature':         float(np.nanmax(curvature)),
        'mean_torsion':          float(np.nanmean(torsion)),
        'max_torsion':           float(np.nanmax(np.abs(torsion))),
    }


# ---------------------------------------------------------------------------
# Visualization export
# ---------------------------------------------------------------------------

def write_polydata(polydata, path):
    vtk, _ = _require_vtk()
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    writer.Write()


def output_path(out_dir, stem, suffix):
    return out_dir / (stem + suffix)


def make_point_markers(points_list, radius=2.0):
    vtk, _ = _require_vtk()
    append = vtk.vtkAppendPolyData()
    for p in points_list:
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(p)
        sphere.SetRadius(radius)
        sphere.Update()
        append.AddInputData(sphere.GetOutput())
    append.Update()
    return append.GetOutput()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

METRIC_FIELDNAMES = [
    'case_id',
    'diameter_mis_mm', 'cross_sectional_area_mm2', 'diameter_ce_mm', 'length_mm',
    'mean_curvature', 'mean_torsion', 'tortuosity', 'surface_area_mm2', 'volume_mm3',
    'error',
]
VTP_FIELDNAMES = ['surface_vtp', 'centerline_vtp', 'endpoints_vtp', 'sourcetarget_vtp']


def process_case(seg_path, out_dir=None, write_vtp=True):
    """Runs the full pipeline for a single segmentation file and returns a metrics dict.
    Raises on failure -- callers doing batch processing should catch and continue."""
    out_dir = out_dir if out_dir is not None else seg_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    name = seg_path.name
    stem = name[:-len('.nii.gz')] if name.endswith('.nii.gz') else seg_path.stem

    image, arr, spacing, qform = read_nifti(seg_path)
    image, arr = keep_largest_component(image, arr)
    surface = get_surface_mesh(image, qform)
    surface = smooth_surface(surface)

    volume = get_volume(arr, spacing)
    surface_area = get_surface_area(surface)
    diameter_mis = get_diameter_mis(arr, spacing)

    preprocessed = preprocess_surface(surface)
    centerline, endpoints, p0, p1 = get_centerline(preprocessed)
    centerline = get_centerline_geometry(centerline)
    cl_metrics = get_centerline_metrics(centerline)

    results = {
        'case_id':                 stem,
        'diameter_mis_mm':         diameter_mis,
        'cross_sectional_area_mm2': cl_metrics['max_cross_section_mm2'],
        'diameter_ce_mm':          cl_metrics['diameter_ce_mm'],
        'length_mm':               cl_metrics['length_mm'],
        'mean_curvature':          cl_metrics['mean_curvature'],
        'mean_torsion':            cl_metrics['mean_torsion'],
        'tortuosity':              cl_metrics['tortuosity'],
        'surface_area_mm2':        surface_area,
        'volume_mm3':              volume,
        'error':                   '',
    }

    if write_vtp:
        surface_path = output_path(out_dir, stem, '_surface.vtp')
        centerline_path = output_path(out_dir, stem, '_centerline.vtp')
        endpoints_path = output_path(out_dir, stem, '_endpoints.vtp')
        sourcetarget_path = output_path(out_dir, stem, '_sourcetarget.vtp')

        write_polydata(surface, surface_path)
        write_polydata(centerline, centerline_path)
        write_polydata(make_point_markers(endpoints), endpoints_path)
        write_polydata(make_point_markers([p0, p1], radius=3.0), sourcetarget_path)

        results['surface_vtp'] = str(surface_path)
        results['centerline_vtp'] = str(centerline_path)
        results['endpoints_vtp'] = str(endpoints_path)
        results['sourcetarget_vtp'] = str(sourcetarget_path)

    return results


def print_results(results):
    print('DiameterMIS:        {:.4f}'.format(results['diameter_mis_mm']), flush=True)
    print('CrossSectionalArea: {:.4f}'.format(results['cross_sectional_area_mm2']), flush=True)
    print('DiameterCE:         {:.4f}'.format(results['diameter_ce_mm']), flush=True)
    print('Length:             {:.4f}'.format(results['length_mm']), flush=True)
    print('Curvature (mean):   {:.6f}'.format(results['mean_curvature']), flush=True)
    print('Torsion (mean):     {:.6f}'.format(results['mean_torsion']), flush=True)
    print('Tortuosity:         {:.5f}'.format(results['tortuosity']), flush=True)
    print('SurfaceAreamm2:     {:.4f}'.format(results['surface_area_mm2']), flush=True)
    print('Volumemm3:          {:.4f}'.format(results['volume_mm3']), flush=True)


def run_single_case(seg_path, out_json, out_dir):
    results = process_case(seg_path, out_dir, write_vtp=True)
    print_results(results)
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('Wrote metrics to {}'.format(out_json), flush=True)


def run_case_subprocess(seg_path, out_dir, write_vtp, timeout_seconds):
    """
    Runs process_case() for one segmentation in its own subprocess (via
    _process_case_worker.py), so a native VMTK hang or crash on this specific case
    (some cases have unusually complex topology that vtkvmtkPolyDataNetworkExtraction
    can choke on) only kills this subprocess, caught here as a timeout or non-zero
    exit code, instead of freezing or taking down the whole batch run.
    """
    tmp_json = seg_path.with_name(seg_path.stem.split('.')[0] + '_tmp_result.json')
    cmd = [
        sys.executable, str(Path(__file__).parent / '_process_case_worker.py'),
        '--seg', str(seg_path),
        '--out-json', str(tmp_json),
    ]
    if out_dir is not None:
        cmd += ['--out-dir', str(out_dir)]
    if write_vtp:
        cmd += ['--write-vtp']

    try:
        proc = subprocess.run(
            cmd, timeout=timeout_seconds,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError('Timed out after {}s (likely a native VMTK hang)'.format(timeout_seconds))

    if proc.stdout:
        print(proc.stdout, end='', flush=True)

    if proc.returncode != 0:
        stderr_lines = [line for line in proc.stderr.strip().splitlines() if line.strip()]
        tail = stderr_lines[-1] if stderr_lines else 'unknown error'
        raise RuntimeError('Subprocess exited with code {} (likely a native crash): {}'.format(
            proc.returncode, tail))

    try:
        with open(tmp_json) as f:
            results = json.load(f)
    finally:
        if tmp_json.exists():
            tmp_json.unlink()
    return results


def run_batch(seg_dir, out_csv, out_dir, write_vtp=False, timeout_seconds=600):
    """
    Processes every .nii.gz in seg_dir and writes one row per case to out_csv.
    Each case runs in its own subprocess (see run_case_subprocess) with a timeout, so
    a hang or native crash on one case can't freeze or kill the whole batch. Failures
    (timeouts, crashes, or ordinary exceptions) are caught and recorded with an error
    message rather than stopping the run -- across dozens of cases some are expected
    to fail on edge-case anatomy, and one bad case shouldn't cost the rest.
    The CSV is rewritten after every case (not just at the end) so an interrupted
    batch doesn't lose already-completed results; rerunning the same command resumes
    by skipping case_ids already present in out_csv without an error.
    """
    seg_files = sorted(seg_dir.glob('*.nii.gz'))
    if not seg_files:
        sys.exit('No .nii.gz files found in {}'.format(seg_dir))

    fieldnames = METRIC_FIELDNAMES + (VTP_FIELDNAMES if write_vtp else [])

    all_results = []
    completed_ids = set()
    if out_csv.exists():
        with open(out_csv, newline='') as f:
            for row in csv.DictReader(f):
                all_results.append(row)
                if not row.get('error'):
                    completed_ids.add(row['case_id'])
        print('DEBUG: resuming, {} cases already completed in {}'.format(
            len(completed_ids), out_csv), flush=True)

    def write_csv():
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_results)

    for i, seg_path in enumerate(seg_files, 1):
        name = seg_path.name
        stem = name[:-len('.nii.gz')] if name.endswith('.nii.gz') else seg_path.stem
        if stem in completed_ids:
            print('=== [{}/{}] {} (already done, skipping) ==='.format(i, len(seg_files), name), flush=True)
            continue

        print('=== [{}/{}] {} ==='.format(i, len(seg_files), name), flush=True)
        try:
            results = run_case_subprocess(seg_path, out_dir, write_vtp, timeout_seconds)
            print_results(results)
        except Exception as e:
            print('ERROR processing {}: {}'.format(name, e), flush=True)
            results = {'case_id': stem, 'error': str(e)}

        all_results = [r for r in all_results if r.get('case_id') != stem]
        all_results.append(results)
        write_csv()

    n_ok = sum(1 for r in all_results if not r.get('error'))
    print('\n{}/{} cases processed successfully. Results written to {}'.format(
        n_ok, len(all_results), out_csv), flush=True)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--seg', type=Path, help='Single segmentation .nii.gz file')
    group.add_argument('--seg-dir', type=Path, help='Folder of segmentation .nii.gz files to batch process')
    parser.add_argument('--out-json', type=Path, help='Where to write the computed metrics (--seg mode)')
    parser.add_argument('--out-csv', type=Path, help='Where to write the metrics table (--seg-dir mode)')
    parser.add_argument('--out-dir', type=Path, default=None,
                         help='Where to write surface/centerline/endpoint .vtp files (default: alongside input)')
    parser.add_argument('--write-vtp', action='store_true',
                         help='Also write per-case .vtp visualization files in --seg-dir mode (off by default; '
                              'always on in --seg mode)')
    parser.add_argument('--timeout', type=int, default=180,
                         help='Per-case timeout in seconds for --seg-dir mode (default: 180). A case that hangs '
                              '(native VMTK hang) or exceeds this is killed and recorded as a failure so the '
                              'batch continues.')
    args = parser.parse_args()

    if args.seg:
        if not args.out_json:
            parser.error('--out-json is required with --seg')
        run_single_case(args.seg, args.out_json, args.out_dir)
    else:
        if not args.out_csv:
            parser.error('--out-csv is required with --seg-dir')
        run_batch(args.seg_dir, args.out_csv, args.out_dir, write_vtp=args.write_vtp, timeout_seconds=args.timeout)


if __name__ == '__main__':
    main()
