"""
Standalone, isolated reproduction of AortaSegmentationLogic.computeMetrics()'s VMTK
calls, meant to be run via PythonSlicer.exe as its own process:

    "C:\\Users\\esranko1\\AppData\\Local\\slicer.org\\3D Slicer 5.10.0\\bin\\PythonSlicer.exe" ^
        test_vmtk_centerline.py --seg <segmentation.nii.gz>

Why this exists: calling ExtractCenterlineLogic/vtkvmtk directly inside the interactive
Slicer application means any native (C++) crash takes down the whole app with zero
diagnostics. Running the identical calls here isolates a crash to just this process and
gives clean stdout/stderr to debug from, without losing the interactive session.

This does NOT go through a segmentation MRML node (avoids needing GUI machinery) -- it
builds the surface via plain marching cubes on the NIfTI file directly, the same way the
original standalone script.py did, then hands that surface to the real
ExtractCenterlineLogic methods (preprocess/extractNetwork/extractCenterline), which is
the same code path AortaSegmentationLogic.computeMetrics() calls in the module.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import vtk
import vtk.util.numpy_support as vtk_np
from scipy.ndimage import label

import slicer  # noqa: F401 -- PythonSlicer provides this; confirms we're running under it
import ExtractCenterline
import vtkvmtkComputationalGeometryPython as vtkvmtkComputationalGeometry


def read_nifti(nifti_path):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(nifti_path))
    reader.Update()
    image = reader.GetOutput()
    qform = reader.GetQFormMatrix()

    nx, ny, nz = image.GetDimensions()
    scalars = image.GetPointData().GetScalars()
    arr = vtk_np.vtk_to_numpy(scalars).reshape(nz, ny, nx).astype(np.float32)
    return image, arr, image.GetSpacing(), qform


def keep_largest_component(image, arr):
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


def keep_largest_surface_component(poly_data):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly_data)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(connectivity.GetOutput())
    cleaner.Update()
    return cleaner.GetOutput()


def get_network_end_points(network):
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


def network_cell_length(network, cell_index):
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
    import heapq
    root_id, root_coords = endpoints[0]
    if len(endpoints) == 2:
        return root_coords, endpoints[1][1]

    adjacency = {}
    for cell_index in range(network.GetNumberOfCells()):
        cell = network.GetCell(cell_index)
        n_pts = cell.GetNumberOfPoints()
        if n_pts < 2:
            continue
        a = cell.GetPointId(0)
        b = cell.GetPointId(n_pts - 1)
        length = network_cell_length(network, cell_index)
        adjacency.setdefault(a, []).append((b, length))
        adjacency.setdefault(b, []).append((a, length))

    distances = {root_id: 0.0}
    visited = set()
    heap = [(0.0, root_id)]
    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        for neighbor, length in adjacency.get(node, []):
            new_dist = dist + length
            if neighbor not in distances or new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(heap, (new_dist, neighbor))

    target_id, target_coords = max(endpoints[1:], key=lambda e: distances.get(e[0], -1.0))
    return root_coords, target_coords


def run(seg_path):
    """Callable directly from Slicer's Python console, e.g.:
    import sys; sys.path.insert(0, r'<this Scripts folder>')
    import test_vmtk_centerline
    test_vmtk_centerline.run(r'C:\\path\\to\\segmentation.nii.gz')
    """
    seg_path = Path(seg_path)

    print('DEBUG: reading NIfTI...', flush=True)
    image, arr, spacing, qform = read_nifti(seg_path)
    image, arr = keep_largest_component(image, arr)

    print('DEBUG: marching cubes...', flush=True)
    surface = get_surface_mesh(image, qform)
    surface = keep_largest_surface_component(surface)
    print('DEBUG: surface: {} points, {} cells'.format(
        surface.GetNumberOfPoints(), surface.GetNumberOfCells()), flush=True)

    centerline_logic = ExtractCenterline.ExtractCenterlineLogic()

    print('DEBUG: preprocess (decimation CLI)...', flush=True)
    preprocessed = centerline_logic.preprocess(
        surface, targetNumberOfPoints=5000, decimationAggressiveness=4.0, subdivide=False
    )
    print('DEBUG: preprocessed: {} points, {} cells'.format(
        preprocessed.GetNumberOfPoints(), preprocessed.GetNumberOfCells()), flush=True)

    print('DEBUG: extractNetwork...', flush=True)
    network_poly_data = centerline_logic.extractNetwork(preprocessed, endPointsMarkupsNode=None)
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(network_poly_data)
    cleaner.Update()
    network = cleaner.GetOutput()
    network.BuildCells()
    network.BuildLinks(0)

    endpoints = get_network_end_points(network)
    print('DEBUG: {} vessel-tree endpoints found: {}'.format(
        len(endpoints), [c for _, c in endpoints]), flush=True)
    if len(endpoints) < 2:
        sys.exit('Could not find two vessel endpoints (found {}).'.format(len(endpoints)))

    p0, p1 = select_aorta_end_points(network, endpoints)
    print('DEBUG: aorta ends selected: p0={} p1={}'.format(p0, p1), flush=True)

    print('DEBUG: creating markups node...', flush=True)
    end_points_node = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
    try:
        end_points_node.AddControlPoint(vtk.vtkVector3d(p0[0], p0[1], p0[2]))
        end_points_node.AddControlPoint(vtk.vtkVector3d(p1[0], p1[1], p1[2]))
        end_points_node.SetNthControlPointSelected(0, False)
        end_points_node.SetNthControlPointSelected(1, True)

        print('DEBUG: extractCenterline (this is the step most likely to crash)...', flush=True)
        centerline_poly_data, _voronoi = centerline_logic.extractCenterline(
            preprocessed, end_points_node, curveSamplingDistance=1.0
        )
    finally:
        slicer.mrmlScene.RemoveNode(end_points_node)

    print('DEBUG: centerline computed: {} points, {} cells'.format(
        centerline_poly_data.GetNumberOfPoints(), centerline_poly_data.GetNumberOfCells()), flush=True)

    print('DEBUG: computing centerline geometry...', flush=True)
    geometry = vtkvmtkComputationalGeometry.vtkvmtkCenterlineGeometry()
    geometry.SetInputData(centerline_poly_data)
    geometry.SetLengthArrayName('Length')
    geometry.SetCurvatureArrayName('Curvature')
    geometry.SetTorsionArrayName('Torsion')
    geometry.SetTortuosityArrayName('Tortuosity')
    geometry.SetFrenetTangentArrayName('FrenetTangent')
    geometry.SetFrenetNormalArrayName('FrenetNormal')
    geometry.SetFrenetBinormalArrayName('FrenetBinormal')
    geometry.Update()
    result = geometry.GetOutput()

    length_array = vtk_np.vtk_to_numpy(result.GetCellData().GetArray('Length'))
    tortuosity_array = vtk_np.vtk_to_numpy(result.GetCellData().GetArray('Tortuosity'))
    radii = vtk_np.vtk_to_numpy(result.GetPointData().GetArray('Radius'))
    curvature = vtk_np.vtk_to_numpy(result.GetPointData().GetArray('Curvature'))
    torsion = vtk_np.vtk_to_numpy(result.GetPointData().GetArray('Torsion'))

    main_cell = int(np.argmax(length_array)) if result.GetNumberOfCells() > 1 else 0
    max_radius = float(np.nanmax(radii))

    print('SUCCESS', flush=True)
    print('Length:             {:.4f}'.format(float(length_array[main_cell])))
    print('Tortuosity:         {:.5f}'.format(float(tortuosity_array[main_cell]) + 1.0))
    print('DiameterCE:         {:.4f}'.format(2 * max_radius))
    print('CrossSectionalArea: {:.4f}'.format(np.pi * max_radius ** 2))
    print('Curvature (mean):   {:.6f}'.format(float(np.nanmean(curvature))))
    print('Torsion (mean):     {:.6f}'.format(float(np.nanmean(torsion))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seg', type=Path, required=True)
    args = parser.parse_args()
    run(args.seg)


if __name__ == '__main__':
    main()
