import os
import uuid
from typing import Dict, Any, Optional, Tuple
import numpy as np
import open3d as o3d
from loguru import logger

def process_point_cloud(ply_path: str, is_large_object: bool = False, output_dir: Optional[str] = None) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Process a .ply point cloud file using Open3D.
    - Removes statistical outliers
    - Optionally removes the support plane (for small objects)
    - Calculates gravity-aligned Oriented Bounding Box
    - Reconstructs a Poisson surface mesh

    Args:
        ply_path: Local path to the input .ply file.
        is_large_object: If True, skip support plane filtering.
        output_dir: Directory to save the reconstructed mesh. If None, uses the same directory as ply_path.

    Returns:
        A tuple of (measurements_dict, mesh_ply_path)
        measurements_dict contains length_mm, width_mm, height_mm, point_count.
        mesh_ply_path is the path to the reconstructed mesh file, or None if it failed.
    """
    logger.info(f"Loading point cloud for processing: {ply_path}")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Point cloud file not found: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    point_count = len(pcd.points)
    if point_count == 0:
        raise ValueError("Point cloud is empty or invalid format.")

    # 1. Denoise via Statistical Outlier Removal
    std_ratio = 1.5 if is_large_object else 2.0
    logger.info(f"Filtering outliers (std_ratio={std_ratio})...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=std_ratio)
    pcd = pcd.select_by_index(ind)
    filtered_count = len(pcd.points)

    if filtered_count < 10:
        raise ValueError("Too few points left after outlier filtering.")

    # 2. Extract points and detect support plane (for small objects)
    points = np.asarray(pcd.points)

    if not is_large_object:
        logger.info("Detecting support plane using RANSAC...")
        # Fit support plane using RANSAC (e.g. table floor surface)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.015,
            ransac_n=3,
            num_iterations=200
        )

        # Keep only the points above the support plane (distance > 1.5cm)
        if len(inliers) > 0:
            normal = plane_model[:3]
            # Direct normal upward
            if normal[1] < 0:
                normal = -normal
            d = plane_model[3]

            # Filter points significantly above the plane
            distances = np.dot(points, normal) + d
            object_mask = distances > 0.015
            filtered_points = points[object_mask]

            if len(filtered_points) >= 10:
                # Update point cloud points
                pcd.points = o3d.utility.Vector3dVector(filtered_points)
                points = filtered_points
                logger.info(f"Support plane filtered. Points left: {len(points)}")

    # 3. Calculate Bounding Box
    logger.info("Calculating gravity-aligned Oriented Bounding Box...")
    obb = pcd.get_oriented_bounding_box()
    extents = obb.extent
    R = obb.R

    # Gravity direction is Y-axis (0, 1, 0) in ARKit coordinate frame
    gravity = np.array([0.0, 1.0, 0.0])

    # Find which axis is most aligned with gravity
    alignments = [abs(np.dot(R[:, i], gravity)) for i in range(3)]
    height_axis_idx = np.argmax(alignments)

    # Horizontal dimensions
    horizontal_indices = [i for i in range(3) if i != height_axis_idx]
    axis_a, axis_b = horizontal_indices[0], horizontal_indices[1]

    # Assign length to the larger horizontal extent, width to the smaller
    if extents[axis_a] >= extents[axis_b]:
        length_idx = axis_a
        width_idx = axis_b
    else:
        length_idx = axis_b
        width_idx = axis_a

    # Dimensions in meters, convert to millimeters
    length_mm = float(extents[length_idx]) * 1000.0
    width_mm = float(extents[width_idx]) * 1000.0
    height_mm = float(extents[height_axis_idx]) * 1000.0

    measurements = {
        "length_mm": round(length_mm, 2),
        "width_mm": round(width_mm, 2),
        "height_mm": round(height_mm, 2),
        "point_count": len(pcd.points)
    }

    # 4. Reconstruct 3D Surface Mesh
    mesh_path = None
    try:
        logger.info("Estimating normals and generating mesh...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(100)

        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

        # Remove low-density outlier vertices
        vertices_to_remove = densities < np.percentile(densities, 5)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        out_dir = output_dir or os.path.dirname(ply_path)
        mesh_filename = f"{uuid.uuid4().hex}_mesh.ply"
        mesh_path = os.path.join(out_dir, mesh_filename)

        o3d.io.write_triangle_mesh(mesh_path, mesh)
        logger.info(f"Mesh reconstructed and saved to {mesh_path}")
    except Exception as e:
        logger.error(f"Mesh generation failed: {e}")
        mesh_path = None

    return measurements, mesh_path
