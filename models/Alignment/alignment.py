import trimesh
import open3d as o3d

import numpy as np
from open3d import io, geometry

def apply_transformation(mesh, translation=None, scale=None, rotation=None):
    """
    Applies translation, scale, and rotation to a 3D mesh.

    Parameters:
    - mesh: open3d.geometry.TriangleMesh, the 3D mesh.
    - translation: 3-element list or numpy array, translation along x, y, z axes.
    - scale: 3-element list or numpy array, scaling along x, y, z axes.
    - rotation: 4-element list or numpy array, rotation angle and it's axis.

    Returns:
    - open3d.geometry.TriangleMesh, the transformed mesh.
    """
    if translation is not None:
        translation_matrix = np.identity(4)
        translation_matrix[:3, 3] = translation
        mesh.transform(translation_matrix)

    if scale is not None:
        scale_matrix = np.identity(4)
        np.fill_diagonal(scale_matrix[:3, :3], scale)
        mesh.transform(scale_matrix)

    if rotation is not None:
        quaternion = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation[0], rotation[1:])
        mesh.rotate(quaternion, center=(0, 0, 0))

    return mesh


def merge_meshes(mesh1, mesh2):
    return mesh1 + mesh2

def find_closest_vertex(normal_vector, mesh):
    '''
    - Find the vertex in the given mesh with the closest distance to the plane
    - The plane is formed by the given normal vector and a point on the vector. 
    - Ensure that the point on the vector is outside of a unit cube.
    
    normal_vector : np.array([A, B, C])
    plane_eqation : Ax + By + Cz + D = 0
    plane_vector  : np.array([A, B, C, D])
    D = - np.linalg.norm(normal_vector), this condition guarantees that plane is outside of a unit cube. 
    '''
    
    plane_vector = np.concatenate((normal_vector, [-np.linalg.norm(normal_vector)]), axis=-1)
    
    vertices = np.asarray(mesh.vertices)
    vertices_ = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
    distance_vector = 1/np.linalg.norm(normal_vector) * vertices_ @ plane_vector
    return vertices[np.argmin(distance_vector)]



    
    
    
# Set path of *.ply file
input = './examples/aircraft.ply'
output = './examples/merged_aircraft.ply'

# Operation
mesh1 = o3d.io.read_triangle_mesh(input)
# mesh2 = apply_transformation(mesh1, translation=[2,2,2], scale=[1,1,1])
# mesh = merge_meshes(mesh1, mesh2)

vertices = np.asarray(mesh1.vertices)
print(vertices.shape)
print(max(vertices[:, 0]), max(vertices[:, 1]), max(vertices[:, 2]))
print(min(vertices[:, 0]), min(vertices[:, 1]), min(vertices[:, 2]))

# Save Mesh
#o3d.io.write_triangle_mesh(output, mesh)

