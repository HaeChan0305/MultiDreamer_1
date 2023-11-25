import numpy as np
import open3d as o3d
from open3d import io, geometry
from PIL import Image

from geometry import depth_to_points


# class Alignment():
#     def __init__(self, input_path, mask_path, depth_path, mesh1_path, mesh2_path):
#         self.input = np.array(Image.open(input_path))
#         self.mask = np.array(Image.open(mask_path))
#         self.depth = np.array(Image.open(depth_path))
#         self.mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
#         self.mesh2 = o3d.io.read_triangle_mesh(mesh2_path)
        

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

def find_key_depth_points():
    mask0 = np.array(Image.open("../../data/output/ref_washing/mask0.jpg"))
    mask1 = np.array(Image.open("../../data/output/ref_washing/mask1.jpg"))
    depth = np.array(Image.open("../../data/output/ref_washing/depth.png"))
    
    print(depth)
    print(mask0)
    

def find_gap(vector, mesh):
    '''
    2번 함수
    
    - vector : np.array([A, B, C]), 원점에서 카메라 방향의 벡터.
    - normal_vector : np.array([a, b, c]), normalized vector.
    - plane : vector를 법선벡터로 가지고, vector 위를 지나는 임의의 점을 지나는 평면, 단 평면은 unit cube 밖에 있어야 함. 
    = plane_vector : np.array([a, b, c, d]), when plane is defied as aX + bY + cZ + d = 0. (l2 norm of (a, b, c) = 1 and d=-1)
    - P1 : mesh vertex 중 plane과 가장 가까운 점.
    - P2 : mesh vertex 중, vector 방향으로 orthogonal view로 봤을 때, plane과 가장 먼 점.
    - gap : distance(P1, P2)와 normal_vector의 내적
    '''
    
    vertices = np.asarray(mesh.vertices)
    normal_vector = vector / np.linalg.norm(vector)
    projections = vertices @ normal_vector
    
    vertices = np.concatenate((vertices, np.ones((vertices.shape[0], 1))), axis=1)
    plane_vector = np.concatenate((normal_vector, [-1]), axis=-1)
    distance_vector = np.abs(vertices @ plane_vector)
    
    P1 = vertices[np.argmin(distance_vector)][:3]
    
    visible_vertices = vertices[projections >= 0]
    P2 = visible_vertices[np.argmax(distance_vector)][:3]
    
    return np.abs((P1 - P2) * normal_vector)
        

    
find_key_depth_points()
    

# Set path of *.ply file
# input = './examples/aircraft.ply'
# output = './examples/merged_aircraft.ply'

# # Operation
# mesh1 = o3d.io.read_triangle_mesh(input)
# # mesh2 = apply_transformation(mesh1, translation=[2,2,2], scale=[1,1,1])
# # mesh = merge_meshes(mesh1, mesh2)

# vertices = np.asarray(mesh1.vertices)
# print(vertices.shape)
# print(max(vertices[:, 0]), max(vertices[:, 1]), max(vertices[:, 2]))
# print(min(vertices[:, 0]), min(vertices[:, 1]), min(vertices[:, 2]))

# Save Mesh
#o3d.io.write_triangle_mesh(output, mesh)
