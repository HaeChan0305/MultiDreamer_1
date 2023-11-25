import numpy as np
import open3d as o3d
from open3d import io, geometry
from PIL import Image

from geometry import depth_to_points

DEBUG = True
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

def get_depth_info(mask, depth):
    '''
    Find the positions and the depth values which are max and min in the given mask area.
    
    - mask  : np.array, (H, W) : 0 or 255
    - depth : np.array, (H, W) : 0 ~ 255
    
    '''
    max_mask = np.where(mask==255, depth, -1)
    max_pos = np.unravel_index(np.argmax(max_mask), depth.shape)
    max_depth = max_mask[max_pos[0], max_pos[1]]
    
    min_mask = np.where(mask==255, depth, 256)
    min_pos = np.unravel_index(np.argmin(min_mask), depth.shape)
    min_depth = min_mask[min_pos[0], min_pos[1]]
    
    if DEBUG:
        print(">>> max_pos, max_depth, min_pos, min_depth", max_pos, max_depth, min_pos, min_depth)
        
        cnt_max_pos = np.sum(np.where(max_mask==max_depth, 1, 0))
        print(">>> The number of points which have the maximum value of depth.", cnt_max_pos)
        
        cnt_min_pos = np.sum(np.where(min_mask==min_depth, 1, 0))
        print(">>> The number of points which have the mininum value of depth.", cnt_min_pos)
        
        depth[max_pos[0], max_pos[1]] = 0
        depth[min_pos[0], min_pos[1]] = 255

        pred = Image.fromarray(depth)
        pred.save("../../data/output/depth_minmax_debugging.png")
        
    return max_pos, max_depth, min_pos, min_depth
    

def get_depth_gap(mask1, mask2, depth):
    '''
    1번 함수
    
    Argument :
        - mask1 : np.array, 0 or 255, object1 에 대한 mask
        - mask2 : np.array, 0 or 255, object2 에 대한 mask, 
        - depth : mp.array, 0 ~ 255,
    
    return :
        - a, object1 내에서 가장 큰 depth 차이 값
        - b, object2 내에서 가장 큰 depth 차이 값
        - c, object1의 가장 밝은 부분(min depth)와 object2의 가장 밝은 부분 차이 값
    '''
    _, max_value1, _, min_value1 = get_depth_info(mask1, depth)
    _, max_value2, _, min_value2 = get_depth_info(mask2, depth)
    
    return max_value1 - min_value1, max_value2 - min_value2, np.abs(min_value1 - min_value2)
    
    

def get_mesh_distance_gap(vector, mesh):
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
        

mask0 = np.array(Image.open("../../data/output/ref_washing/mask0.jpg"))
mask1 = np.array(Image.open("../../data/output/ref_washing/mask1.jpg"))
colorized_depth = np.array(Image.open("../../data/output/ref_washing/depth.png")) # (H, W, 4)
depth = colorized_depth[:, :, 0] # (H, W)
    
print(get_depth_gap(mask0, mask1, depth))
    

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
