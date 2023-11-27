import numpy as np
import torch
import open3d as o3d
from PIL import Image
import spicy


#############
# depth_to_points 분석을 통해, object별 (z_max - z_min) 값을 찾아야 됨.
#   - depth_to_mesh의 view vector = [0, 0, -1]
#   - 배경이 min, max 값에 영향을 미칠 수 있으므로, mask를 씌워야됨.
#############
# def get_scene_gap(depth_to_points, mask):
#     z_points = depth_to_points[:, :, 2]
    
#     max_mask = np.where(mask==255, z_points, -np.inf)
#     z_max = np.max(max_mask)
    
#     min_mask = np.where(mask==255, z_points, np.inf)
#     z_min = np.min(min_mask)
    
#     return z_max - z_min

# z_gap0 = get_scene_gap(depth_to_points, mask0)
# z_gap1 = get_scene_gap(depth_to_points, mask1)
# print(z_gap0)
# print(z_gap1)

#############
# Rotation
#   - SyncDreamer output Mesh를 적절히 rotate 시켜서 Depth_to_Mesh와 view direction이 맞도록 설정.
#############

def rotate(mesh, axis, degree):
    '''
    Arguments:
        - axis : List
        - degree : Int
    
    Return:
        - Mesh
    '''
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)
    angle = np.radians(degree)
    
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(np.multiply(axis, angle))
    return mesh.rotate(rotation_matrix)

def rotation_alignment(mesh):
    '''
    x -90도, y 90도, x -30도
    '''
    mesh = rotate(mesh, [1, 0, 0], -90)
    mesh = rotate(mesh, [0, 1, 0], 90)
    mesh = rotate(mesh, [1, 0, 0], -30)
    return mesh

#############
# Masking
#   - depth_to_points를 object 별 mask로 잘라내야됨.
#############
def masking(mask, depth_to_points):
    '''
    Arguments :
        - mask : np.array(H, W)
        - depth_to_points : np.array(H, W, 3)
        
    Return :
        masked_points : np.array(M, 3)
    '''
    
    masked_points = depth_to_points[np.where(mask==255)]
    return masked_points
    

#############
# Sampling Strategy
#   - vertex norm이 view direction과 90도 이상 차이나는 vertex 제외
#
#############
def vertex_normal_sampling(points, M):
    '''
    Arguments :
        - points : np.array(N, 3)
        - M : depth_to_points 의 points 수
        
    Return :
        - sampled_points : np.array(N', 3)
    '''
    assert points.shape[0] > M
    
    view_direction = np.array([0, 0, -1]) #depth_to_points의 default view direction
    projection = points @ view_direction
    sampled_points = points[np.where(projection > 0)]
    
    print(">>> vertex_normal_sampling : ", points.shape[0], "->", sampled_points.shape[0])
    assert sampled_points.shape[0] > M
    return sampled_points
    
    
def uniform_sampling(points, M):
    assert points.shape[0] > M
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    sampling_rate = points.shape[0] // M
    sampled_point_cloud = o3d.geometry.uniform_down_sample(point_cloud, every_k_points=sampling_rate)
    sampled_points = np.array(sampled_point_cloud.points)
    
    print(">>> uniform_sampling : ", points.shape[0], "->", sampled_points.shape[0])
    return sampled_points
    
def rest_sampling(points, M):
    assert points.shape[0] > M
    

def sampling(mesh, masked_depth_to_points):
    mesh_points = np.array(mesh.vertices)
    M, _ = masked_depth_to_points.shape
    
    sampled_points = vertex_normal_sampling(mesh_points, M)
    sampled_points = uniform_sampling(sampled_points, M)
    
    if sampled_points > M:
        pass
    elif sampled_points < M:
        pass
    
    assert sampled_points.shape[0] == M
    return sampled_points
    

#############
# Optimizing
#   y와 x @ d에 대한 chamfer distance가 최소가 되는 d를 구하는 문제
#   - y : (M', 3), object별 masked 된 depth_to_points
#   - x : (M', 3), X의 points만큼 sampled된 vertices
#   - d : (4, 4), X_를 scaling, translation 시키는 transformation matrix 
#############
def chamfer_distance(X, Y):
    dist_x_to_y = np.linalg.norm(X[:, None, :] - Y, axis=-1).min(axis=-1)
    dist_y_to_x = np.linalg.norm(Y[:, None, :] - X, axis=-1).min(axis=-1)
    
    total_distance = np.sum(dist_x_to_y) + np.sum(dist_y_to_x)
    return total_distance

def params_to_matrix(params):
    '''
    params : np.array([sx, sy, sz, tx, ty, tz])
    '''
    sx, sy, sz, tx, ty, tz = params
    return np.array([[sx,  0,  0, 0],
                     [0,  sy,  0, 0],
                     [0,   0, sz, 0],
                     [tx, ty, tz, 1]])

def objective_function(params, x, y):
    '''
    params : np.array([sx, sy, sz, tx, ty, tz]), six elements for transformation
    x : sampled vertices in the mesh
    y : masked points in the depth_to_points
    '''
    matrix = params_to_matrix(params)
    
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    y_ = x @ matrix
    y_ = y_[:, :-1]
    return chamfer_distance(y, y_)


def optimizeing(x, y):
    '''
    x : sampled vertices in the mesh
    y : masked points in the depth_to_points
    params : np.array([sx, sy, sz, tx, ty, tz]), six elements for transformation
    '''
    initial_params = np.random.randn(6)
    result = spicy.optimize.minimize(objective_function, initial_params, args=(x, y), method='BFGS')
    
    optimized_params = result.x
    return optimized_params
    
    
#############
# Applying
#   - optimized_matrix를 mesh vertices에 적용시킴.
#############
def applying_transformation(mesh, optimized_params):
    '''
    무조건 scaling 먼저!
    '''
    
    # Scaling
    scale = optimized_params[:3]
    scale_matrix = np.identity(4)
    np.fill_diagonal(scale_matrix[:3, :3], scale)
    mesh = mesh.transform(scale_matrix)
    
    # Translation
    translation = optimized_params[3:]
    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = translation
    mesh = mesh.transform(translation_matrix)
    
    return mesh
    

def main():
    # input으로 받을 수 있게 변경할 것.
    input_path = "../../data/input/toycar_and_chair.png"
    mask0_path = "../../data/output/1/mask0.jpg"
    mask1_path = "../../data/output/1/mask1.jpg"
    mesh0_path = "../../data/output/1/mesh0.ply"
    mesh1_path = "../../data/output/1/mesh1.ply"
    depth_to_points_path = "../../data/output/1/depth.npy"

    # Load data
    input = np.array(Image.open(input_path))
    mask0 = np.array(Image.open(mask0_path))
    mask1 = np.array(Image.open(mask1_path))
    mesh0 = o3d.io.read_triangle_mesh(mesh0_path)
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    depth_to_points = np.load(depth_to_points_path)

    # Rotation
    rotated_mesh0 = rotation_alignment(mesh0)
    rotated_mesh1 = rotation_alignment(mesh1)

    o3d.io.write_triangle_mesh('../../data/output/1/rotated_mesh0.ply', rotated_mesh0)
    o3d.io.write_triangle_mesh('../../data/output/1/rotated_mesh1.ply', rotated_mesh1)
    
    # Masking
    masked_points0 = masking(mask0, depth_to_points)
    masked_points1 = masking(mask1, depth_to_points)
    
    # Sampling
    sampled_points0 = sampling(mesh0, masked_points0)
    sampled_points1 = sampling(mesh1, masked_points1)
    
    # Optimizing
    optimized_params0 = optimizeing(sampled_points0, masked_points0)
    optimized_params1 = optimizeing(sampled_points1, masked_points1)
    
    # Applying
    result_mesh0 = applying_transformation(rotated_mesh0, optimized_params0)
    result_mesh1 = applying_transformation(rotated_mesh1, optimized_params1)
    
    # Exporting
    output0 = "../../data/output/1/result_mesh0.ply"
    output1 = "../../data/output/1/result_mesh1.ply"
    o3d.io.write_triangle_mesh(output0, result_mesh0)
    o3d.io.write_triangle_mesh(output1, result_mesh1)
    
main()