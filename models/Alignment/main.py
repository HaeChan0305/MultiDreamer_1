import numpy as np
import cupy as cp
import torch
import open3d as o3d
from PIL import Image
import scipy


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
    sampled_point_cloud = point_cloud.uniform_down_sample(every_k_points=sampling_rate)
    sampled_points = np.array(sampled_point_cloud.points)
    
    print(">>> uniform_sampling : ", points.shape[0], "->", sampled_points.shape[0])
    return sampled_points
    

def random_sampling(points, M):
    N = points.shape[0]
    assert N > M
    
    indices = np.random.choice(N, size=M, replace=False)
    sampled_points = points[indices, :]
    
    print(">>> random_sampling : ", points.shape[0], "->", sampled_points.shape[0])
    return sampled_points


def sampling(mesh, masked_depth_to_points):
    mesh_points = np.array(mesh.vertices)
    M, _ = masked_depth_to_points.shape
    
    sampled_points = vertex_normal_sampling(mesh_points, M)
    
    # 원래 코드
    # sampled_points = uniform_sampling(sampled_points, M)
    # N = sampled_points.shape[0]
    
    # if N > M:
    #     sampled_points = random_sampling(sampled_points, M)
    # elif N < M:
    #     sampled_points = random_sampling(masked_depth_to_points, N)
    
    # 연산량 감소하기 위한 코드
    sampled_points = uniform_sampling(sampled_points, 10000)
    sampled_depth_to_points = uniform_sampling(masked_depth_to_points, 10000)
    N = sampled_points.shape[0]
    M = sampled_depth_to_points.shape[0]
    
    if N > M:
        sampled_points = random_sampling(sampled_points, M)
    elif N < M:
        sampled_depth_to_points = random_sampling(sampled_depth_to_points, N)
    
    
    assert sampled_points.shape[0] == sampled_depth_to_points.shape[0]
    return sampled_points, sampled_depth_to_points
    

#############
# Optimizing
#   y와 x @ d에 대한 chamfer distance가 최소가 되는 d를 구하는 문제
#   - y : (M', 3), object별 masked 된 depth_to_points
#   - x : (M', 3), X의 points만큼 sampled된 vertices
#   - d : (4, 4), X_를 scaling, translation 시키는 transformation matrix 
#############
def chamfer_distance(X, Y):
    # print(">>> IN  : chamfer_distacne()")
    dist_x_to_y = cp.linalg.norm(X[:, None, :] - Y, axis=-1).min(axis=-1)
    dist_y_to_x = cp.linalg.norm(Y[:, None, :] - X, axis=-1).min(axis=-1)
    
    total_distance = cp.sum(dist_x_to_y) + cp.sum(dist_y_to_x)
    # print(">>> OUT : chamfer_distacne()")
    return total_distance

def params_to_matrix(params):
    '''
    params : np.array([s, tx, ty, tz])
    '''
    s, tx, ty, tz = params
    return np.array([[s,   0,  0, 0],
                     [0,   s,  0, 0],
                     [0,   0,  s, 0],
                     [tx, ty, tz, 1]])

def objective_function(params, x, y):
    '''
    params : np.array([s, tx, ty, tz]), six elements for transformation
    x : sampled vertices in the mesh
    y : masked points in the depth_to_points
    '''
    matrix_cpu = params_to_matrix(params)
    matrix_gpu = cp.asarray(matrix_cpu)
    
    x_cpu = np.hstack((x, np.ones((x.shape[0], 1))))
    x_gpu = cp.asarray(x_cpu)
    y_gpu_ = x_gpu @ matrix_gpu
    y_gpu_ = y_gpu_[:, :-1]
    y_gpu = cp.asarray(y)
    
    result_gpu = chamfer_distance(y_gpu, y_gpu_)
    result_cpu = result_gpu.get()
    return result_cpu


def optimizing(x, y):
    '''
    x : sampled vertices in the mesh
    y : masked points in the depth_to_points
    params : np.array([sx, sy, sz, tx, ty, tz]), six elements for transformation
    '''
    initial_params = np.random.randn(4)
    result = scipy.optimize.minimize(objective_function, initial_params, args=(x, y), method='BFGS')
    
    optimized_params = result.x
    optimized_matrix = params_to_matrix(optimized_params)
    return optimized_matrix
    

def main():
    # input으로 받을 수 있게 변경할 것.
    mask0_path = "../../data/output/1/mask0.jpg"
    mask1_path = "../../data/output/1/mask1.jpg"
    mesh0_path = "../../data/output/1/mesh0.ply"
    mesh1_path = "../../data/output/1/mesh1.ply"
    depth_to_points_path = "../../data/output/1/depth.npy"

    # Load data
    print(">>> Loading data ...")
    mask0 = np.array(Image.open(mask0_path))
    mask1 = np.array(Image.open(mask1_path))
    mesh0 = o3d.io.read_triangle_mesh(mesh0_path)
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    depth_to_points = np.load(depth_to_points_path)

    # Rotation
    print(">>> Rotation ...")
    rotated_mesh0 = rotation_alignment(mesh0)
    rotated_mesh1 = rotation_alignment(mesh1)

    o3d.io.write_triangle_mesh('../../data/output/1/rotated_mesh0.ply', rotated_mesh0)
    o3d.io.write_triangle_mesh('../../data/output/1/rotated_mesh1.ply', rotated_mesh1)
    
    # Masking
    print(">>> Masking ...")
    masked_points0 = masking(mask0, depth_to_points)
    masked_points1 = masking(mask1, depth_to_points)
    
    # Sampling
    print(">>> Sampling ...")
    sampled_points0, sampled_masked_points0 = sampling(rotated_mesh0, masked_points0)
    sampled_points1, sampled_masked_points1 = sampling(rotated_mesh1, masked_points1)
    
    # Optimizing
    print(">>> Optimizing ...")
    optimized_matrix0 = optimizing(sampled_points0, sampled_masked_points0)
    optimized_matrix1 = optimizing(sampled_points1, sampled_masked_points1)
    
    # Applying
    print(">>> Applying ...")
    result_mesh0 = rotated_mesh0.transform(optimized_matrix0)
    result_mesh1 = rotated_mesh1.transform(optimized_matrix1)
    
    # Exporting
    print(">>> Exporting ...")
    output0 = "../../data/output/1/result_mesh0.ply"
    output1 = "../../data/output/1/result_mesh1.ply"
    o3d.io.write_triangle_mesh(output0, result_mesh0)
    o3d.io.write_triangle_mesh(output1, result_mesh1)
    
main()