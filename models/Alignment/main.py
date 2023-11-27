import numpy as np
import cupy as cp
import torch
import open3d as o3d
from PIL import Image
import scipy
import os

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
    H, W = mask.shape
    H_, W_, _ = depth_to_points.shape
    assert(H == H_ and W == W_)
    
    return depth_to_points[np.where(mask==255)]
    

#############
# Sampling Strategy
#   - vertex norm이 view direction과 90도 이상 차이나는 vertex 제외
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
    
    print("\t\t>>> vertex_normal_sampling : ", points.shape[0], "->", sampled_points.shape[0])
    assert sampled_points.shape[0] > M
    return sampled_points
    
    
def uniform_sampling(points, M):
    assert points.shape[0] > M
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    sampling_rate = points.shape[0] // M
    sampled_point_cloud = point_cloud.uniform_down_sample(every_k_points=sampling_rate)
    sampled_points = np.array(sampled_point_cloud.points)
    
    print("\t\t>>> uniform_sampling : ", points.shape[0], "->", sampled_points.shape[0])
    return sampled_points
    

def random_sampling(points, M):
    N = points.shape[0]
    assert N > M
    
    indices = np.random.choice(N, size=M, replace=False)
    sampled_points = points[indices, :]
    
    print("\t\t>>> random_sampling : ", points.shape[0], "->", sampled_points.shape[0])
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
    sampled_points = uniform_sampling(sampled_points, 15000)
    sampled_depth_to_points = uniform_sampling(masked_depth_to_points, 15000)
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
    dist_x_to_y = cp.linalg.norm(X[:, None, :] - Y, axis=-1).min(axis=-1)
    dist_y_to_x = cp.linalg.norm(Y[:, None, :] - X, axis=-1).min(axis=-1)
    
    total_distance = cp.sum(dist_x_to_y) + cp.sum(dist_y_to_x)
    return total_distance

def params_to_matrix(params):
    '''
    params : np.array([s, tx, ty, tz])
    '''
    s, tx, ty, tz = params
    s = np.abs(s)
    assert s > 0
    return np.array([[ s,  0,  0,  0],
                     [ 0,  s,  0,  0],
                     [ 0,  0,  s,  0],
                     [tx, ty, tz,  1]])

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
    params : np.array([s, tx, ty, tz]), six elements for transformation
    '''
    initial_params = np.random.randn(4)
    result = scipy.optimize.minimize(objective_function, initial_params, args=(x, y), method='BFGS')
    
    optimized_params = result.x
    return optimized_params
    
#############
# Applying
#############
def applying(mesh, optimized_params):
    s, tx, ty, tz = optimized_params
    s = np.abs(s)
    
    mesh.scale(s, center=(0, 0, 0))
    mesh.translate((tx, ty, tz))
    
    return mesh
    

def process_per_object(mask, mesh, depth_to_points):
    # Rotation
    print("\t>>> Rotation ...")
    rotated_mesh = rotation_alignment(mesh)
    
    # Masking
    print("\t>>> Masking ...")
    masked_points = masking(mask, depth_to_points)
    
    # Sampling
    print("\t>>> Sampling ...")
    sampled_points, sampled_masked_points = sampling(rotated_mesh, masked_points)
    
    # Optimizing
    print("\t>>> Optimizing ...")
    optimized_params = optimizing(sampled_points, sampled_masked_points)
    
    # Applying
    print("\t>>> Applying ...")
    result_mesh = applying(rotated_mesh, optimized_params)
    
    return result_mesh
    

def process(sample):
    DIR = f"../../data/output/{sample}/"
    
    # input으로 받을 수 있게 변경할 것.
    mask0_path = DIR + "mask0.jpg"
    mask1_path = DIR + "mask1.jpg"
    mesh0_path = DIR + "mesh0.ply"
    mesh1_path = DIR + "mesh1.ply"
    depth_to_points_path = DIR + "depth.npy"

    # Load data
    print(">>> Loading data ...")
    mask0 = np.array(Image.open(mask0_path))
    mask1 = np.array(Image.open(mask1_path))
    mesh0 = o3d.io.read_triangle_mesh(mesh0_path)
    mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
    depth_to_points = np.load(depth_to_points_path)

    # Process
    print(">>> Object 0 ...")
    result_mesh0 = process_per_object(mask0, mesh0, depth_to_points)
    print(">>> Object 1 ...")
    result_mesh1 = process_per_object(mask1, mesh1, depth_to_points)
    merged_mesh = result_mesh0 + result_mesh1
    
    # Exporting
    print(">>> Exporting ...")
    o3d.io.write_triangle_mesh(DIR + "result_mesh0.ply", result_mesh0)
    o3d.io.write_triangle_mesh(DIR + "result_mesh1.ply", result_mesh1)
    o3d.io.write_triangle_mesh(DIR + "result_merged_mesh.ply", merged_mesh)
    
    
def main():
    samples = [1, 4, 6, 9, 10, 12, 13, 14, 15, 19, 22]
    for sample in samples:
        print(f">>> Start sample {sample}")
        try:
            process(sample)
            print(f">>> SUCCESS : sample {sample}")
        except Exception as e:
            print(e)
            print(f">>> FAIL : sample {sample}")

main()