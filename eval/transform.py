import numpy as np
import cupy as cp
import torch
import open3d as o3d
from PIL import Image
import scipy
import os
import argparse


#############
# Sampling Strategy
#   - vertex norm이 view direction과 90도 이상 차이나는 vertex 제외
#############
def uniform_sampling(points, M):
    assert points.shape[0] > M
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    sampling_rate = points.shape[0] // M
    sampled_point_cloud = point_cloud.uniform_down_sample(every_k_points=sampling_rate)
    sampled_points = np.array(sampled_point_cloud.points)
    
    print("\t\t\t>>> uniform_sampling : ", points.shape[0], "->", sampled_points.shape[0])
    return sampled_points
    

def random_sampling(points, M):
    N = points.shape[0]
    assert N > M
    
    indices = np.random.choice(N, size=M, replace=False)
    sampled_points = points[indices, :]
    
    print("\t\t\t>>> random_sampling : ", points.shape[0], "->", sampled_points.shape[0])
    return sampled_points


def sampling(gt_mesh, cp_mesh):
    gt_points = np.array(gt_mesh.vertices)
    cp_points = np.array(cp_mesh.vertices)
    
    # 원래 코드
    # sampled_points = uniform_sampling(sampled_points, M)
    # N = sampled_points.shape[0]
    
    # if N > M:
    #     sampled_points = random_sampling(sampled_points, M)
    # elif N < M:
    #     sampled_points = random_sampling(masked_depth_to_points, N)
    
    # 연산량 감소하기 위한 코드
    sampled_gt_points = uniform_sampling(gt_points, 10000)
    sampled_cp_points = uniform_sampling(cp_points, 10000)
    N = sampled_gt_points.shape[0]
    M = sampled_cp_points.shape[0]
    
    if N > M:
        sampled_gt_points = random_sampling(sampled_gt_points, M)
    elif N < M:
        sampled_cp_points = random_sampling(sampled_cp_points, N)
    
    assert sampled_gt_points.shape[0] == sampled_cp_points.shape[0]
    return sampled_gt_points, sampled_cp_points
    

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

def process_per_object(gt_mesh, mesh):
    # Sampling
    print("\t\t>>> Sampling ...")
    sampled_gt_points, sampled_mesh_points = sampling(gt_mesh, mesh)
    
    # Optimizing
    print("\t\t>>> Optimizing ...")
    optimized_params = optimizing(sampled_gt_points, sampled_mesh_points)
    
    # Applying
    print("\t\t>>> Applying ...")
    result_mesh = applying(mesh, optimized_params)
    
    return result_mesh


def process(sample):
    DIR = f"../../data/eval/"
    
    # input으로 받을 수 있게 변경할 것.
    ground_truth_path = DIR + f"{sample}_ground_truth.ply"
    mesh_path = DIR + f"{sample}_mesh.ply"
    result_merged_mesh_path = DIR + f"{sample}_result_merged_mesh.ply"

    # Load data
    print("\t>>> Loading data ...")
    ground_truth = o3d.io.read_triangle_mesh(ground_truth_path)
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    result_merged_mesh = o3d.io.read_triangle_mesh(result_merged_mesh_path)

    # Process
    print("\t>>> Object 0 ...")
    trans_mesh = process_per_object(ground_truth, mesh)
    print("\t>>> Object 1 ...")
    trans_result_merged_mesh = process_per_object(ground_truth, result_merged_mesh)
    
    # Exporting
    print("\t>>> Exporting ...")
    o3d.io.write_triangle_mesh(DIR + f"{sample}_trans_mesh.ply", trans_mesh)
    o3d.io.write_triangle_mesh(DIR + f"{sample}_trans_result_merged_mesh.ply", trans_result_merged_mesh)
    
    
def main(args):
    samples = args.samples
    print(samples)
    
    for sample in samples:
        print(f">>> Start sample {sample}")
        try:
            process(sample)
            print(f">>> SUCCESS : sample {sample}")
        except Exception as e:
            print(e)
            print(f">>> FAIL : sample {sample}")


parser = argparse.ArgumentParser()
parser.add_argument("--samples", nargs='+')
args = parser.parse_args()

main(args)

