"""
  Example/
    conda activate eval
    python eval.py --dir "../data/eval" --input 23 --filename "eval"

  Option/
    --dir 
      : (Require, str) The path of the folder where target .ply file exists
    --input 
      : (Required, int) In the "dir", 3 files should be exist, 
        {input}_ground_truth.ply / the ground truth mesh
        {input}__result_merged_mesh.ply / the result of MutiDreamer
        {input}_mesh.ply / the result of SyncDreamer(baseline)
    --filename 
      : (Optional, str) If you want to record your result to csv file, provide the file name without ".csv"
        If the file exist in the same folder, it will add new line. If not, it will create a new file.
"""

import argparse
import torch
from pytorch3d.loss import chamfer_distance
import open3d as o3d
import numpy as np
import mesh2sdf

from utils import add_csv

SDF_GRID = 128

def load_point_cloud(file_path):
  """
    Brief description of the function.

    Detailed description and explanation of parameters.

    :file_path: Description of arg1.
    :return: Description of the return value.
  """
  ply = o3d.io.read_point_cloud(file_path)
  points = np.asarray(ply.points)
  return points

def load_triangle_mesh(file_path):
  """
    load .ply file as mesh, return vertices and faces
  """
  mesh = o3d.io.read_triangle_mesh(file_path)
  return torch.tensor(mesh.vertices), torch.tensor(mesh.triangles)
    
def random_sampling(points, num_points):
  """
    random sample 'num_points' points from 'points'
  """
  indices = torch.randperm(points.shape[0])[:20000]
  return points[indices]

def compute_mesh_volume(mesh_vertices, mesh_faces):
  """
    compute the volume of a mesh using the divergence theorem.
  """
  # Compute face normals
  face_normals = torch.cross(
      mesh_vertices[mesh_faces[:, 1]] - mesh_vertices[mesh_faces[:, 0]],
      mesh_vertices[mesh_faces[:, 2]] - mesh_vertices[mesh_faces[:, 0]],
      dim=-1
  )

  # Compute the volume using the divergence theorem
  mesh_volume = torch.abs(torch.einsum('i,ij->', mesh_vertices[mesh_faces[:, 0]], face_normals) / 6.0)

  return mesh_volume

def cal_chamf_dist(path_1, path_2):
  """
    calculate a chamfer distance between to meshes
  """
  # Load point clouds from .ply files and convert to tensor
  points1 = torch.tensor(load_point_cloud(path_1), dtype=torch.float32)
  points2 = torch.tensor(load_point_cloud(path_2), dtype=torch.float32)

  # Reshape tensors to (1, num_points, 3)
  points1 = points1.view(1, -1, 3)
  points2 = points2.view(1, -1, 3)

  # Randomly sample points if the point clouds have different numbers of points
  if points1.shape[1] != points2.shape[1]:
      max_points = max(points1.shape[1], points2.shape[1])
      
      # Ensure both point clouds have the same number of points (max_points)
      points1 = random_sampling(points1.view(-1, 3), max_points).view(1, -1, 3)
      points2 = random_sampling(points2.view(-1, 3), max_points).view(1, -1, 3)

  # Compute chamfer distance
  dist_chamfer, _ = chamfer_distance(points1, points2)

  # Get the mean chamfer distance
  mean_dist_chamfer = dist_chamfer.mean().item()

  return mean_dist_chamfer

def cal_vol_iou(path_1, path_2):
  """
    calculate a volumne IoU, fscore between to meshes
  """
  # Compute the volume of each mesh
  # Load triangle meshes
  mesh1 = o3d.io.read_triangle_mesh(path_1)
  mesh2 = o3d.io.read_triangle_mesh(path_2)

  # Convert Open3D mesh to numpy arrays
  mesh1_vertices = np.asarray(mesh1.vertices)
  mesh1_faces = np.asarray(mesh1.triangles)
  mesh2_vertices = np.asarray(mesh2.vertices)
  mesh2_faces = np.asarray(mesh2.triangles)

  # Compute SDF for each mesh
  sdf_pr = mesh2sdf.compute(mesh1_vertices, mesh1_faces, SDF_GRID, fix=False, return_mesh=False)
  sdf_gt = mesh2sdf.compute(mesh2_vertices, mesh2_faces, SDF_GRID, fix=False, return_mesh=False)

  # Convert SDF to binary volumes
  vol_pr = sdf_pr < 0
  vol_gt = sdf_gt < 0

  # Compute Volume IoU
  intersection = np.sum(vol_pr & vol_gt)
  union = np.sum(vol_gt | vol_pr)
  iou = intersection / union

  precision = intersection / np.sum(vol_gt)
  recall = intersection / np.sum(vol_pr)
  fscore = 2 * (precision * recall) / (precision + recall)

  return iou, fscore

def main(arg):
  # gt
  path_1 = arg.dir + f'/{arg.input}_ground_truth.ply'
  # our
  path_2 = arg.dir + f'/{arg.input}_result_merged_mesh.ply'
  # base
  path_3 = arg.dir + f'/{arg.input}_mesh.ply'

  cham = cal_chamf_dist(path_1, path_2)
  iou, fscore = cal_vol_iou(path_1, path_2)
  cham_base = cal_chamf_dist(path_1, path_3)
  iou_base, fscore_base = cal_vol_iou(path_1, path_3)

  print(f"{arg.input} result => cham [{cham} < {cham_base}] / iou [{iou} > {iou_base}] / f [{fscore} > {fscore_base}]")

  # record at csv file
  if arg.filename != None :
    add_csv(arg.filename, [arg.input, cham, cham_base, iou, iou_base, fscore, fscore_base])

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", required=True, type=int)
  parser.add_argument("--dir", required=True, type=str)
  parser.add_argument("--filename", type=str)

  arg = parser.parse_args()
  main(arg)