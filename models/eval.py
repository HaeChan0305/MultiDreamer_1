
"""
  conda activate eval
"""
import torch
from pytorch3d.loss import chamfer_distance
import open3d as o3d
import numpy as np
from utils import alert
import mesh2sdf

SDF_GRID = 64

def load_point_cloud(file_path):
  """
    load .ply file as point cloud, return points
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
  alert("load point cloud")
  # Load point clouds from .ply files and convert to tensor
  points1 = torch.tensor(load_point_cloud(path_1), dtype=torch.float32)
  points2 = torch.tensor(load_point_cloud(path_2), dtype=torch.float32)

  # Reshape tensors to (1, num_points, 3)
  points1 = points1.view(1, -1, 3)
  points2 = points2.view(1, -1, 3)

  alert("start sampling")
  # Randomly sample points if the point clouds have different numbers of points
  if points1.shape[1] == points2.shape[1]:
      max_points = max(points1.shape[1], points2.shape[1])
      
      # Ensure both point clouds have the same number of points (max_points)
      points1 = random_sampling(points1.view(-1, 3), max_points).view(1, -1, 3)
      points2 = random_sampling(points2.view(-1, 3), max_points).view(1, -1, 3)

  alert("calculate chamfer_distance")
  # Compute chamfer distance
  dist_chamfer, _ = chamfer_distance(points1, points2)

  # Get the mean chamfer distance
  mean_dist_chamfer = dist_chamfer.mean().item()

  return mean_dist_chamfer

def cal_vol_iou(path_1, path_2):
  """
    calculate a volumne IoU between to meshes
  """
  # Compute the volume of each mesh
  alert("calculate mesh volume")
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

  return iou

def main():
  path_1 = '../data/haechan_test/rotated_chair.ply'
  path_2 = '../data/haechan_test/rotated_chair.ply'

  print("chamfer distance : ", cal_chamf_dist(path_1, path_2))
  print("volume IoU : ", cal_vol_iou(path_1, path_2))

if __name__=="__main__":
  main()