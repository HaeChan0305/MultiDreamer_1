import numpy as np
import torch
import open3d as o3d
from open3d import io, geometry
from PIL import Image


#############
# input으로 받을 수 있게 변경할 것.
#############
input_path = "../../data/haechan_test/toycar_and_chair.png"
mask0_path = "../../data/haechan_test/mask0.jpg"
mask1_path = "../../data/haechan_test/mask1.jpg"
mesh0_path = "../../data/haechan_test/car.ply"
mesh1_path = "../../data/haechan_test/chair.ply"
depth_to_mesh_path = "../../data/haechan_test/depth_to_mesh.ply"

#############
# Load data
#############
input = np.array(Image.open(input_path))

mask0 = np.array(Image.open(mask0_path))
mask1 = np.array(Image.open(mask1_path))

mesh0 = o3d.io.read_triangle_mesh(mesh0_path)
mesh1 = o3d.io.read_triangle_mesh(mesh1_path)
depth_to_mesh = o3d.io.read_triangle_mesh(depth_to_mesh_path)

#############
# 기본 정보
#############
H, W, _ = input.shape #(H, W, 4), RGBA mode


#############
# depth_to_mesh 상에서 가장 앞에 점, 가장 뒤에 점 찾기
#   - depth_to_mesh의 view vector = [0, 0, -1]
#############

print(depth_to_mesh)
vertices = np.asarray(depth_to_mesh.vertices)
print(min(vertices[:, 0]), min(vertices[:, 1]), min(vertices[:, 2]))


