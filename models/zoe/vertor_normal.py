import trimesh
import numpy as np
import matplotlib.pyplot as plt

# 예제로 사용할 *.ply 파일 로드
mesh = trimesh.load("../../data/output/4/result_mesh1.ply")

# 벡터 노멀 계산
mesh.compute_vertex_normals()

# Mesh와 벡터 노멀 시각화
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Mesh 표시
mesh.show(ax=ax)

# 벡터 노멀 표시
scale_factor = 0.1  # 벡터 노멀의 크기 조절을 위한 스케일 팩터
for vertex, normal in zip(mesh.vertices, mesh.vertex_normals):
    arrow_end = vertex + scale_factor * normal
    ax.plot([vertex[0], arrow_end[0]], [vertex[1], arrow_end[1]], [vertex[2], arrow_end[2]], color='r')

# 축 범위 설정
ax.set_xlim(mesh.bounds[0])
ax.set_ylim(mesh.bounds[1])
ax.set_zlim(mesh.bounds[2])

# 그래프 표시
plt.show()
