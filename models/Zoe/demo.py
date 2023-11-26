# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

from torchvision.transforms import ToTensor
from PIL import Image
from zoedepth.utils.misc import colorize
import torch

from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config
from zoedepth.utils.geometry import depth_to_points
from zoedepth.ui.gradio_im_to_3d import get_mesh

# ****************** [1] prepare model ******************
print("-"*10, "prepare model", "-"*10)

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
assert torch.cuda.is_available()
DEVICE = "cuda"

conf = get_config()
model = build_model(conf).to(DEVICE)
model.eval()

# ****************** [2] prepare img ******************
img = Image.open("../../data/haechan_test/toycar_and_chair.png")
X = ToTensor()(img)

if X.shape[0] == 4 : # if RGBA image transform to RGB format
    X = X[:3, :, :]

X = X.unsqueeze(0).to(DEVICE)

# ****************** [3-1] predict depth ******************

print("-"*10, "start predicting", "-"*10)
with torch.no_grad():
    out = model.infer(X).cpu() #(1, H, W) : 1.xx ~ 2.xx

# tensor 내에 최댓값이 몇개인지 count해주는 함수
# max_value = torch.max(out)
# max_mask = out == max_value
# count_max = int(torch.sum(max_mask).item())
# print("Number of maximum: ", count_max)
#--
# out = out.squeeze()
# print(out.shape)
# torch.save(out, "../../data/output/depth_value.pt")

# points = depth_to_points(out[0].numpy()) #(H, W, 3) : -1.xx ~ 2.xx
# colorized_depth = colorize(out) #(H, W, 4) : 0 ~ 255  [100 100 100 255]

# depth = torch.from_numpy(colorized_depth[:, :, 0]) #(H, W) : depth랑 colorized_depth랑 image로 변환했을 때 차이점은 없음. 
# assert torch.min(depth) == 0 and torch.max(depth) == 255

# # Save depth map as image
# pred = Image.fromarray(colorized_depth)
# pred = pred.resize(img.size, Image.LANCZOS)
# pred.save("../../data/output/colorized_depthmap.png")

# Same result as above
# pred = Image.fromarray(colorized_depth[:, :, 0])
# pred = pred.resize(img.size)
# pred.save("../../data/output/depthmap.png")

# ****************** [3-2] depth_to_points ******************
pts3d = depth_to_points(out[0].numpy(), R=None, t=None)
print(pts3d.shape)
print(min(pts3d[:, 0]), min(pts3d[:, 1]), min(pts3d[:, 2]))

# ****************** [3-3] predict mesh ******************
output_path = "../../data/haechan_test/depth_to_mesh.ply"
get_mesh(model, img.convert('RGB'), output_path)