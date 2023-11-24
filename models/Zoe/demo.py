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

# ****************** [1] prepare model ******************
print("-"*10, "prepare model", "-"*10)

torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True) 
assert torch.cuda.is_available()
DEVICE = "cuda"

conf = get_config()
model = build_model(conf).to(DEVICE)
model.eval()

# ****************** [2] prepare img ******************
img = Image.open("data/test.png")
X = ToTensor()(img)

if X.shape[0] == 4 : # if RGBA image transform to RGB format
    X = X[:3, :, :]

X = X.unsqueeze(0).to(DEVICE)

# ****************** [3] predict depth ******************

print("-"*10, "start predicting", "-"*10)
with torch.no_grad():
    out = model.infer(X).cpu()

points = depth_to_points(out[0])
new_img = colorize(out)

depth = torch.from_numpy(new_img[:, :, 0])
assert torch.min(depth) == 0 and torch.max(depth) == 255

print(depth) # [width * height] tensor of integer 0~255


# Save depth map as image
pred = Image.fromarray(new_img)
pred = pred.resize(img.size, Image.LANCZOS)
pred.save("data/pred.png")
