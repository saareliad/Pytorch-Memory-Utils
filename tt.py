import torch
import inspect

from torchvision import models
from gpu_mem_track import  MemTracker
from t3nsor import TTLinear

device = torch.device('cuda:0')

frame = inspect.currentframe()          # define a frame to track
gpu_tracker = MemTracker(frame)         # define a GPU tracker

gpu_tracker.track()                     # run function between the code line where uses GPU

model = TTLinear(410,4100).to(device).eval()

gpu_tracker.track()                     # run function between the code line where uses GPU

x = torch.randn(30, 3, 410).to(device)  #

gpu_tracker.track()

model(x)

gpu_tracker.track()

model(x)
gpu_tracker.track()

x =  x.cpu()
model = model.cpu()
torch.cuda.empty_cache()

gpu_tracker.track()
