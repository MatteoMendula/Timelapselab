import torch

print('Torch import is ok!')
print('Torch version: {}'.format(torch.__version__))

print('Is CUDA supported by this system? {}'.format(torch.cuda.is_available()))
print('CUDA version: {}'.format(torch.version.cuda))
print('CUDA device name: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))

import torchvision

print('Torchvision import is ok!')
print('Torchvision version: {}'.format(torchvision.__version__))
boxes = [[0.0, 0.0, 1.0, 1.0],
         [2.0, 1.0, 1.0, 2.0]]
boxes = torch.tensor(boxes)
scores = torch.tensor([1., 0.5])
keep = torchvision.ops.nms(boxes, scores, 0.7)
print('Torchvision operations should be ok!')
