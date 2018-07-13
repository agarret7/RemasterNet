import torch
import numpy as np
from frame_interpolation import FrameInterpolator

model = FrameInterpolator().cuda()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

for epoch in range(10000):

    inp1 = np.random.rand(1, 3, 1920, 1080).astype(np.float32)
    inp1 = torch.from_numpy(inp1).cuda()

    inp2 = np.random.rand(1, 3, 1920, 1080).astype(np.float32)
    inp2 = torch.from_numpy(inp2).cuda()

    target = (inp1 + inp2) / 2

    optimizer.zero_grad()
    out = model((inp1, inp2))

    loss = torch.nn.MSELoss()(out, target)

    print(loss)

    loss.backward()
    optimizer.step()
