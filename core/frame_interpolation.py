import torch
import numpy as np

class FrameInterpolator(torch.nn.Module):

    def __init__(self):

        super(FrameInterpolator, self).__init__()

        self.activation_func = torch.nn.ReLU()

        self.conv1 = torch.nn.Sequential(*[
            torch.nn.Conv2d(3, 64, 5, 1, 2),
            torch.nn.MaxPool2d(2)
        ])

        self.proj1 = torch.nn.Conv2d(3, 64, 1, 2, 0)

        self.conv2 = torch.nn.Sequential(*[
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.MaxPool2d(2)
        ])

        self.proj2 = torch.nn.Conv2d(64, 64, 1, 2, 0)

        self.conv3 = torch.nn.Sequential(*[
            torch.nn.Conv2d( 64, 128, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.MaxPool2d(2)
        ])

        self.proj3 = torch.nn.Conv2d(64, 128, 1, 2, 0)

        self.deconv1 = torch.nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.proj4   = torch.nn.ConvTranspose2d(128, 64, 1, 2, 0, 1)

        self.deconv2 = torch.nn.ConvTranspose2d( 64, 64, 3, 2, 1, 1)
        self.proj5   = torch.nn.ConvTranspose2d( 64, 64, 1, 2, 0, 1)

        self.deconv3 = torch.nn.ConvTranspose2d( 64, 64, 3, 2, 1, 1)
        self.proj6   = torch.nn.ConvTranspose2d( 64, 64, 1, 2, 0, 1)

        self.conv4   = torch.nn.Conv2d(64, 3, 3, 1, 1)
        self.proj7   = torch.nn.Conv2d(64, 3, 1, 1, 0)

    def forward(self, x):

        f1, f2 = x

        f1 = self.activation_func(self.conv1(f1) + self.proj1(f1))
        f1 = self.activation_func(self.conv2(f1) + self.proj2(f1))
        f1 = self.activation_func(self.conv3(f1) + self.proj3(f1))

        f2 = self.activation_func(self.conv1(f2) + self.proj1(f2))
        f2 = self.activation_func(self.conv2(f2) + self.proj2(f2))
        f2 = self.activation_func(self.conv3(f2) + self.proj3(f2))

        f = f1 + f2

        f = self.activation_func(self.deconv1(f) + self.proj4(f))
        f = self.activation_func(self.deconv2(f) + self.proj5(f))
        f = self.activation_func(self.deconv3(f) + self.proj6(f))
        f = self.activation_func(self.conv4(f)   + self.proj7(f))

        return f

if __name__ == "__main__":

    model = FrameInterpolator().cuda()

    inp1 = np.random.rand(1, 3, 128, 128).astype(np.float32)
    inp1 = torch.from_numpy(inp1).cuda()

    inp2 = np.random.rand(1, 3, 128, 128).astype(np.float32)
    inp2 = torch.from_numpy(inp2).cuda()

    out = model((inp1, inp2))

    print(out.size())
