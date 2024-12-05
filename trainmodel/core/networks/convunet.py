import torch
import torch.nn as nn
import torch.nn.functional as F

def reversedl(it):
    return list(reversed(list(it)))

class ConvUNet(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                    (96, 96), 
                    (96, 128), 
                    (128, 192), 
                    (192, 256), 
                    (256, 384),
                    (512, 512, 384)
                 ],
                 pool=F.max_pool2d):
        super().__init__()
        
        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans

        for dims in dims_and_depths:
            layers = []

            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))
        
        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim
            
            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans, 
                dims_and_depths[:-1]+[reversedl(dims_and_depths[-1])] # Reverse bottleneck
            ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))
        
    def forward(self, x):
        skips = []

        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)
        
        x = skips[-1]
        outputs = []

        if len(self.out_path) == len(self.ds_path):
            outputs.append(self.out_path[-1](x))

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))
        
        return reversedl(outputs)

class ConvUNet_L(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                     (64, 64),
                     (64, 96),
                     (96, 128),
                     (128, 192),
                     (192, 256),
                     (512, 512, 256)
                 ],
                 pool=F.max_pool2d):
        super().__init__()

        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans

        for dims in dims_and_depths:
            layers = []

            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans,
                dims_and_depths[:-1] + [reversedl(dims_and_depths[-1])]  # Reverse bottleneck
        ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))

    def forward(self, x):
        skips = []

        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)

        x = skips[-1]
        outputs = []

        if len(self.out_path) == len(self.ds_path):
            outputs.append(self.out_path[-1](x))

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))

        return reversedl(outputs)

class ConvUNet_S(nn.Module):
    def __init__(self,
                 in_chans,
                 out_chans,
                 dims_and_depths=[
                     (32, 32),
                     (32, 64),
                     (64, 96),
                     (96, 160),
                     (160, 256),
                     (384, 384, 256)
                 ],
                 pool=F.max_pool2d):
        super().__init__()

        self.pool = pool

        self.ds_path = nn.ModuleList()
        prev_dim = in_chans

        for dims in dims_and_depths:
            layers = []

            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        self.us_path = nn.ModuleList()

        for dims in reversedl(dims_and_depths)[1:]:
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[-1], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]

            for dim in reversedl(dims)[1:]:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(
                out_chans,
                dims_and_depths[:-1] + [reversedl(dims_and_depths[-1])]  # Reverse bottleneck
        ):
            self.out_path.append(nn.Conv2d(dims[0], out_chan, 1))

    def forward(self, x):
        skips = []

        for stage in self.ds_path:
            x = stage(x)
            skips.append(x)
            x = self.pool(x, 2)

        x = skips[-1]
        outputs = []

        if len(self.out_path) == len(self.ds_path):
            outputs.append(self.out_path[-1](x))

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))

        return reversedl(outputs)

class ConvUNet_T(nn.Module):
    def __init__(self, in_chans, out_chans, pool=F.max_pool2d):
        super().__init__()

        self.pool = pool
        dims_and_depths_ds = [
            (32, 32),
            (64,),
            (96,),
            (144,),
            (248,),
            (384, 384)
        ]
        dims_and_depths_us = [
            (32, 32),
            (64, 48),
            (96, 96),
            (144, 128),
            (248, 224)
        ]
        self.K = len(out_chans)
        self.ds_path = nn.ModuleList()
        prev_dim = in_chans
        for dims in dims_and_depths_ds:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        self.us_path = nn.ModuleList()
        for dims in reversedl(dims_and_depths_us):
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[0], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]
            layers.append(nn.Conv2d(prev_dim, prev_dim, 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(out_chans, dims_and_depths_us):
            self.out_path.append(nn.Conv2d(dims[-1], out_chan, 1))

    def forward(self, x):
        skips = []

        for i in range(self.K):
            x = self.ds_path[i](x)
            skips.append(x)
            x = self.pool(x, 2)
        x = self.ds_path[self.K](x)
        skips.append(x)

        x = skips[-1]
        outputs = []

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode='nearest'), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))

        return reversedl(outputs)

class ConvUNet_T2(nn.Module):
    def __init__(self, in_chans, out_chans, pool=F.max_pool2d):
        super().__init__()

        self.pool = pool
        dims_and_depths_ds = [
            (32, 32),
            (64,),
            (96,),
            (160,),
            (256,),
            (384, 384)
        ]
        dims_and_depths_us = [
            (32, 32),
            (64, 64),
            (96, 96),
            (160, 128),
            (256, 224)
        ]
        self.K = len(out_chans)
        self.ds_path = nn.ModuleList()
        prev_dim = in_chans
        for dims in dims_and_depths_ds:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        self.us_path = nn.ModuleList()
        for dims in reversedl(dims_and_depths_us):
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[0], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]
            layers.append(nn.Conv2d(prev_dim, prev_dim, 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(out_chans, dims_and_depths_us):
            self.out_path.append(nn.Conv2d(dims[-1], out_chan, 1))

    def forward(self, x):
        skips = []

        for i in range(self.K):
            x = self.ds_path[i](x)
            skips.append(x)
            x = self.pool(x, 2)
        x = self.ds_path[self.K](x)
        skips.append(x)

        x = skips[-1]
        outputs = []

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode='nearest'), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))

        return reversedl(outputs)

class ConvUNet_oidn(nn.Module):
    def __init__(self, in_chans, out_chans, pool=F.max_pool2d):
        super().__init__()

        self.pool = pool
        dims_and_depths_ds = [
            (32, 32),
            (48,),
            (64,),
            (80,),
            (96,),
            (192, 192)
        ]
        dims_and_depths_us = [
            (32, 32),
            (48, 64),
            (64, 96),
            (80, 128),
            (96, 160)
        ]
        self.K = len(out_chans)
        self.ds_path = nn.ModuleList()
        prev_dim = in_chans
        for dims in dims_and_depths_ds:
            layers = []
            for dim in dims:
                layers.append(nn.Conv2d(prev_dim, dim, 3, padding='same'))
                layers.append(nn.LeakyReLU(0.3))
                prev_dim = dim

            self.ds_path.append(nn.Sequential(*layers))

        self.us_path = nn.ModuleList()
        for dims in reversedl(dims_and_depths_us):
            layers = []
            layers.append(nn.Conv2d(prev_dim + dims[0], dims[-1], 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))
            prev_dim = dims[-1]
            layers.append(nn.Conv2d(prev_dim, prev_dim, 3, padding='same'))
            layers.append(nn.LeakyReLU(0.3))

            self.us_path.append(nn.Sequential(*layers))

        self.out_path = nn.ModuleList()
        for out_chan, dims in zip(out_chans, dims_and_depths_us):
            self.out_path.append(nn.Conv2d(dims[-1], out_chan, 1))

    def forward(self, x):
        skips = []

        for i in range(self.K):
            x = self.ds_path[i](x)
            skips.append(x)
            x = self.pool(x, 2)
        x = self.ds_path[self.K](x)
        skips.append(x)

        x = skips[-1]
        outputs = []

        for stage, (i, skip) in zip(self.us_path, reversedl(enumerate(skips))[1:]):
            x = torch.cat((F.interpolate(x, scale_factor=2, mode='nearest'), skip), 1)
            x = stage(x)
            if i < len(self.out_path):
                outputs.append(self.out_path[i](x))

        return reversedl(outputs)