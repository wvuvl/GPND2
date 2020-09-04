# Copyright 2019-2020 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
import numpy as np
import lreq as ln
import math
from registry import *


def upscale2d(x, factor=2):
    s = x.shape
    x = torch.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = x.repeat(1, 1, 1, factor, 1, factor)
    x = torch.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def downscale2d(x, factor=2):
    return F.avg_pool2d(x, factor, factor)


class Blur(nn.Module):
    def __init__(self, channels):
        super(Blur, self).__init__()
        f = np.array([1, 2, 1], dtype=np.float32)
        f = f[:, np.newaxis] * f[np.newaxis, :]
        f /= np.sum(f)
        kernel = torch.Tensor(f).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('weight', kernel)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, weight=self.weight, groups=self.groups, padding=1)


class EncodeBlock(nn.Module):
    def __init__(self, inputs, outputs, last=False):
        super(EncodeBlock, self).__init__()
        self.conv_1 = ln.Conv2d(inputs, inputs, 3, 1, 1, bias=False)
        # self.conv_1 = ln.Conv2d(inputs + (1 if last else 0), inputs, 3, 1, 1, bias=False)
        self.bias_1 = nn.Parameter(torch.Tensor(1, inputs, 1, 1))
        self.blur = Blur(inputs)
        self.last = last
        if last:
            self.dense = ln.Linear(inputs * 4 * 4, outputs)
        else:
            self.conv_2 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)

        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x):
        x = self.conv_1(x) + self.bias_1
        x = F.leaky_relu(x, 0.2)

        if self.last:
            x = self.dense(x.view(x.shape[0], -1))
        else:
            x = self.conv_2(self.blur(x))
            x = downscale2d(x)
            x = x + self.bias_2
            x = F.leaky_relu(x, 0.2)
        return x


class DecodeBlock(nn.Module):
    def __init__(self, inputs, outputs, has_first_conv=True, layer=0):
        super(DecodeBlock, self).__init__()
        self.has_first_conv = has_first_conv
        self.inputs = inputs
        self.has_first_conv = has_first_conv
        if has_first_conv:
            self.conv_1 = ln.Conv2d(inputs, outputs, 3, 1, 1, bias=False)
        else:
            self.dense = ln.Linear(inputs, outputs * 4 * 4)

        self.blur = Blur(outputs)
        self.noise_weight_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_1.data.zero_()
        self.bias_1 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        self.conv_2 = ln.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.noise_weight_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))
        self.noise_weight_2.data.zero_()
        self.bias_2 = nn.Parameter(torch.Tensor(1, outputs, 1, 1))

        self.layer = layer

        with torch.no_grad():
            self.bias_1.zero_()
            self.bias_2.zero_()

    def forward(self, x, noise):
        if self.has_first_conv:
            x = upscale2d(x)
            x = self.conv_1(x)
            x = self.blur(x)
        else:
            x = self.dense(x).view(x.shape[0], -1, 4, 4)

        if noise:
            if noise == 'batch_constant':
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn([1, 1, x.shape[2], x.shape[3]]))
            else:
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_1,
                                  tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        else:
            s = math.pow(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8
        x = x + self.bias_1

        x = F.leaky_relu(x, 0.2)

        x = self.conv_2(x)

        if noise:
            if noise == 'batch_constant':
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn([1, 1, x.shape[2], x.shape[3]]))
            else:
                x = torch.addcmul(x, value=1.0, tensor1=self.noise_weight_2,
                                  tensor2=torch.randn([x.shape[0], 1, x.shape[2], x.shape[3]]))
        else:
            s = math.pow(self.layer + 1, 0.5)
            x = x + s * torch.exp(-x * x / (2.0 * s * s)) / math.sqrt(2 * math.pi) * 0.8

        x = x + self.bias_2

        x = F.leaky_relu(x, 0.2)
        return x


class FromRGB(nn.Module):
    def __init__(self, channels, outputs):
        super(FromRGB, self).__init__()
        self.from_rgb = ln.Conv2d(channels, outputs, 1, 1, 0)

    def forward(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        return x


class ToRGB(nn.Module):
    def __init__(self, inputs, channels):
        super(ToRGB, self).__init__()
        self.inputs = inputs
        self.channels = channels
        self.to_rgb = ln.Conv2d(inputs, channels, 1, 1, 0, gain=0.03)

    def forward(self, x):
        x = self.to_rgb(x)
        return x


@ENCODERS.register("EncoderDefault")
class Encoder(nn.Module):
    def __init__(self, startf, maxf, layer_count, latent_size, channels=3):
        super(Encoder, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count
        self.channels = channels
        self.latent_size = latent_size

        mul = 2
        inputs = startf
        self.encode_block: nn.ModuleList[EncodeBlock] = nn.ModuleList()

        resolution = 2 ** (self.layer_count + 1)

        self.from_rgb = FromRGB(channels, inputs)

        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            last = i == (self.layer_count - 1)
            if last:
                outputs = latent_size

            block = EncodeBlock(inputs, outputs, last)

            resolution //= 2

            self.encode_block.append(block)
            inputs = outputs
            mul *= 2

    def encode(self, x):
        x = self.from_rgb(x)
        x = F.leaky_relu(x, 0.2)

        for i in range(self.layer_count):
            x = self.encode_block[i](x)
        return x

    def forward(self, x):
        return self.encode(x)


@GENERATORS.register("GeneratorDefault")
class Generator(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3):
        super(Generator, self).__init__()
        self.maxf = maxf
        self.startf = startf
        self.layer_count = layer_count

        self.channels = channels

        mul = 2 ** (self.layer_count - 1)

        inputs = latent_size

        self.layer_to_resolution = [0 for _ in range(layer_count)]
        resolution = 2

        self.style_sizes = []

        self.decode_block: nn.ModuleList[DecodeBlock] = nn.ModuleList()
        for i in range(self.layer_count):
            outputs = min(self.maxf, startf * mul)

            has_first_conv = i != 0

            block = DecodeBlock(inputs, outputs, has_first_conv, layer=i)

            resolution *= 2
            self.layer_to_resolution[i] = resolution

            self.style_sizes += [2 * (inputs if has_first_conv else outputs), 2 * outputs]

            self.decode_block.append(block)
            inputs = outputs
            mul //= 2

        self.to_rgb = ToRGB(outputs, channels)

    def decode(self, x, noise):
        for i in range(self.layer_count):
            x = self.decode_block[i](x, noise)

        x = self.to_rgb(x)
        return x

    def forward(self, x, noise):
        return self.decode(x, noise)


class MappingBlock(nn.Module):
    def __init__(self, inputs, output, lrmul):
        super(MappingBlock, self).__init__()
        self.fc = ln.Linear(inputs, output, lrmul=lrmul)

    def forward(self, x):
        x = F.leaky_relu(self.fc(x), 0.2)
        return x


@DISCRIMINATORS.register("D")
class Discriminator(nn.Module):
    def __init__(self, mapping_layers=5, net_inputs=256, hidden_size=256, net_outputs=1):
        super(Discriminator, self).__init__()
        inputs = net_inputs
        self.mapping_layers = mapping_layers
        self.map_blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()
        for i in range(mapping_layers):
            outputs = hidden_size if i == mapping_layers - 1 else net_outputs
            block = ln.Linear(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.map_blocks.append(block)

    def forward(self, x):
        for i in range(self.mapping_layers):
            x = self.map_blocks[i](x)
        return x


@DISCRIMINATORS.register("Dz")
class ZDiscriminator(nn.Module):
    def __init__(self, z_size, d=256):
        super(ZDiscriminator, self).__init__()

        class Block(nn.Module):
            def __init__(self, inputs, output, lrmul):
                super(Block, self).__init__()
                self.fc1 = ln.Linear(inputs, output, lrmul=lrmul)
                self.fc2 = ln.Linear(output, output, lrmul=lrmul)

            def forward(self, x):
                x = F.leaky_relu(self.fc1(x), 0.2)
                x = F.leaky_relu(self.fc2(x), 0.2)
                return x

        self.block1 = Block(z_size, d, 1.0)
        self.block2 = Block(d, d, 1.0)
        self.block3 = Block(d, d, 1.0)
        self.fc = ln.Linear(d * 3, 1, lrmul=1.0)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        return self.fc(torch.cat([x1, x2, x3], dim=1))
