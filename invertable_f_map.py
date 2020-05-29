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

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import lreq as ln
from gram_schmidt import gram_schmidt
from registry import *


class MappingBlock(nn.Module):
    def __init__(self, inputs, output, lrmul):
        super(MappingBlock, self).__init__()
        with torch.no_grad():
            self.fc = ln.Linear(inputs, output, lrmul=lrmul)
            self.fc.weight.data = self.fc.weight.data.double()
            # print("Before", torch.norm(self.fc.weight))
            self.fc.weight.data = self.fc.weight.data * 0.9 + gram_schmidt(self.fc.weight.data) * 0.1
            # print("After", torch.norm(self.fc.weight))
            self.fc.bias.data = self.fc.bias.data.double()
            self.i_fc = ln.Linear(output, inputs, lrmul=lrmul)
            self.last_activation = None
            self.alpha = 0.2

    def compute_inverse(self):
        with torch.no_grad():
            self.i_fc.weight.data = torch.inverse(self.fc.weight)
            self.i_fc.bias.data = -torch.matmul(self.i_fc.weight, self.fc.bias)

    def forward(self, x):
        x = self.fc(x)
        self.last_activation = (x > 0).detach()
        x = F.leaky_relu(x, self.alpha)  # max(self.fc(x), self.alpha* self.fc(x))
        return x

    def reverse(self, x):
        x = F.leaky_relu(x, 1.0 / self.alpha)  # min(x, x / self.alpha)
        self.last_activation = (x > 0).detach()
        x = self.i_fc(x)
        return x

    def jacobian(self):
        h_p = (self.alpha + (1.0 - self.alpha) * self.last_activation)
        return h_p.double()[..., None] * self.fc.weight


@MAPPINGS.register("Invertable_F_Map")
class Mapping(nn.Module):
    def __init__(self, mapping_layers=5, latent_size=256, dlatent_size=None, mapping_fmaps=None):
        super(Mapping, self).__init__()
        dlatent_size = dlatent_size if dlatent_size else latent_size
        mapping_fmaps = mapping_fmaps if mapping_fmaps else latent_size
        inputs = latent_size
        self.mapping_layers = mapping_layers
        self.blocks: nn.ModuleList[MappingBlock] = nn.ModuleList()

        for i in range(mapping_layers):
            outputs = dlatent_size if i == mapping_layers - 1 else mapping_fmaps
            block = MappingBlock(inputs, outputs, lrmul=0.1)
            inputs = outputs
            self.blocks.append(block)

    def compute_inverse(self):
        for i in range(self.mapping_layers):
            self.blocks[i].compute_inverse()

    def forward(self, x):
        x = x.double()
        for i in range(self.mapping_layers):
            x = self.blocks[i](x)
        return x.float()

    def reverse(self, x):
        x = x.double()
        for i in range(self.mapping_layers):
            x = self.blocks[self.mapping_layers - 1 - i].reverse(x)
        return x.float()

    def jacobian(self):
        j = self.blocks[0].jacobian()
        for i in range(1, self.mapping_layers):
            j = torch.matmul(self.blocks[i].jacobian(), j)
        return j


if __name__ == "__main__":
    def test_mapping_block():
        b = MappingBlock(32, 32, 0.01)
        b.compute_inverse()

        x = torch.randn(1000, 32).double()

        r = b(x)

        _x = b.reverse(r)

        # print(x)
        # print(r)
        # print(_x)
        # print(x - _x)

        print(torch.norm(x))
        print(torch.norm(_x - x))

        mse = ((x - _x) ** 2).mean(dim=1)
        avg_psnr = (10 * torch.log10(1.0 / mse)).mean()
        print('===> MSE:  {:.8f}'.format(mse.mean()))
        print('===> Avg. PSNR: {:.8f} dB'.format(avg_psnr))


    def test_f_map():
        b = Mapping(4, 32)
        b.compute_inverse()

        x = torch.randn(1000, 32)
        m = torch.distributions.Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        logp = m.log_prob(x)
        logPz_orig = np.sum(logp.cpu().numpy())
        r = b(x)

        _x = b.reverse(r)

        logp = m.log_prob(_x)
        logPz_rec = np.sum(logp.cpu().numpy())

        # print(x)
        # print(r)
        # print(_x)
        # print(x - _x)

        print(torch.norm(x))
        print(torch.norm(_x - x))

        mse = ((x - _x) ** 2).mean(dim=1)
        avg_psnr = (10 * torch.log10(1.0 / mse)).mean()
        print('===> MSE:  {:.8f}'.format(mse.mean()))
        print('===> Avg. PSNR: {:.8f} dB'.format(avg_psnr))

        print(logPz_orig)
        print(logPz_rec)


    def test_jacobian():
        def compute_jacobian_using_finite_differences(input, func, epsilon=1e-5):
            with torch.no_grad():
                input_size = np.prod(input.shape[1:]).item()
                e = torch.eye(input_size, dtype=input.dtype).view(input_size, 1, input_size)
                input_ = input.view(1, input.shape[0], input_size)

                input2 = torch.stack([input_ + e * epsilon, input_ - e * epsilon])

                y = func(input2.reshape([input.shape[0] * input_size * 2] + list(input.shape[1:])))

                output_size = np.prod(y.shape[1:]).item()

                J = torch.zeros(input_size, input.shape[0], output_size, requires_grad=False)
                J += y[:input_size * input.shape[0]].view(input_size, input.shape[0], output_size)
                J -= y[input_size * input.shape[0]:].view(input_size, input.shape[0], output_size)
                J /= 2.0 * epsilon

                J = torch.transpose(J, dim0=0, dim1=1)
                J = torch.transpose(J, dim0=1, dim1=2)

                return J

        b = Mapping(4, 32)
        b.compute_inverse()

        x = torch.randn(1000, 32)
        x = x.double()

        r = b(x)

        _x = b.reverse(r)

        j = b.jacobian()

        j_numeracal = compute_jacobian_using_finite_differences(x, b)

        # print(j)
        # print(j_numeracal)
        print(torch.norm(j_numeracal))
        print(torch.norm(j_numeracal - j))

        mse = ((j_numeracal - j) ** 2).mean(dim=1)
        avg_psnr = (10 * torch.log10(1.0 / mse)).mean()
        print('===> MSE:  {:.8f}'.format(mse.mean()))
        print('===> Avg. PSNR: {:.8f} dB'.format(avg_psnr))

    with torch.no_grad():
        test_mapping_block()
        test_f_map()
        test_jacobian()
