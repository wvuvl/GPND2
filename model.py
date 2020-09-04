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

import random
import losses
from net import *
from invertable_f_map import *
import numpy as np


class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, channels=3, generator="", encoder=""):
        super(Model, self).__init__()

        self.layer_count = layer_count

        self.discriminator = DISCRIMINATORS["D"](
            net_inputs=latent_size,
            hidden_size=latent_size,
            net_outputs=1,
            mapping_layers=1)

        self.generator = GENERATORS[generator](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.encoder = ENCODERS[encoder](
            startf=startf,
            layer_count=layer_count,
            maxf=maxf,
            latent_size=latent_size,
            channels=channels)

        self.z_discriminator = DISCRIMINATORS["Dz"](latent_size)

        self.latent_size = latent_size

    def generate(self, z=None, count=32, noise=True):
        if z is None:
            z = torch.randn(count, self.latent_size)

        rec = self.generator.forward(z, noise)
        return z, rec

    def encode(self, x):
        z = self.encoder(x)
        d = self.discriminator(z)
        return z, d[:, 0]

    def forward(self, x, d_train, ae):
        if ae:
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            _, rec = self.generate(z=z, noise=True)

            z_rec, d_result_real = self.encode(rec)

            lae = torch.mean(((z_rec - z.detach())**2))

            return lae

        elif d_train:
            with torch.no_grad():
                _, Xp = self.generate(count=x.shape[0], noise=True)

            self.encoder.requires_grad_(True)
            self.z_discriminator.requires_grad_(False)

            z1, d_result_real = self.encode(x)

            z2, d_result_fake = self.encode(Xp.detach())

            loss_d = losses.discriminator_logistic_simple_gp(d_result_fake, d_result_real, x)

            zd_result_fake = self.z_discriminator(z1)

            loss_zg = losses.generator_logistic_non_saturating(zd_result_fake)
            return loss_d, loss_zg
        else:
            with torch.no_grad():
                z = torch.randn(x.shape[0], self.latent_size)

            self.encoder.requires_grad_(False)
            self.z_discriminator.requires_grad_(True)

            _, rec = self.generate(count=x.shape[0], z=z.detach(), noise=True)

            z_fake, d_result_fake = self.encode(rec)

            zd_result_fake = self.z_discriminator(z_fake.detach())
            z_real = torch.randn(x.shape[0], self.latent_size).requires_grad_(True)
            zd_result_real = self.z_discriminator(z_real)

            loss_g = losses.generator_logistic_non_saturating(d_result_fake)
            loss_zd = losses.discriminator_logistic_simple_gp(zd_result_fake, zd_result_real, z_real)

            return loss_g, loss_zd

    def lerp(self, other, betta):
        if hasattr(other, 'module'):
            other = other.module
        with torch.no_grad():
            params = list(self.discriminator.parameters()) + list(self.generator.parameters()) + list(self.encoder.parameters())
            other_param = list(other.discriminator.parameters()) + list(other.generator.parameters()) + list(other.encoder.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
