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


class DLatent(nn.Module):
    def __init__(self, dlatent_size):
        super(DLatent, self).__init__()
        buffer = torch.zeros(dlatent_size, dtype=torch.float32)
        self.register_buffer('buff', buffer)


class Model(nn.Module):
    def __init__(self, startf=32, maxf=256, layer_count=3, latent_size=128, mapping_layers=5, dlatent_avg_beta=None,
                 channels=3, generator="", encoder=""):
        super(Model, self).__init__()

        self.layer_count = layer_count

        self.mapping_tl = MAPPINGS["MappingToLatent"](
            latent_size=latent_size,
            dlatent_size=latent_size,
            mapping_fmaps=latent_size,
            mapping_layers=3)

        self.decoder = GENERATORS[generator](
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

        self.z_discriminator = ZDiscriminator(latent_size)

        self.dlatent_avg = DLatent(latent_size)
        self.latent_size = latent_size
        self.dlatent_avg_beta = dlatent_avg_beta

    def generate(self, z=None, count=32, noise=True, return_w=False):
        if z is None:
            z = torch.randn(count, self.latent_size)
        w = z  # self.mapping_fl(z)

        if self.dlatent_avg_beta is not None:
            with torch.no_grad():
                batch_avg = w.mean(dim=0)
                self.dlatent_avg.buff.data.lerp_(batch_avg.data, 1.0 - self.dlatent_avg_beta)

        rec = self.decoder.forward(w, noise)
        if return_w:
            return w, rec
        else:
            return rec

    def encode(self, x):
        Z = self.encoder(x)
        Z_ = self.mapping_tl(Z)
        return Z, Z_[:, 0]

    def forward(self, x, d_train, ae):
        if ae:
            self.encoder.requires_grad_(True)

            z = torch.randn(x.shape[0], self.latent_size)
            w, rec = self.generate(z=z, noise=True, return_w=True)

            Z, d_result_real = self.encode(rec)

            assert Z.shape == w.shape

            Lae = torch.mean(((Z - w.detach())**2))

            return Lae

        elif d_train:
            with torch.no_grad():
                Xp = self.generate(count=x.shape[0], noise=True)

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

            rec = self.generate(count=x.shape[0], z=z.detach(), noise=True)

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
            params = list(self.mapping_tl.parameters()) + list(self.decoder.parameters()) + list(self.encoder.parameters()) + list(self.dlatent_avg.parameters())
            other_param = list(other.mapping_tl.parameters()) + list(other.decoder.parameters()) + list(other.encoder.parameters()) + list(other.dlatent_avg.parameters())
            for p, p_other in zip(params, other_param):
                p.data.lerp_(p_other.data, 1.0 - betta)
