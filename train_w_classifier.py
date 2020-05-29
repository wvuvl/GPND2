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

import torch.utils.data
import os
from checkpointer import Checkpointer
from dataloader import *
from tracker import LossTracker
from model import Model
from launcher import run
from defaults import get_cfg_defaults


def train(cfg, logger, local_rank, folding_id, inliner_classes):
    torch.cuda.set_device(local_rank)
    model_s = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
    model_s.cuda(local_rank)
    model_s.eval()
    model_s.requires_grad_(False)

    model_dict = {
        'discriminator_s': model_s.encoder,
        'generator_s': model_s.decoder,
        'mapping_tl_s': model_s.mapping_tl,
        'mapping_fl_s': model_s.mapping_fl
    }

    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    output_folder = os.path.join(cfg.OUTPUT_DIR, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    checkpointer = Checkpointer(output_folder,
                                model_dict,
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()
    last_epoch = list(extra_checkpoint_data['auxiliary']['scheduler'].values())[0]['last_epoch']
    logger.info("Model trained for %d epochs" % last_epoch)

    model_s.mapping_fl.compute_inverse()

    ###############################
    # Check
    ##############################
    z = torch.randn(1000, cfg.MODEL.LATENT_SPACE_SIZE)
    w = model_s.mapping_fl(z)
    _z = model_s.mapping_fl.reverse(w)

    criterion = torch.nn.MSELoss()

    mse = ((z - _z) ** 2).mean(dim=1)

    avg_psnr = (10 * torch.log10(1.0 / mse)).mean()

    print('===> MSE:  {:.8f}'.format(mse.mean()))
    print('===> Avg. PSNR: {:.8f} dB'.format(avg_psnr))


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='', default_config='configs/mnist.yaml',
        world_size=1, folding_id=0, inliner_classes=[3])
