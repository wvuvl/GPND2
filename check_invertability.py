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
from torchvision.utils import save_image
from net import *
import os
import utils
from checkpointer import Checkpointer
from scheduler import ComboMultiStepLR
from custom_adam import LREQAdam
from dataloader import *
from tqdm import tqdm
from dlutils.pytorch import count_parameters
import dlutils.pytorch.count_parameters as count_param_override
from tracker import LossTracker
from model import Model
from launcher import run
from defaults import get_cfg_defaults
import driver
import time
from PIL import Image
from math import log10


def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        z_regression=cfg.MODEL.Z_REGRESSION
    )
    model.cuda(local_rank)
    model.train()

    model_s = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
        z_regression=cfg.MODEL.Z_REGRESSION)
    model_s.cuda(local_rank)
    model_s.eval()
    model_s.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    decoder_optimizer = LREQAdam([
        {'params': decoder.parameters()},
        {'params': mapping_fl.parameters()}
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': encoder.parameters()},
        {'params': mapping_tl.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_tl,
        'mapping_fl': mapping_fl,
        'dlatent_avg': dlatent_avg,
        'discriminator_s': model_s.encoder,
        'generator_s': model_s.decoder,
        'mapping_tl_s': model_s.mapping_tl,
        'mapping_fl_s': model_s.mapping_fl
    }

    folding_id = 0
    inliner_classes = [3]

    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    output_folder = os.path.join(cfg.OUTPUT_DIR, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    tracker = LossTracker(output_folder)

    checkpointer = Checkpointer(output_folder,
                                model_dict,
                                {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer,
                                    'scheduler': scheduler,
                                    'tracker': tracker
                                },
                                logger=logger,
                                save=local_rank == 0)

    extra_checkpoint_data = checkpointer.load()
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

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
        world_size=gpu_count)
