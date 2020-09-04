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
from net import Wclassifier
from custom_adam import LREQAdam
import losses


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

    classifier = Wclassifier(latent_size=32, dlatent_size=1, mapping_fmaps=256)

    optimizer = LREQAdam([
        {'params': classifier.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    tracker = LossTracker(cfg.OUTPUT_DIR)
    rnd = np.random.RandomState(5)

    for it in range(30):
        for i in range(1000):
            z = torch.randn(64, cfg.MODEL.LATENT_SPACE_SIZE)
            w = model_s.mapping_fl(z)

            w_fake = torch.randn(512, cfg.MODEL.LATENT_SPACE_SIZE)
            w_fake2 = torch.tensor((rnd.rand(512, cfg.MODEL.LATENT_SPACE_SIZE) * 4 - 1)).float().cuda()

            d_result_real = classifier(w)
            d_result_fake = torch.cat((classifier(w_fake), classifier(w_fake2)), dim=0)

            optimizer.zero_grad()
            loss_d = losses.discriminator_classic(d_result_fake, d_result_real)
            loss_d.backward()
            optimizer.step()
            tracker.update(dict(loss_d=loss_d))

        logger.info('\n%s, lr: %.12f' % (str(tracker),
            optimizer.param_groups[0]['lr']))

        tracker.register_means(it)
        tracker.plot()

    torch.save(classifier.state_dict(), os.path.join(output_folder, "wclassifier.pkl"))


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='', default_config='configs/mnist.yaml',
        world_size=1, folding_id=0, inliner_classes=[3])
