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
from utils import utils
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


def save_sample(lod2batch, tracker, sample, samplez, x, logger, model, cfg, encoder_optimizer, decoder_optimizer, output_folder):
    os.makedirs('results', exist_ok=True)

    logger.info('\n[%d/%d] - ptime: %.2f, %s, lr: %.12f,  %.12f, max mem: %f",' % (
        (lod2batch.current_epoch + 1), cfg.TRAIN.TRAIN_EPOCHS, lod2batch.per_epoch_ptime, str(tracker),
        encoder_optimizer.param_groups[0]['lr'], decoder_optimizer.param_groups[0]['lr'],
        torch.cuda.max_memory_allocated() / 1024.0 / 1024.0))

    with torch.no_grad():
        model.eval()
        sample = sample[:lod2batch.get_per_GPU_batch_size()]
        samplez = samplez[:lod2batch.get_per_GPU_batch_size()]

        sample_in = sample

        Z, _ = model.encode(sample_in)

        rec1 = model.decoder(Z, noise=False)
        rec2 = model.decoder(Z, noise=True)

        Z = samplez
        g_rec = model.decoder(Z, noise=True)

        resultsample = torch.cat([sample_in, rec2, g_rec], dim=0)

        @utils.async_func
        def save_pic(x_rec):
            tracker.register_means(lod2batch.current_epoch + lod2batch.iteration * 1.0 / lod2batch.get_dataset_size())
            tracker.plot()

            result_sample = x_rec
            result_sample = result_sample.cpu()
            f = os.path.join(output_folder,
                             'sample_%d_%d.jpg' % (
                                 lod2batch.current_epoch + 1,
                                 lod2batch.iteration // 1000)
                             )
            print("Saved to %s" % f)
            save_image(result_sample, f, nrow=min(64, lod2batch.get_per_GPU_batch_size()))

        save_pic(resultsample)


def train(cfg, logger, local_rank, world_size, distributed):
    torch.cuda.set_device(local_rank)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        dlatent_avg_beta=cfg.MODEL.DLATENT_AVG_BETA,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
    )
    model.cuda(local_rank)
    model.train()

    model_s = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
    )
    model_s.cuda(local_rank)
    model_s.eval()
    model_s.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    z_discriminator = model.z_discriminator
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
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    z_discriminator_optimizer = LREQAdam([
        {'params': z_discriminator.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': encoder.parameters()},
        {'params': mapping_tl.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': decoder_optimizer,
                                    'z_discriminator_optimizer': z_discriminator_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

    model_dict = {
        'discriminator': encoder,
        'generator': decoder,
        'mapping_tl': mapping_tl,
        'z_discriminator': z_discriminator,
        'dlatent_avg': dlatent_avg,
        'discriminator_s': model_s.encoder,
        'generator_s': model_s.decoder,
        'mapping_tl_s': model_s.mapping_tl,
        'z_discriminator_s': model_s.z_discriminator,
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

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = decoder.layer_to_resolution

    train_set, _, _ = make_datasets(cfg, folding_id, inliner_classes)

    #dataset = TFRecordsDataset(cfg, logger, rank=local_rank, world_size=world_size, buffer_size_mb=1024, channels=cfg.MODEL.CHANNELS)

    rnd = np.random.RandomState(3456)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    samplez = torch.tensor(latents).float().cuda()

    lod2batch = driver.Driver(cfg, logger, world_size, dataset_size=len(train_set) * world_size)

    sample = next(make_dataloader(train_set, cfg.TRAIN.BATCH_SIZE, torch.cuda.current_device()))
    sample = sample[1]
    sample = sample.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
    # sample = (sample / 127.5 - 1.)

    lod2batch.set_epoch(scheduler.start_epoch(), [encoder_optimizer, decoder_optimizer])

    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [encoder_optimizer, decoder_optimizer])

        logger.info("Batch size: %d, Batch size per GPU: %d, dataset size: %d" % (
                                                                lod2batch.get_batch_size(),
                                                                lod2batch.get_per_GPU_batch_size(),
                                                                len(train_set) * world_size))

        data_loader = make_dataloader(train_set, lod2batch.get_per_GPU_batch_size(), torch.cuda.current_device())
        train_set.shuffle()

        scheduler.set_batch_size(lod2batch.get_batch_size())

        model.train()

        need_permute = False
        epoch_start_time = time.time()

        i = 0
        for y, x in data_loader:
            x = x.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)

            i += 1
            with torch.no_grad():
                if x.shape[0] != lod2batch.get_per_GPU_batch_size():
                    continue
                if need_permute:
                    x = x.permute(0, 3, 1, 2)

            encoder_optimizer.zero_grad()
            loss_d, loss_zg = model(x, d_train=True, ae=False)
            tracker.update(dict(loss_d=loss_d, loss_zg=loss_zg))
            (loss_zg + loss_d).backward()
            encoder_optimizer.step()

            decoder_optimizer.zero_grad()
            z_discriminator_optimizer.zero_grad()
            loss_g, loss_zd = model(x, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g, loss_zd=loss_zd))
            (loss_g + loss_zd).backward()
            decoder_optimizer.step()
            z_discriminator_optimizer.step()

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            lae = model(x, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            (lae).backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            betta = 0.5 ** (lod2batch.get_batch_size() / (1000.0))
            model_s.lerp(model, betta)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            lod2batch.step()
            if local_rank == 0:
                # if lod2batch.is_time_to_save():
                #     checkpointer.save("model_tmp_intermediate_lod%d" % lod_for_saving_model)
                if lod2batch.is_time_to_report():
                    save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s, cfg, encoder_optimizer,
                                decoder_optimizer, output_folder)

        scheduler.step()

        if local_rank == 0:
            checkpointer.save("model_tmp")
            save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s, cfg, encoder_optimizer, decoder_optimizer, output_folder)

    logger.info("Training finish!... save training results")
    if local_rank == 0:
        checkpointer.save("model_final").wait()


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='', default_config='configs/mnist.yaml',
        world_size=gpu_count)
