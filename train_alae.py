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
from novelty_detector import eval_model_on_valid
import operator


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
        rec2 = model.generator(Z, noise=True)

        g_rec = model.generator(samplez, noise=True)

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


def train(cfg, logger, local_rank, world_size, folding_id=0, inliner_classes=[3]):
    torch.cuda.set_device(local_rank)
    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
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
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
    )
    model_s.cuda(local_rank)
    model_s.eval()
    model_s.requires_grad_(False)

    generator = model.generator
    encoder = model.encoder
    discriminator = model.discriminator
    z_discriminator = model.z_discriminator

    count_param_override.print = lambda a: logger.info(a)

    logger.info("Trainable parameters generator:")
    count_parameters(generator)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    generator_optimizer = LREQAdam([
        {'params': generator.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    z_discriminator_optimizer = LREQAdam([
        {'params': z_discriminator.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    encoder_optimizer = LREQAdam([
        {'params': encoder.parameters()},
        {'params': discriminator.parameters()},
    ], lr=cfg.TRAIN.BASE_LEARNING_RATE, betas=(cfg.TRAIN.ADAM_BETA_0, cfg.TRAIN.ADAM_BETA_1), weight_decay=0)

    scheduler = ComboMultiStepLR(optimizers=
                                 {
                                    'encoder_optimizer': encoder_optimizer,
                                    'generator_optimizer': generator_optimizer,
                                    'z_discriminator_optimizer': z_discriminator_optimizer
                                 },
                                 milestones=cfg.TRAIN.LEARNING_DECAY_STEPS,
                                 gamma=cfg.TRAIN.LEARNING_DECAY_RATE,
                                 reference_batch_size=32, base_lr=cfg.TRAIN.LEARNING_RATES)

    model_dict = {
        'encoder': encoder,
        'generator': generator,
        'discriminator': discriminator,
        'z_discriminator': z_discriminator,
        'encoder_s': model_s.encoder,
        'generator_s': model_s.generator,
        'discriminator_s': model_s.discriminator,
        'z_discriminator_s': model_s.z_discriminator,
    }

    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    output_folder = os.path.join(cfg.OUTPUT_DIR, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    tracker = LossTracker(output_folder)

    checkpointer = Checkpointer(output_folder,
                                model_dict,
                                {
                                    'encoder_optimizer': encoder_optimizer,
                                    'decoder_optimizer': generator_optimizer,
                                    'scheduler': scheduler,
                                    'tracker': tracker
                                },
                                logger=logger,
                                save=True)

    extra_checkpoint_data = checkpointer.load()
    logger.info("Starting from epoch: %d" % (scheduler.start_epoch()))

    arguments.update(extra_checkpoint_data)

    layer_to_resolution = generator.layer_to_resolution

    train_set, _, _ = make_datasets(cfg, folding_id, inliner_classes)

    rnd = np.random.RandomState(3456)
    latents = rnd.randn(32, cfg.MODEL.LATENT_SPACE_SIZE)
    samplez = torch.tensor(latents).float().cuda()

    lod2batch = driver.Driver(cfg, logger, world_size, dataset_size=len(train_set) * world_size)

    sample = next(make_dataloader(train_set, cfg.TRAIN.BATCH_SIZE, torch.cuda.current_device()))
    sample = sample[1]
    sample = sample.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
    # sample = (sample / 127.5 - 1.)

    lod2batch.set_epoch(scheduler.start_epoch(), [encoder_optimizer, generator_optimizer])

    scores_list = []

    try:
        with open(os.path.join(output_folder, "scores.txt"), "r") as f:
            lines = f.readlines()
            lines = [l[:-1].strip() for l in lines]
            lines = [l.split(' ') for l in lines]
            lines = [l for l in lines if len(l) == 2]
            scores_list = [(x[0], float(x[1]))for x in lines]
            # for l in scores_list:
            #     print("%s: %f" % l)
    except FileNotFoundError:
        pass

    def save(epoch):
        score = eval_model_on_valid(cfg, logger, model_s, folding_id, inliner_classes)
        filename = "model_%d" % epoch
        checkpointer.save(filename).wait()
        scores_list.append((filename, score))
        with open(os.path.join(output_folder, "scores.txt"), "w") as f:
            f.writelines([x[0] + " " + str(x[1]) + "\n" for x in scores_list])

    def last_score():
        return 0 if len(scores_list) == 0 else scores_list[-1][1]

    epoch = None
    for epoch in range(scheduler.start_epoch(), cfg.TRAIN.TRAIN_EPOCHS):
        model.train()
        lod2batch.set_epoch(epoch, [encoder_optimizer, generator_optimizer])

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

            generator_optimizer.zero_grad()
            z_discriminator_optimizer.zero_grad()
            loss_g, loss_zd = model(x, d_train=False, ae=False)
            tracker.update(dict(loss_g=loss_g, loss_zd=loss_zd))
            (loss_g + loss_zd).backward()
            generator_optimizer.step()
            z_discriminator_optimizer.step()

            encoder_optimizer.zero_grad()
            generator_optimizer.zero_grad()
            lae = model(x, d_train=True, ae=True)
            tracker.update(dict(lae=lae))
            (lae).backward()
            encoder_optimizer.step()
            generator_optimizer.step()

            betta = 0.5 ** (lod2batch.get_batch_size() / (1000.0))
            model_s.lerp(model, betta)

            epoch_end_time = time.time()
            per_epoch_ptime = epoch_end_time - epoch_start_time

            # tracker.update(dict(score_a=score_a, score_b=score_b, score_c=score_c))
            tracker.update(dict(score=last_score()))

            lod2batch.step()
            # if lod2batch.is_time_to_save():
            #     checkpointer.save("model_tmp_intermediate_lod%d" % lod_for_saving_model)
            if lod2batch.is_time_to_report():
                save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s, cfg, encoder_optimizer,
                            generator_optimizer, output_folder)

        scheduler.step()

        if epoch % 20 == 0:
            save(epoch)

        save_sample(lod2batch, tracker, sample, samplez, x, logger, model_s, cfg, encoder_optimizer, generator_optimizer, output_folder)

    logger.info("Training finish!... save training results")
    if epoch is not None:
        save(epoch)
    best_model_name, best_model_score = max(scores_list, key=operator.itemgetter(1))
    checkpointer.tag_best_checkpoint(best_model_name)


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    run(train, get_cfg_defaults(), description='', default_config='configs/mnist.yaml',
        world_size=gpu_count)
