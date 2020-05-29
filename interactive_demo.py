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
from net import *
from model import Model
from launcher import run
from checkpointer import Checkpointer
from dlutils.pytorch import count_parameters
from defaults import get_cfg_defaults
from dataloader import *
import lreq
import dlutils

from PIL import Image
import bimpy


lreq.use_implicit_lreq.set(True)


def sample(cfg, logger):
    torch.cuda.set_device(0)

    folding_id = 0
    inliner_classes = [3]

    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    output_folder = os.path.join(cfg.OUTPUT_DIR, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER
    )
    model.cuda(0)
    model.eval()
    model.requires_grad_(False)

    decoder = model.decoder
    encoder = model.encoder
    mapping_tl = model.mapping_tl
    mapping_fl = model.mapping_fl
    dlatent_avg = model.dlatent_avg

    logger.info("Trainable parameters generator:")
    count_parameters(decoder)

    logger.info("Trainable parameters discriminator:")
    count_parameters(encoder)

    arguments = dict()
    arguments["iteration"] = 0

    model_dict = {
        'discriminator_s': encoder,
        'generator_s': decoder,
        'mapping_tl_s': mapping_tl,
        'mapping_fl_s': mapping_fl,
        'dlatent_avg': dlatent_avg
    }

    checkpointer = Checkpointer(output_folder,
                                model_dict,
                                {},
                                logger=logger,
                                save=False)

    extra_checkpoint_data = checkpointer.load()

    model.eval()

    layer_count = cfg.MODEL.LAYER_COUNT

    def encode(x):
        w, _ = model.encode(x)
        return w

    def decode(x):
        return model.decoder(x, noise=True)

    def decode_p(x):
        x = model.decoder.decode_block[0](x, True)
        # orig_x = x
        # orig_shape = x.shape
        # x = x.view(*x.shape[:2], -1)
        # x = x.cpu().numpy()
        # x = np.take(x, np.random.permutation(x.shape[2]), axis=2)
        # x = torch.tensor(x).cuda().view(*orig_shape)
        #
        # x = x * 0.5 + orig_x * 0.5

        axis = np.random.randint(2, 3)
        i = np.random.randint(0, x.shape[axis])

        x = x.cpu().numpy()
        x = np.take(x, np.random.permutation(x.shape[2]), axis=2)
        x = np.take(x, np.random.permutation(x.shape[3]), axis=3)
        x = np.take(x, np.random.permutation(x.shape[2]), axis=2)
        x = np.take(x, np.random.permutation(x.shape[3]), axis=3)
        x = torch.tensor(x).cuda()

        # x = torch.cat((x[:, :, :, x.shape[3]//2:], x[:, :, :, :x.shape[3]//2]), dim=3)

        for i in range(1, model.decoder.layer_count):
            x = model.decoder.decode_block[i](x, True)
        x = model.decoder.to_rgb(x)
        return x

    train_set, valid_set, test_set = make_datasets(cfg, folding_id, inliner_classes)

    sample = next(make_dataloader(valid_set, cfg.TRAIN.BATCH_SIZE, torch.cuda.current_device()))
    sample = sample[1]
    sample = sample.view(-1, cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)

    randomize = bimpy.Bool(True)

    ctx = bimpy.Context()

    rnd = np.random.RandomState(5)
    current_sample = [0]

    def loadNext():
        x = sample[current_sample[0]]
        current_sample[0] += 1

        img_src = (x * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()

        latents_original = encode(x[None, ...].cuda())
        latents = latents_original[0].clone()
        latents -= model.dlatent_avg.buff.data
        return latents, img_src

    def loadRandom():
        latents = rnd.randn(1, cfg.MODEL.LATENT_SPACE_SIZE)
        lat = torch.tensor(latents).float().cuda()
        dlat = torch.tensor((rnd.rand(1, cfg.MODEL.LATENT_SPACE_SIZE) * 4 - 1)).float().cuda() # = mapping_fl(lat)
        dlat = mapping_fl(lat)
        x = decode_p(dlat)[0]
        img_src = (x * 255).type(torch.long).clamp(0, 255).cpu().type(torch.uint8).transpose(0, 2).transpose(0, 1).numpy()
        latents_original = dlat
        latents = latents_original[0].clone()
        latents -= model.dlatent_avg.buff.data

        return latents, img_src

    latents, img_src = loadNext()

    ctx.init(1800, 1600, "Styles")

    def update_image(w):
        with torch.no_grad():
            w = w + model.dlatent_avg.buff.data
            w = w[None, ...]
            x_rec = decode(w)
            resultsample = (x_rec * 255).type(torch.long).clamp(0, 255)
            resultsample = resultsample.cpu()[0, :, :, :]
            return resultsample.type(torch.uint8).transpose(0, 2).transpose(0, 1)

    im_size = 2 ** (cfg.MODEL.LAYER_COUNT + 1)
    im = update_image(latents)
    print(im.shape)
    im = bimpy.Image(im)

    display_original = True

    seed = 0

    while not ctx.should_close():
        with ctx:
            new_latents = latents

            if display_original:
                im = bimpy.Image(np.concatenate([img_src, img_src, img_src], axis=2))
            else:
                im = update_image(new_latents)
                im = bimpy.Image(np.concatenate([im, im, im], axis=2))

            bimpy.begin("Principal directions")
            bimpy.columns(2)
            bimpy.set_column_width(0, im_size + 20)
            bimpy.image(im)
            bimpy.next_column()

            # bimpy.slider_float(label, v, -40.0, 40.0)

            bimpy.checkbox("Randomize noise", randomize)

            if randomize.value:
                seed += 1

            torch.manual_seed(seed)

            if bimpy.button('Next'):
                latents, img_src = loadNext()
                display_original = True
            if bimpy.button('Display Reconstruction'):
                display_original = False
            if bimpy.button('Generate random'):
                latents, img_src = loadRandom()
                display_original = True

            bimpy.end()


if __name__ == "__main__":
    gpu_count = 1
    run(sample, get_cfg_defaults(), description='ALAE-interactive', default_config='configs/mnist.yaml',
        world_size=gpu_count, write_log=False)
