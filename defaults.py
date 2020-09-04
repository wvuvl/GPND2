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

from yacs.config import CfgNode as CN


_C = CN()

_C.DATASET = CN()
_C.DATASET.FLIP_IMAGES = True

_C.DATASET.PERCENTAGES = [10, 20, 30, 40, 50]

_C.DATASET.MAX_RESOLUTION_LEVEL = 10
# Values for MNIST
_C.DATASET.MEAN = 0.1307
_C.DATASET.STD = 0.3081

_C.DATASET.PATH = "mnist"
_C.DATASET.TOTAL_CLASS_COUNT = 10
_C.DATASET.FOLDS_COUNT = 5

_C.MODEL = CN()
_C.MODEL.LATENT_SIZE = 32
_C.MODEL.INPUT_IMAGE_SIZE = 32
_C.MODEL.INPUT_IMAGE_CHANNELS = 1
# If zd_merge true, will use zd discriminator that looks at entire batch.
_C.MODEL.Z_DISCRIMINATOR_CROSS_BATCH = False

_C.MODEL.LAYER_COUNT = 6
_C.MODEL.START_CHANNEL_COUNT = 64
_C.MODEL.MAX_CHANNEL_COUNT = 512
_C.MODEL.LATENT_SPACE_SIZE = 256
_C.MODEL.DLATENT_AVG_BETA = 0.995
_C.MODEL.TRUNCATIOM_PSI = 0.7
_C.MODEL.TRUNCATIOM_CUTOFF = 8
_C.MODEL.STYLE_MIXING_PROB = 0.9
_C.MODEL.MAPPING_LAYERS = 5
_C.MODEL.CHANNELS = 3
_C.MODEL.GENERATOR = "GeneratorDefault"
_C.MODEL.ENCODER = "EncoderDefault"
_C.MODEL.Z_REGRESSION = False

_C.TRAIN = CN()

_C.TRAIN.BATCH_SIZE = 256
_C.TRAIN.TRAIN_EPOCHS = 80
_C.TRAIN.BASE_LEARNING_RATE = 0.0015
_C.TRAIN.ADAM_BETA_0 = 0.0
_C.TRAIN.ADAM_BETA_1 = 0.99
_C.TRAIN.LEARNING_DECAY_RATE = 0.1
_C.TRAIN.LEARNING_DECAY_STEPS = []

_C.TRAIN.BATCH_1GPU = 256

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 256

_C.MAKE_PLOTS = True

_C.TRAIN.SNAPSHOT_FREQ = 300

_C.TRAIN.REPORT_FREQ = 100

_C.TRAIN.LEARNING_RATES = 0.002

_C.OUTPUT_DIR = 'results'
_C.RESULTS_NAME = 'results.csv'


def get_cfg_defaults():
    return _C.clone()
