# Copyright 2018-2020 Stanislav Pidhorskyi
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

import dlutils
import random
import pickle
from defaults import get_cfg_defaults
import numpy as np
from os import path
from scipy import misc
import logging
from PIL import Image
import os


def get_cifar10(test):
    dlutils.download.cifar10()
    mnist = dlutils.reader.Cifar10('cifar10/cifar-10-batches-bin', train=not test, test=test).items

    images = [x[1] for x in mnist]
    labels = [x[0] for x in mnist]

    images = np.asarray(images)

    return [(l, im) for l, im in zip(labels, images)]


def partition(cfg, logger):
    # to reproduce the same shuffle
    random.seed(0)
    data_train = get_cifar10(test=False)
    data_test = get_cifar10(test=True)

    random.shuffle(data_train)
    random.shuffle(data_test)
    valid_size = int(len(data_train) * 0.15)
    data_valid = data_train[-valid_size:]
    data_train = data_train[:-valid_size]

    print("Train count: %d" % len(data_train))
    print("Valid count: %d" % len(data_valid))
    print("Test count: %d" % len(data_test))

    os.makedirs(os.path.dirname(cfg.DATASET.PATH), exist_ok=True)

    with open(cfg.DATASET.PATH % "valid", 'wb') as f:
        pickle.dump(data_valid, f)

    with open(cfg.DATASET.PATH % "train", 'wb') as f:
        pickle.dump(data_train, f)

    with open(cfg.DATASET.PATH % "test", 'wb') as f:
        pickle.dump(data_test, f)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/cifar10_os.yaml')
    cfg.freeze()
    logger = logging.getLogger("logger")
    partition(cfg, logger)
