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


def get_mnist(test):
    dlutils.download.mnist()
    mnist = dlutils.reader.Mnist('mnist', train=not test, test=test).items

    images = [x[1] for x in mnist]
    labels = [x[0] for x in mnist]

    images = np.asarray(images)

    _images = []
    for im in images:
        im = Image.fromarray(im)
        im = np.array(im.resize((32, 32), Image.BILINEAR))
        # im = misc.imresize(im, (32, 32), interp='bilinear')
        _images.append(im)
    images = np.asarray(_images)

    return [(l, im) for l, im in zip(labels, images)]


def partition(cfg, logger):
    # to reproduce the same shuffle
    random.seed(0)
    mnist_train = get_mnist(test=False)
    mnist_test = get_mnist(test=True)

    random.shuffle(mnist_train)
    random.shuffle(mnist_test)
    valid_size = int(len(mnist_train) * 0.15)
    mnist_valid = mnist_train[-valid_size:]
    mnist_train = mnist_train[:-valid_size]

    print("Train count: %d" % len(mnist_train))
    print("Valid count: %d" % len(mnist_valid))
    print("Test count: %d" % len(mnist_test))

    os.makedirs(os.path.dirname(cfg.DATASET.PATH), exist_ok=True)

    with open(cfg.DATASET.PATH % "valid", 'wb') as f:
        pickle.dump(mnist_valid, f)

    with open(cfg.DATASET.PATH % "train", 'wb') as f:
        pickle.dump(mnist_train, f)

    with open(cfg.DATASET.PATH % "test", 'wb') as f:
        pickle.dump(mnist_test, f)


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/mnist_os.yaml')
    cfg.freeze()
    logger = logging.getLogger("logger")
    partition(cfg, logger)
