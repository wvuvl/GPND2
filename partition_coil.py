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

import random
import pickle
from defaults import get_cfg_defaults
import imageio
import os
import re
import numpy as np
import logging
from PIL import Image


def partition(cfg, logger):
    regexp = r"obj(\d*)__(\d*).png"

    coil = []

    walk_path = os.path.join(os.path.dirname(cfg.DATASET.PATH), "coil-100")
    for root, dirs, files in os.walk(walk_path):
        root_ = os.path.basename(root)
        for f in files:
            try:
                m = re.match(regexp, f)
                if m is None:
                    continue
                print(f)
                id = m.group(1)
                pic = m.group(2)
                print(id)
                image = imageio.imread(os.path.join(root, f))
                image = Image.fromarray(image)
                image = np.array(image.resize((32, 32), Image.BILINEAR))
                image = np.transpose(image, (2, 0, 1))

                coil.append((int(id), image))
            except Exception as e:
                raise e

    folds = cfg.DATASET.FOLDS_COUNT

    # Split coil into 5 folds:
    class_bins = {}

    random.shuffle(coil)

    for x in coil:
        if x[0] not in class_bins:
            class_bins[x[0]] = []
        class_bins[x[0]].append(x)

    coil_folds = [[] for _ in range(folds)]

    for _class, data in class_bins.items():
        count = len(data)
        print("Class %d count: %d" % (_class, count))

        count_per_fold = count // folds

        for i in range(folds):
            coil_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]

    print("Folds sizes:")
    for i in range(len(coil_folds)):
        print(len(coil_folds[i]))

        output = open(cfg.DATASET.PATH % i, 'wb')
        pickle.dump(coil_folds[i], output)
        output.close()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/coil100.yaml')
    cfg.freeze()
    logger = logging.getLogger("logger")
    partition(cfg, logger)
