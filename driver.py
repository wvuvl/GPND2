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

import torch
import math
import time
from collections import defaultdict


class Driver:
    def __init__(self, cfg, logger, world_size, dataset_size):
        if world_size == 8:
            self.batch = cfg.TRAIN.BATCH_8GPU
        if world_size == 4:
            self.batch = cfg.TRAIN.BATCH_4GPU
        if world_size == 2:
            self.batch = cfg.TRAIN.BATCH_2GPU
        if world_size == 1:
            self.batch = cfg.TRAIN.BATCH_1GPU

        self.world_size = world_size
        self.minibatch_base = 16
        self.cfg = cfg
        self.dataset_size = dataset_size
        self.current_epoch = 0
        self.logger = logger
        self.iteration = 0
        self.epoch_end_time = 0
        self.epoch_start_time = 0
        self.per_epoch_ptime = 0
        self.reports = cfg.TRAIN.REPORT_FREQ
        self.snapshots = cfg.TRAIN.SNAPSHOT_FREQ
        self.tick_start_nimg_report = 0
        self.tick_start_nimg_snapshot = 0

    def get_batch_size(self):
        return self.batch

    def get_dataset_size(self):
        return self.dataset_size

    def get_per_GPU_batch_size(self):
        return self.get_batch_size() // self.world_size

    def is_time_to_report(self):
        if self.iteration >= self.tick_start_nimg_report + self.reports * 1000:
            self.tick_start_nimg_report = self.iteration
            return True
        return False

    def is_time_to_save(self):
        if self.iteration >= self.tick_start_nimg_snapshot + self.snapshots * 1000:
            self.tick_start_nimg_snapshot = self.iteration
            return True
        return False

    def step(self):
        self.iteration += self.get_batch_size()
        self.epoch_end_time = time.time()
        self.per_epoch_ptime = self.epoch_end_time - self.epoch_start_time

    def set_epoch(self, epoch, optimizers):
        self.current_epoch = epoch
        self.iteration = 0
        self.tick_start_nimg_report = 0
        self.tick_start_nimg_snapshot = 0
        self.epoch_start_time = time.time()
