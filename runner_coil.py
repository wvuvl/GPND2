import logging
import sys
import utils.multiprocessing
from defaults import get_cfg_defaults
from save_to_csv import save_results

import novelty_detector
from random import randint
from random import seed
import os

full_run = False


logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

if len(sys.argv) > 1:
    cfg_file = 'configs/' + sys.argv[1]
else:
    cfg_file = 'configs/coil100.yaml'

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_file)

seed(0)
settings = []

for outlier_percentage, inlier_classes in [(50, 1), (25, 4), (15, 7)]:

    for fold in range(5 if full_run else 1):
        for i in range(20):
            inliner_classes = []
            for c in range(inlier_classes):
                inliner_classes += [randint(1, 100)]
            settings.append(dict(fold=fold, inliner_classes=inliner_classes, percetange=outlier_percentage))

    def f(setting):
        import train_alae
        import novelty_detector
        import torch

        fold_id = setting['fold']
        inliner_classes = setting['inliner_classes']
        outlier_percentage = setting['percetange']
        device = torch.cuda.current_device()

        _cfg = cfg.clone()

        _cfg.DATASET.PERCENTAGES = [outlier_percentage]
        _cfg.freeze()

        train_alae.train(cfg=_cfg, logger=logger, local_rank=device, world_size=1, folding_id=fold_id, inliner_classes=inliner_classes)

        res = novelty_detector.main(cfg=_cfg, logger=logger, local_rank=device, folding_id=fold_id, inliner_classes=inliner_classes)
        return res


gpu_count = min(utils.multiprocessing.get_gpu_count(), 8)

results = utils.multiprocessing.map(f, gpu_count, settings)

save_results(results, os.path.join(cfg.OUTPUT_DIR, cfg.RESULTS_NAME))
