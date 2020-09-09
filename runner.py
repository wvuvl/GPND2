from save_to_csv import save_results
import logging
import sys
import utils.multiprocessing
from defaults import get_cfg_defaults
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
    cfg_file = 'configs/cifar10_os.yaml'

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_file)
cfg.freeze()

classes_count = cfg.DATASET.TOTAL_CLASS_COUNT

settings = []

fold_range = 5
if not full_run:
    fold_range = 1
if cfg.DATASET.OFFICIAL_SPLIT:
    fold_range = 1


for fold in range(5 if full_run else 1):
    for i in range(classes_count):
        settings.append(dict(fold=fold, digit=i))


def f(setting):
    import train_alae
    import novelty_detector
    import torch

    fold_id = setting['fold']
    inliner_classes = setting['digit']
    device = torch.cuda.current_device()

    train_alae.train(cfg=cfg, logger=logger, local_rank=device, world_size=1, folding_id=fold_id, inliner_classes=[inliner_classes])

    res = novelty_detector.main(cfg=cfg, logger=logger, local_rank=device, folding_id=fold_id, inliner_classes=[inliner_classes])
    return res


gpu_count = min(utils.multiprocessing.get_gpu_count(), 8)

results = utils.multiprocessing.map(f, gpu_count, settings)

save_results(results, os.path.join(cfg.OUTPUT_DIR, cfg.RESULTS_NAME))
