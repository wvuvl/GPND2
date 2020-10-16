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
    cfg_file = 'configs/covid_1.yaml'

cfg = get_cfg_defaults()
cfg.merge_from_file(cfg_file)
cfg.freeze()


def f(fold_id):
    import train_alae
    import novelty_detector
    import torch

    inliner_classes = 0

    # torch.cuda.set_device(0)
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.cuda.current_device()
    print("Running on ", torch.cuda.get_device_name(device))
    train_alae.train(cfg=cfg, logger=logger, local_rank=device, world_size=1, folding_id=fold_id, inliner_classes=[inliner_classes])

    res = novelty_detector.main(cfg=cfg, logger=logger, local_rank=device, folding_id=fold_id, inliner_classes=[inliner_classes])
    return res


gpu_count = min(utils.multiprocessing.get_gpu_count(), 8)

results = utils.multiprocessing.map(f, gpu_count, list(range(gpu_count)))

save_results(results, os.path.join(cfg.OUTPUT_DIR, cfg.RESULTS_NAME))
