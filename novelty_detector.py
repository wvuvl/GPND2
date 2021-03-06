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

import torch.utils.data
from torchvision.utils import save_image
from net import *
import os
import utils
from checkpointer import Checkpointer
import numpy as np
import logging
import scipy.optimize
import pickle
from dataloader import *
from utils.jacobian import *
from torchvision.utils import save_image
#from dataloading import make_datasets, make_dataloader, create_set_with_outlier_percentage
from model import Model
from launcher import run
from defaults import get_cfg_defaults
from evaluation import get_f1, evaluate
from utils.threshold_search import find_maximum
from utils.save_plot import save_plot
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.metrics import roc_auc_score
from scipy.special import loggamma
from timeit import default_timer as timer
from scipy.optimize import minimize, dual_annealing
from utils.threshold_search import find_maximum_mv, find_maximum_mv_it


def r_pdf(x, bins, counts):
    if bins[0] < x < bins[-1]:
        i = np.digitize(x, bins) - 1
        return max(counts[i], 1e-308)
    if x < bins[0]:
        return max(counts[0] * x / bins[0], 1e-308)
    return 1e-308


def extract_statistics(cfg, train_set, model, output_folder, no_plots=False):
    rlist = []
    zlist = []

    data_loader = make_dataloader(train_set, cfg.TEST.BATCH_SIZE, torch.cuda.current_device())

    for y, x in data_loader:
        x = x.view(x.shape[0], cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
        z, _ = model.encode(x)

        rec = model.generator(z, False)

        recon_batch = rec.cpu().detach().numpy()
        x = x.cpu().detach().numpy()

        for i in range(x.shape[0]):
            distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())
            rlist.append(distance)

        z = z.cpu().detach().numpy()

        zlist.append(z)

    zlist = np.concatenate(zlist)

    counts, bin_edges = np.histogram(rlist, bins=30, density=True)

    if cfg.MAKE_PLOTS and not no_plots:
        plt.plot(bin_edges[1:], counts, linewidth=2)
        save_plot(r"Distance, $\left \|\| I - \hat{I} \right \|\|$",
                  'Probability density',
                  r"PDF of distance for reconstruction error, $p\left(\left \|\| I - \hat{I} \right \|\| \right)$",
                  output_folder + '/reconstruction_error.pdf')

    for i in range(cfg.MODEL.LATENT_SPACE_SIZE):
        plt.hist(zlist[:, i], density=True, bins='auto', histtype='step')

    if cfg.MAKE_PLOTS and not no_plots:
        save_plot(r"$z$",
                  'Probability density',
                  r"PDF of embeding $p\left(z \right)$",
                  output_folder + '/embeddingz.pdf')

    def fmin(func, x0, args, disp):
        x0 = [2.0, 0.0, 1.0]
        return scipy.optimize.fmin(func, x0, args, xtol=1e-12, ftol=1e-12, disp=0)

    gennorm_param = np.zeros([3, cfg.MODEL.LATENT_SPACE_SIZE])
    for i in range(cfg.MODEL.LATENT_SPACE_SIZE):
        betta, loc, scale = scipy.stats.gennorm.fit(zlist[:, i], optimizer=fmin)
        gennorm_param[0, i] = betta
        gennorm_param[1, i] = loc
        gennorm_param[2, i] = scale

    return counts, bin_edges, gennorm_param


def eval_model_on_valid(cfg, logger, model_s, folding_id, inliner_classes):
    train_set, valid_set, test_set = make_datasets(cfg, logger, folding_id, inliner_classes)
    print('Validation set size: %d' % len(valid_set))

    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    output_folder = os.path.join(cfg.OUTPUT_DIR, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        counts, bin_edges, gennorm_param = extract_statistics(cfg, train_set, model_s, output_folder)

    novelty_detector = model_s, bin_edges, counts, gennorm_param
    p = 50
    alpha, beta, threshold, f1 = compute_threshold_coeffs(cfg, logger, valid_set, inliner_classes, p, novelty_detector)
    return f1


def run_novely_prediction_on_dataset(cfg, dataset, inliner_classes, percentage, novelty_detector, concervative=False):
    model_s, bin_edges, counts, gennorm_param, = novelty_detector
    dataset.shuffle()

    dataset = create_set_with_outlier_percentage(dataset, inliner_classes, percentage, concervative)

    result = []
    gt_novel = []

    data_loader = make_dataloader(dataset, cfg.TEST.BATCH_SIZE, torch.cuda.current_device())

    include_jacobian = False

    N = cfg.MODEL.INPUT_IMAGE_CHANNELS * cfg.MODEL.INPUT_IMAGE_SIZE * cfg.MODEL.INPUT_IMAGE_SIZE - cfg.MODEL.LATENT_SPACE_SIZE
    logC = loggamma(N / 2.0) - (N / 2.0) * np.log(2.0 * np.pi)

    def logPe_func(x):
        # p_{\|W^{\perp}\|} (\|w^{\perp}\|)
        # \| w^{\perp} \|}^{m-n}
        return logC - (N - 1) * np.log(x), np.log(r_pdf(x, bin_edges, counts))

    for label, x in data_loader:
        x = x.view(x.shape[0], cfg.MODEL.INPUT_IMAGE_CHANNELS, cfg.MODEL.INPUT_IMAGE_SIZE, cfg.MODEL.INPUT_IMAGE_SIZE)
        z, _ = model_s.encode(x)

        rec = model_s.generator(z, False)

        if include_jacobian:
            # J = compute_jacobian(x, z)
            J = compute_jacobian_using_finite_differences_v3(z, model_s.generator)
            J = J.cpu().numpy()

        z = z.cpu().detach().numpy()

        recon_batch = rec.cpu().detach().numpy()
        x = x.cpu().detach().numpy()

        for i in range(x.shape[0]):
            if include_jacobian:
                u, s, vh = np.linalg.svd(J[i, :, :], full_matrices=False)
                logD = np.sum(np.log(np.abs(1.0 / s)))  # | \mathrm{det} S^{-1} |
                # logD = np.log(np.abs(1.0/(np.prod(s))))
            else:
                logD = 0

            p = scipy.stats.gennorm.pdf(z[i], gennorm_param[0, :], gennorm_param[1, :], gennorm_param[2, :])
            logPz = np.sum(np.log(p))

            # Sometimes, due to rounding some element in p may be zero resulting in Inf in logPz
            # In this case, just assign some large negative value to make sure that the sample
            # is classified as unknown.
            if not np.isfinite(logPz):
                logPz = -1000

            distance = np.linalg.norm(x[i].flatten() - recon_batch[i].flatten())

            logPe_p1, logPe_p2 = logPe_func(distance)

            result.append((logD, logPz, logPe_p1, logPe_p2))
            gt_novel.append(label[i].item() in inliner_classes)

    result = np.asarray(result, dtype=np.float32)
    ground_truth = np.asarray(gt_novel, dtype=np.float32)
    return result, ground_truth


def compute_threshold_coeffs(cfg, logger, valid_set, inliner_classes, percentage, novelty_detector):
    y_scores_components, y_true = run_novely_prediction_on_dataset(cfg, valid_set, inliner_classes, percentage, novelty_detector, concervative=True)

    y_scores_components = np.asarray(y_scores_components, dtype=np.float32)

    use_auc = False

    def evaluate_auc(threshold, beta, alpha):
        coeff = np.asarray([[-1, beta, alpha, 1]], dtype=np.float32)
        y_scores = (y_scores_components * coeff).mean(axis=1)

        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0

        return auc

    def evaluate_f1(threshold, beta, alpha):
        coeff = np.asarray([[-1, beta, alpha, 1]], dtype=np.float32)
        y_scores = (y_scores_components * coeff).mean(axis=1)

        y_false = np.logical_not(y_true)

        y = np.greater(y_scores, threshold)
        true_positive = np.sum(np.logical_and(y, y_true))
        false_positive = np.sum(np.logical_and(y, y_false))
        false_negative = np.sum(np.logical_and(np.logical_not(y), y_true))
        return get_f1(true_positive, false_positive, false_negative)

    def func(x):
        beta, alpha = x

        if use_auc:
            return evaluate_auc(0, beta, alpha)

        # Find threshold
        def eval(th):
            return evaluate_f1(th, beta, alpha)

        best_th, best_f1 = find_maximum(eval, *cfg.THRESHOLD_NARROW_WINDOW, 1e-2)

        return best_f1

    if cfg.ALPHA_BETA_TUNING:
        cmax, vmax = find_maximum_mv(func, [0.0, 0.0], [30.0, 1.0], xtoll=0.001, ftoll=0.001, verbose=True,
                                     n=8, max_iter=6)
        beta, alpha = cmax
    else:
        beta, alpha = cfg.BETA, cfg.ALPHA

    # Find threshold
    def eval(th):
        return evaluate_f1(th, beta, alpha)

    threshold, best_f1 = find_maximum(eval, *cfg.THRESHOLD_FINAL_WINDOW, 1e-3)

    logger.info("Best e: %f Best beta: %f Best a: %f best f1: %f" % (threshold, beta, alpha, best_f1))
    return alpha, beta, threshold, best_f1


def test(cfg, logger, test_set, inliner_classes, percentage, novelty_detector, alpha, beta, threshold, output_folder):
    y_scores_components, y_true = run_novely_prediction_on_dataset(cfg, test_set, inliner_classes, percentage, novelty_detector, concervative=True)
    y_scores_components = np.asarray(y_scores_components, dtype=np.float32)

    coeff = np.asarray([[-1, beta, alpha, 1]], dtype=np.float32)

    y_scores = (y_scores_components * coeff).mean(axis=1)

    with open(os.path.join(output_folder, "test_eval_normal.pkl"), "wb") as f:
        pickle.dump((y_scores, y_true), f)

    return evaluate(cfg, logger, percentage, inliner_classes, y_scores, threshold, y_true)


def main(cfg, logger, local_rank, folding_id, inliner_classes):
    torch.cuda.set_device(local_rank)
    train_set, valid_set, test_set = make_datasets(cfg, logger, folding_id, inliner_classes)
    train_set.shuffle()

    print('Validation set size: %d' % len(valid_set))
    print('Test set size: %d' % len(test_set))

    model_s = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        channels=cfg.MODEL.INPUT_IMAGE_CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER)
    model_s.cuda(local_rank)
    model_s.eval()
    model_s.requires_grad_(False)

    model_dict = {
        'encoder_s': model_s.encoder,
        'generator_s': model_s.generator,
    }

    output_folder = os.path.join('results_' + str(folding_id) + "_" + "_".join([str(x) for x in inliner_classes]))
    output_folder = os.path.join(cfg.OUTPUT_DIR, output_folder)
    os.makedirs(output_folder, exist_ok=True)

    checkpointer = Checkpointer(output_folder,
                                model_dict,
                                logger=logger,
                                save=False,
                                test=True)

    extra_checkpoint_data = checkpointer.load()
    last_epoch = list(extra_checkpoint_data['auxiliary']['scheduler'].values())[0]['last_epoch']
    logger.info("Model trained for %d epochs" % last_epoch)

    with torch.no_grad():
        counts, bin_edges, gennorm_param = extract_statistics(cfg, train_set, model_s, output_folder)

    novelty_detector = model_s, bin_edges, counts, gennorm_param,

    percentages = cfg.DATASET.PERCENTAGES
    # percentages = [50]

    results = {}
    for p in percentages:
        # plt.figure(num=None, figsize=(8, 6), dpi=180, facecolor='w', edgecolor='k')
        alpha, beta, threshold, _ = compute_threshold_coeffs(cfg, logger, valid_set, inliner_classes, p, novelty_detector)
        with open(os.path.join(output_folder, 'coeffs_percentage_%d.txt' % int(p)), 'w') as f:
            f.write("%f %f %f\n" % (alpha, beta, threshold))
        results[p] = test(cfg, logger, test_set, inliner_classes, p, novelty_detector, alpha, beta, threshold, output_folder)

    return results


if __name__ == "__main__":
    run(main, get_cfg_defaults(), description='', default_config='configs/mnist.yaml',
        world_size=1, folding_id=0, inliner_classes=[3])
