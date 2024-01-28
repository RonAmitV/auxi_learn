import argparse
import copy
import logging
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torch.nn import functional as F
from torchsummary import summary
from tqdm import trange

from auxilearn.hypernet import MonoHyperNet, MonoNoFCCNNHyperNet
from auxilearn.optim import MetaOptimizer
from experiments.nyuv2.data import nyu_dataloaders
from experiments.nyuv2.metrics import compute_iou, compute_miou
from experiments.nyuv2.model import SegNetSplit
from experiments.utils import get_device, set_logger, set_seed

# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(description="NYU - trainer CNN")
parser.add_argument("--dataroot", default="datasets/nyuv2", type=str, help="dataset root")
parser.add_argument("--n-meta-loss-accum", type=int, default=1, help="Number of batches to accumulate for meta loss")
parser.add_argument("--eval-every", type=int, default=1, help="num. epochs between test set eval")
parser.add_argument("--seed", type=int, default=45, help="random seed")
args = parser.parse_args()

# set seed - for reproducibility
set_seed(args.seed)
# logger config
set_logger()

# ======
# params
# ======
num_epochs = 200
batch_size = 8
val_batch_size = 2
aux_size = 0.025
meta_lr = 1e-4
meta_wd = 1e-5
hypergrad_every = 50


# =========
# load data
# =========
train_loader, val_loader, test_loader = nyu_dataloaders(
    datapath=args.dataroot,
    validation_indices="experiments/nyuv2/hpo_validation_indices.json",
    aux_set=False, # In our case - we don't split the training set into train and aux sets
    aux_size=aux_size,
    batch_size=batch_size,
    val_batch_size=val_batch_size,
)
train_loader_2 = deepcopy(train_loader)


# ====
# loss
# ====
def calc_loss(seg_pred, seg, depth_pred, depth, pred_normal, normal):
    """Per-pixel loss, i.e., loss image"""
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(depth.device)

    # semantic loss: depth-wise cross entropy
    seg_loss = F.nll_loss(seg_pred, seg, ignore_index=-1, reduction="none")

    # depth loss: l1 norm
    depth_loss = torch.sum(torch.abs(depth_pred - depth) * binary_mask, dim=1)

    # normal loss: dot product
    normal_loss = 1 - torch.sum((pred_normal * normal) * binary_mask, dim=1)

    return torch.stack((seg_loss, depth_loss, normal_loss), dim=1)


# =====
# model
# =====
# define model, optimiser and scheduler
device = get_device()
SegNet_SPLIT = SegNetSplit(logsigma=False)
summary(SegNet_SPLIT, input_size=(3, 288, 384), device="cpu")

SegNet_SPLIT = SegNet_SPLIT.to(device)

# ================
# hyperparam model
# ================
auxiliary_net = MonoNoFCCNNHyperNet(main_task=0, reduction="mean")
summary(auxiliary_net, input_size=(3, 288, 384), device="cpu")
auxiliary_net = auxiliary_net.to(device)

# ==========
# optimizers
# ==========
optimizer = optim.Adam(SegNet_SPLIT.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

meta_opt = optim.SGD(
    auxiliary_net.parameters(),
    lr=meta_lr,
    momentum=0.9,
    weight_decay=meta_wd,
)

meta_optimizer = MetaOptimizer(
    meta_optimizer=meta_opt,
    hpo_lr=1e-4,
    truncate_iter=3,
    max_grad_norm=100,
)


# ========
# evaluate
# ========
def evaluate(dataloader, model=None):
    model = model if model is not None else SegNet_SPLIT
    model.eval()
    total = 0
    eval_dict = defaultdict(float)

    with torch.no_grad():
        for _, batch_cpu in enumerate(dataloader):
            batch = (t.to(device) for t in batch_cpu)
            eval_data, eval_label, eval_depth, eval_normal = batch
            eval_label = eval_label.type(torch.LongTensor).to(device)

            eval_pred = model(eval_data)
            # loss
            eval_loss = calc_loss(
                eval_pred[0],
                eval_label,
                eval_pred[1],
                eval_depth,
                eval_pred[2],
                eval_normal,
            )

            eval_loss = eval_loss.mean(dim=(0, 2, 3))
            curr_batch_size = eval_data.shape[0]
            total += curr_batch_size
            curr_eval_dict = {
                "seg_loss": eval_loss[0].item() * curr_batch_size,
                "seg_miou": compute_miou(eval_pred[0], eval_label).item() * curr_batch_size,
                "seg_pixacc": compute_iou(eval_pred[0], eval_label).item() * curr_batch_size,
            }

            for k, v in curr_eval_dict.items():
                eval_dict[k] += v

    for k, v in eval_dict.items():
        eval_dict[k] = v / total

    model.train()

    return eval_dict


# ==============
# hypergrad step
# ==============
def hyperstep():
    main_loss_w_plus = 0.0
    main_loss_w_minus = 0.0
    for i_hyper_batch, hyper_batch_cpu in enumerate(train_loader_2):
        if i_hyper_batch < args.n_meta_loss_accum:
            
            # Compute the main task loss on current weights
            hyper_batch = (t.to(device) for t in hyper_batch_cpu)
            val_data, val_label, val_depth, val_normal = hyper_batch
            val_label = val_label.type(torch.LongTensor).to(device)
            val_pred = SegNet_SPLIT(val_data)
            val_loss = calc_loss(
                val_pred[0],
                val_label,
                val_pred[1],
                val_depth,
                val_pred[2],
                val_normal,
            )
            # (batch, task, height, width)
            val_loss = val_loss.mean(dim=(0, 2, 3))
            main_loss_w_plus += val_loss[0].mean(0)
            
            # create a "negative" batch - with wrong labels
            hyper_batch = (t.to(device) for t in hyper_batch_cpu)
            neg_data, neg_label, neg_depth, neg_normal = hyper_batch
                        
            # Compute the main task loss on current weights, with the "negative" batch
            # Flip labels
            neg_label = 12 - neg_label.type(torch.LongTensor).to(device)
            neg_depth = 11.0 - neg_depth
            neg_normal *= -1
            neg_pred = SegNet_SPLIT(neg_data)
            neg_loss = calc_loss(
                neg_pred[0],
                neg_label,
                neg_pred[1],
                neg_depth,
                neg_pred[2],
                neg_normal,
            )
            neg_loss = neg_loss.mean(dim=(0, 2, 3))
            main_loss_w_minus += neg_loss[0].mean(0)
            
            
            
            

    # inner_loop_end_train_loss, e.g. dL_train/dw
    total_meta_train_loss = 0.0
    for n_train_step, train_batch_cpu in enumerate(train_loader):
        if n_train_step < args.n_meta_loss_accum:
            train_batch = (t.to(device) for t in train_batch_cpu)

            train_data, train_label, train_depth, train_normal = train_batch
            train_label = train_label.type(torch.LongTensor).to(device)

            train_pred = SegNet_SPLIT(train_data)

            train_loss = calc_loss(
                train_pred[0],
                train_label,
                train_pred[1],
                train_depth,
                train_pred[2],
                train_normal,
            )

            meta_train_loss = auxiliary_net(train_loss)
            total_meta_train_loss += meta_train_loss

    # hyperpatam step
    curr_hypergrads = meta_optimizer.step(
        main_loss=main_loss_w_plus,
        train_loss=total_meta_train_loss,
        aux_params=list(auxiliary_net.parameters()),
        primary_param=list(SegNet_SPLIT.parameters()),
        return_grads=True,
    )

    return curr_hypergrads


# ==========
# train loop
# ==========
best_metric = np.NINF
best_model_epoch = 0
best_miou, best_pixacc = 0, 0
step = 0


epoch_iter = trange(num_epochs)
for epoch in epoch_iter:
    # iteration for all batches
    SegNet_SPLIT.train()
    for k, batch_cpu in enumerate(train_loader):
        step += 1
        batch = (t.to(device) for t in batch_cpu)
        train_data, train_label, train_depth, train_normal = batch
        train_label = train_label.type(torch.LongTensor).to(device)

        train_pred = SegNet_SPLIT(train_data)

        optimizer.zero_grad()
        train_losses = calc_loss(
            train_pred[0],
            train_label,
            train_pred[1],
            train_depth,
            train_pred[2],
            train_normal,
        )

        # (batch, task, height, width)
        avg_train_losses = train_losses.mean(dim=(0, 2, 3))

        # weighted loss
        loss = auxiliary_net(train_losses)

        epoch_iter.set_description(f"[{epoch} {k}] Training loss {loss.data.cpu().numpy().item():.5f}")

        loss.backward()
        optimizer.step()

        # hyperparams step
        if step % hypergrad_every == 0:
            curr_hypergrads = hyperstep()

            if isinstance(auxiliary_net, MonoHyperNet):
                # monotonic network
                auxiliary_net.clamp()

    scheduler.step(epoch=epoch)

    if (epoch + 1) % args.eval_every == 0:
        val_metrics = evaluate(val_loader)
        test_metrics = evaluate(test_loader)

        logging.info(
            f"Epoch: {epoch + 1}, Test mIoU = {test_metrics['seg_miou']:.4f}, "
            f"Test PixAcc = {test_metrics['seg_pixacc']:.4f}",
        )

        if val_metrics["seg_miou"] >= best_metric:
            logging.info(f"Saving model, epoch {epoch + 1}")
            best_model_epoch = epoch + 1
            best_metric = val_metrics["seg_miou"]
            best_model = copy.deepcopy(SegNet_SPLIT)
            best_hypernet_model = copy.deepcopy(auxiliary_net)

# final evaluation
logging.info(f"End of training, best model from epoch {best_model_epoch}")

test_metrics = evaluate(test_loader, model=best_model)
logging.info(
    f"Epoch: {epoch + 1}, Test mIoU = {test_metrics['seg_miou']:.4f}, Test PixAcc = {test_metrics['seg_pixacc']:.4f}",
)
# ----------------------------------------------------------------------
