import argparse
import copy
import logging
from collections import defaultdict

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


# ----------------------------------------------------------------------


# ========
# evaluate
# ========
def evaluate(dataloader, prim_model: torch.nn.Module, device=None):
    prim_model.eval()
    total = 0
    eval_dict = defaultdict(float)

    with torch.no_grad():
        for _, batch_cpu in enumerate(dataloader):
            batch = (t.to(device) for t in batch_cpu)
            eval_data, eval_label, eval_depth, eval_normal = batch
            eval_label = eval_label.type(torch.LongTensor).to(device)

            eval_pred = prim_model(eval_data)
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

    prim_model.train()

    return eval_dict


# ----------------------------------------------------------------------
# ==============
# hypergrad step
# ==============
def hyperstep(
    prim_model: torch.nn.Module,
    aux_model: torch.nn.Module,
    train_loader,
    meta_optimizer,
    device,
    n_meta_loss_accum: int,
):
    # Compute these terms by summing over batches:
    main_loss_w_plus = 0.0  #  \Lmain (W, D)
    aux_loss_w_plus = 0.0  #  \Laux(W, \phi,  D)
    main_loss_w_minus = 0.0  #  \Lmain (W, D^-)
    aux_loss_w_minus = 0.0  #  \Laux(W, \phi,  D^-)

    for i_hyper_batch, hyper_batch_cpu in enumerate(train_loader):
        if i_hyper_batch >= n_meta_loss_accum:
            break

        # Get the "positive" real data batch
        hyper_batch = (t.to(device) for t in hyper_batch_cpu)
        x_data, pos_label, pos_depth, pos_normal = hyper_batch
        pos_label = pos_label.type(torch.LongTensor).to(device)

        net_pred = prim_model(x_data)

        # Compute the tasks losses on current weights
        pos_losses = calc_loss(
            net_pred[0],
            pos_label,
            net_pred[1],
            pos_depth,
            net_pred[2],
            pos_normal,
        )
        # (batch, task, height, width)

        # Compute the the total auxiliary loss: \Laux(W, \phi,  D)
        aux_loss_w_plus += aux_model(pos_losses)

        # average over batch and spatial dimensions
        pos_losses = pos_losses.mean(dim=(0, 2, 3))  # [n_tasks]

        # Compute the main task loss:   \Lmain (W, D)
        main_loss_w_plus += pos_losses[0].mean(0)  # [1]

        # create a "negative" batch - with wrong random labels
        neg_label = torch.randint_like(pos_label, -1, 13)
        neg_depth = torch.rand_like(pos_depth) * 10
        neg_normal = torch.rand_like(pos_normal) * 2 - 1

        # Compute the main task loss on current weights, with the "negative" batch
        neg_losses = calc_loss(
            net_pred[0],
            neg_label,
            net_pred[1],
            neg_depth,
            net_pred[2],
            neg_normal,
        )
        # Compute the the total auxiliary loss on "negative" batch: \Laux(W, \phi,  D^-)
        aux_loss_w_minus += aux_model(neg_losses)

        # average over batch and spatial dimensions
        neg_losses = neg_losses.mean(dim=(0, 2, 3))  # [n_tasks]

        # Compute the main task loss on "negative" batch   \Lmain (W, D^-)
        main_loss_w_minus += neg_losses[0].mean(0)

    # The total train loss is the sum of the main and auxiliary losses
    train_loss_w_plus = main_loss_w_plus + aux_loss_w_plus

    train_loss_w_minus = main_loss_w_minus + aux_loss_w_minus

    # Compute the the total hypergradient by the implicit differentiation of of the positive term
    # minus the implicit differentiation of thenegative term

    # The gradient of \Lmain (W^+(\phi), D) w.r.t. \phi
    plus_hypergrads = meta_optimizer.step(
        main_loss=main_loss_w_plus,
        train_loss=train_loss_w_plus,
        aux_params=list(aux_model.parameters()),
        primary_param=list(prim_model.parameters()),
        take_step=False,
        return_grads=True,
    )
    # The gradient of \Lmain (W^-(\phi), D) w.r.t. \phi
    neg_hypergrads = meta_optimizer.step(
        main_loss=main_loss_w_plus,
        train_loss=train_loss_w_minus,
        aux_params=list(aux_model.parameters()),
        primary_param=list(prim_model.parameters()),
        take_step=False,
        return_grads=True,
    )

    hypergrads = [plus - neg for plus, neg in zip(plus_hypergrads, neg_hypergrads, strict=False)]

    # Take a meta_optimisation step using the hypergradients
    meta_optimizer.optimizer_step(aux_params=list(aux_model.parameters()), hyper_gards=hypergrads)

    return hypergrads


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NYU - trainer CNN")
    parser.add_argument("--dataroot", default="datasets/nyuv2", type=str, help="dataset root")
    parser.add_argument(
        "--n-meta-loss-accum",
        type=int,
        default=1,
        help="Number of batches to accumulate for meta loss",
    )
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
        aux_set=False,  # In our case - we don't split the training set into train and aux sets
        aux_size=aux_size,
        batch_size=batch_size,
        val_batch_size=val_batch_size,
    )

    # =====
    # model
    # =====
    # define model, optimizer and scheduler
    device = get_device()
    prim_model = SegNetSplit(logsigma=False)
    summary(prim_model, input_size=(3, 288, 384), device="cpu")
    prim_model = prim_model.to(device)

    # ================
    # hyperparam model
    # ================
    aux_model = MonoNoFCCNNHyperNet(main_task=0, reduction="mean")
    summary(aux_model, input_size=(3, 288, 384), device="cpu")
    aux_model = aux_model.to(device)

    # ==========
    # optimizers
    # ==========
    optimizer = optim.Adam(prim_model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    meta_opt = optim.SGD(
        aux_model.parameters(),
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

    # ==========
    # train loop
    # ==========
    best_metric = np.NINF
    best_model_epoch = 0
    step = 0

    epoch_iter = trange(num_epochs)
    for epoch in epoch_iter:
        # iteration for all batches
        prim_model.train()
        for k, batch_cpu in enumerate(train_loader):
            step += 1
            batch = (t.to(device) for t in batch_cpu)
            train_data, train_label, train_depth, train_normal = batch
            train_label = train_label.type(torch.LongTensor).to(device)

            train_pred = prim_model(train_data)

            optimizer.zero_grad()
            train_losses = calc_loss(
                train_pred[0],
                train_label,
                train_pred[1],
                train_depth,
                train_pred[2],
                train_normal,
            )  # (batch, task, height, width)

            # compute the total train loss
            train_loss = aux_model(train_losses)

            epoch_iter.set_description(f"[{epoch} {k}] Training loss {train_loss.data.cpu().numpy().item():.5f}")

            train_loss.backward()
            optimizer.step()

            # hyperparams step
            if step % hypergrad_every == 0:
                hyperstep(
                    prim_model=prim_model,
                    aux_model=aux_model,
                    train_loader=train_loader,
                    meta_optimizer=meta_optimizer,
                    device=device,
                    n_meta_loss_accum=args.n_meta_loss_accum,
                )

                if isinstance(aux_model, MonoHyperNet):
                    # monotonic network
                    aux_model.clamp()

        lr_scheduler.step()

        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(dataloader=val_loader, prim_model=prim_model, device=device)
            test_metrics = evaluate(dataloader=test_loader, prim_model=prim_model, device=device)

            logging.info(
                f"Epoch: {epoch + 1}, Test mIoU = {test_metrics['seg_miou']:.4f}, "
                f"Test PixAcc = {test_metrics['seg_pixacc']:.4f}",
            )

            if val_metrics["seg_miou"] >= best_metric:
                logging.info(f"Saving model, epoch {epoch + 1}")
                best_model_epoch = epoch + 1
                best_metric = val_metrics["seg_miou"]
                best_model = copy.deepcopy(prim_model)
                # best_hypernet_model = copy.deepcopy(aux_model)

    # final evaluation
    logging.info(f"End of training, best model from epoch {best_model_epoch}")

    test_metrics = evaluate(dataloader=test_loader, prim_model=best_model, device=device)
    logging.info(
        f"Epoch: {epoch + 1}, Test mIoU = {test_metrics['seg_miou']:.4f}, Test PixAcc = {test_metrics['seg_pixacc']:.4f}",
    )


# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
