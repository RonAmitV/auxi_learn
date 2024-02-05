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

from experiments.nyuv2.data import get_nyu_dataset
from experiments.nyuv2.metrics import compute_iou, compute_miou
from experiments.nyuv2.model import SegNetSplit
from experiments.utils import get_device, set_logger, set_seed, to_np


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NYU - trainer CNN")
    parser.add_argument("--dataroot", default="datasets/nyuv2", type=str, help="dataset root")
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

    # =========
    # load data
    # =========
    train_set, val_set, test_set = get_nyu_dataset(
        datapath=args.dataroot,
        validation_indices="experiments/nyuv2/hpo_validation_indices.json",
    )
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=val_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
    )
    train_loader2 = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
    )
    data_iter2 = iter(train_loader2)
    n_aux_tasks  = 2 # depth and normal
    
    # =====
    # model
    # =====
    # define model, optimizer and scheduler
    device = get_device()
    prim_model = SegNetSplit(logsigma=False)
    summary(prim_model, input_size=(3, 288, 384), device="cpu")
    prim_model = prim_model.to(device)

    # ==========
    # optimizers
    # ==========
    optimizer = optim.Adam(prim_model.parameters(), lr=1e-4)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # ==========
    # train loop
    # ==========
    best_metric = -np.inf
    best_model_epoch = 0
    step = 0

    epoch_iter = trange(num_epochs)
    for epoch in epoch_iter:
        # iteration for all batches
        prim_model.train()
        for k, batch_cpu1 in enumerate(train_loader):
            step += 1
            batch1 = (t.contiguous().to(device) for t in batch_cpu1)
            train_data1, train_label1, train_depth1, train_normal1 = batch1
            train_label1 = train_label1.type(torch.LongTensor).to(device)

            train_pred1 = prim_model(train_data1)
            # Get the per-pixel predictions from the model (segmentation, depth, normal)
            seg_pred1, depth_pred1, normal_pred1 = train_pred1

            optimizer.zero_grad()

            # main task loss (average over batch and pixels)
            main_loss1 = calc_main_loss(seg_pred=seg_pred1, seg=train_label1).mean()
            
            epoch_iter.set_description(f"[{epoch} {k}] Train loss_main {to_np(main_loss1):.5f}")

            main_loss1.backward()
            optimizer.step()
            main_loss1 = to_np(main_loss1)
            
            # save the current model state
            model_state = copy.deepcopy(prim_model.state_dict())
            
            # Get another batch for the auxiliary tasks
            try:
                batch_cpu2 = next(data_iter2)
            except StopIteration:
                data_iter2 = iter(train_loader2)
                batch_cpu2 = next(data_iter2)
            batch2 = (t.contiguous().to(device) for t in batch_cpu2)
            train_data2, train_label2, train_depth2, train_normal2 = batch2
            train_label2 = train_label2.type(torch.LongTensor).to(device)
            

            #  Probe the main task loss on the first batch after taking a gradient step with each auxiliary task
            main_loss_probes = np.zeros(n_aux_tasks)
            for i_task in range(n_aux_tasks):
                optimizer.zero_grad()
                # Get the current task auxiliary loss (average over batch and pixels) on batch 2
                # Get the per-pixel predictions from the model (segmentation, depth, normal)
                train_pred2 = prim_model(train_data2)
                seg_pred2, depth_pred2, normal_pred2 = train_pred2
                aux_loss = calc_auxiliary_loss(i_task, depth_pred2, train_depth2, normal_pred2, train_normal2).mean()
                aux_loss.backward()
                optimizer.step()
                # check the main task loss with batch 1
                train_pred1 = prim_model(train_data1)
                seg_pred1, _, _ = train_pred1
                main_loss_check = calc_main_loss(seg_pred=seg_pred1, seg=train_label1).mean()
                main_loss_probes[i_task] = to_np(main_loss_check)
                # restore the model state
                prim_model.load_state_dict(model_state)
                del aux_loss, main_loss_check
            
            # choose the auxiliary task with the smallest main task loss after step (or keep the previous model)
            best_aux_task = np.argmin(main_loss_probes)
            if main_loss_probes[best_aux_task] < main_loss1:
                # If the main task loss is reduced, take a gradient step with the best auxiliary task.
                train_pred2 = prim_model(train_data2)
                seg_pred2, depth_pred2, normal_pred2 = train_pred2
                aux_loss = calc_auxiliary_loss(best_aux_task, depth_pred2, train_depth2, normal_pred2, train_normal2).mean()
                aux_loss.backward()
                optimizer.step()
            
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


# ====
# tasks losses
# ====
def calc_main_loss(seg_pred, seg):
    """Per-pixel loss, i.e., loss image
    Args:
        seg_pred: (batch, n_classes, height, width) - estimation log probabilities of the segmentation classes.
        seg: (batch, height, width) The ground truth segmentation.
    Returns:
        seg_loss: (batch, height, width) = The per-pixel loss for the segmentation task.
    """
    # semantic loss: depth-wise cross entropy
    seg_loss = F.nll_loss(input=seg_pred, target=seg, ignore_index=-1, reduction="none")

    return seg_loss


# ----------------------------------------------------------------------


def calc_auxiliary_loss(i_task, depth_pred, depth, pred_normal, normal):
    """Calculate the axuiliary task losses. Per-pixel loss, i.e., loss image.
    Args:
        depth_pred: (batch, 1, height, width) - estimation of the depth.
        depth: (batch, height, width) The ground truth depth.
        pred_normal: (batch, 3, height, width) - estimation of the normal.
        normal: (batch, 3, height, width) The ground truth normal.
    Returns:
        task_losses: (batch, task, height, width) - The per-pixel loss for the auxiliary tasks.
    """
    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(depth, dim=1) != 0).type(torch.FloatTensor).unsqueeze(1).to(depth.device)
    
    if i_task == 0:
        # depth loss: l1 norm
        # depth loss: l1 norm
        depth_loss = torch.sum(torch.abs(depth_pred - depth) * binary_mask, dim=1)
        return depth_loss
    
    if i_task == 1:
        # normal loss: dot product
        normal_loss = 1 - torch.sum((pred_normal * normal) * binary_mask, dim=1)
        return normal_loss

    raise ValueError("Invalid task index")

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
            batch = (t.contiguous().to(device) for t in batch_cpu)
            eval_data, eval_label, eval_depth, eval_normal = batch
            eval_label = eval_label.type(torch.LongTensor).to(device)

            eval_pred = prim_model(eval_data)

            eval_loss = calc_main_loss(seg_pred=eval_pred[0], seg=eval_label)

            # average over batch and pixels
            eval_loss = eval_loss.mean()
            curr_batch_size = eval_data.shape[0]
            total += curr_batch_size
            curr_eval_dict = {
                "seg_loss": eval_loss.item() * curr_batch_size,
                "seg_miou": compute_miou(eval_pred, eval_label).item() * curr_batch_size,
                "seg_pixacc": compute_iou(eval_pred, eval_label).item() * curr_batch_size,
            }

            for k, v in curr_eval_dict.items():
                eval_dict[k] += v

    for k, v in eval_dict.items():
        eval_dict[k] = v / total

    prim_model.train()

    return eval_dict


# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
