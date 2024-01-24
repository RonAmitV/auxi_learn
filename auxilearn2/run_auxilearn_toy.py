# source: https://github.com/AvivNavon/AuxiLearn
# https://jovian.com/ahmadyahya11/pytorch-cnn-cifar10
import tarfile
from pathlib import Path

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor

from src.auxilearn.hypernet import MonoNonlinearHyperNet

# set anomaly detection to detect gradient problems
torch.autograd.set_detect_anomaly(True)
# ----------------------------------------------------------------------


def get_single_class_loss(model_out, labels, i_class):
    """Returns the loss for the main task given the model output and the labels.
    The main task is a multi-class classification problem for a specific class.
    """
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model_out[:, i_class],
        (labels == i_class).float(),
        reduction="none",
    )
    return loss  # [batch_size]


# ----------------------------------------------------------------------


def get_tasks_losses(model_out, labels):
    """Returns the losses for each task given the model output and the labels.
    Each task is a binary classification problem for a specific class.
    """
    losses = []
    for i in range(model_out.shape[1]):
        losses.append(get_single_class_loss(model_out=model_out, labels=labels, i_class=i))
    losses = torch.stack(losses, dim=1)  # [batch_size, n_classes]
    return losses


# ----------------------------------------------------------------------


def main():
    num_epochs = 10
    batch_size = 256
    val_batch_size = 256
    val_ratio = 0.1  # ratio of the dataset examples to be used for validation
    aux_wd = 1e-4  # auxiliary model's optimizer  weight decay
    hypergrad_every = 20 # number of steps between each update of the auxiliary model's parameters
    random_seed = 42
    main_task_class = 0  # class index of the main task
    max_grad_norm = None  # max norm for grad clipping

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)

    # Download the dataset
    data_dir = Path("local/datasets/cifar10")
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, root=data_dir)
        # Extract from archive - no need if extracted once already
        with tarfile.open(data_dir / "cifar10.tgz", "r:gz") as tar:
            tar.extractall(path=data_dir.parent)
        (data_dir / "cifar10.tgz").unlink()

    # prepare dataset
    dataset = ImageFolder(data_dir / "train", transform=ToTensor())
    print("classes: ", dataset.classes)
    n_classes = len(dataset.classes)

    # split dataset into train and validation sets
    val_size = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # create data loaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, val_batch_size, num_workers=4)
    
    # val_iter = iter(val_dl)

    # create primary model - a CNN that classifies images
    primary_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", weights="ResNet18_Weights.DEFAULT")
    primary_model.fc = torch.nn.Linear(512, n_classes)
    primary_model = primary_model.to(device)

    # auxiliary_model (compute the total train auxiliary loss given all the tasks losses)
    auxiliary_model = MonoNonlinearHyperNet(
        input_dim=n_classes,  # total number of tasks
        main_task=main_task_class,  # class index of the main task
        hidden_sizes=[3],  # sizes of the hidden layers
    ).to(device)

    # optimizers
    primary_optimizer = torch.optim.AdamW(primary_model.parameters())

    aux_optimizer = torch.optim.AdamW(auxiliary_model.parameters(), lr=1e-3, weight_decay=aux_wd)

    # training loop
    step = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        primary_model.train()
        for batch_cpu in train_loader:
            step += 1
            # calculate batch loss using 'primary_model' and 'auxiliary_model'
            batch = (t.to(device) for t in batch_cpu)
            images, labels = batch

            # Forward pass with primary model
            model_out = primary_model(images)

            # get the list of train losses for each task
            train_losses = get_tasks_losses(model_out, labels)

            # combine losses using 'auxiliary_model' to get the total train loss
            train_loss = auxiliary_model(train_losses).mean()  # mean over the batch
            primary_optimizer.zero_grad()

            #  Compute the gradients of the train loss (main + auxiliary) w.r.t. the primary model's parameters
            train_loss.backward()
            
            # grad clipping
            if max_grad_norm is not None:
                clip_grad_norm_(primary_model.parameters(), max_norm=max_grad_norm)
                
            # take a step in the primary model's parameters
            primary_optimizer.step()
                    
            #  auxiliary parameters step
            if hypergrad_every and step % hypergrad_every == 0:
                aux_optimizer.zero_grad()
                primary_optimizer.zero_grad()
                
                # # get a validation batch
                # try:
                #     val_batch = next(val_iter)
                # except StopIteration:
                #     val_iter = iter(val_dl)
                #     val_batch = next(val_iter)
                # val_batch = (t.to(device) for t in val_batch)
                # images, labels = val_batch
                    
                # Forward pass with primary model
                model_out = primary_model(images)

                # get the list of train losses for each task
                train_losses = get_tasks_losses(model_out, labels)

                # combine losses using 'auxiliary_model' to get the total train loss
                train_loss = auxiliary_model(train_losses).mean()  # mean over the batch

                #  Compute the gradients of the train loss (main + auxiliary) w.r.t. the primary model's parameters
                d_train_loss_dw = torch.autograd.grad(train_loss, list(primary_model.parameters()), create_graph=True, retain_graph=True)

                # grad clipping
                if max_grad_norm is not None:
                    clip_grad_norm_(d_train_loss_dw, max_norm=max_grad_norm)
                    
                main_loss = train_losses[:, main_task_class].mean()  # mean over the batch

                # compute the gradient of the main loss w.r.t. the primary model's parameters at current fixed point weights (assumed to be approx W*)
                d_main_loss_dw = torch.autograd.grad(main_loss, list(primary_model.parameters()), create_graph=True)

                # Compute approx to  aux_lr * H^-1 @ d_main_loss_dw, where H is the Hessian of the train loss w.r.t. the primary model's parameters.
                d_main_dw_times_inv_hess = approx_vector_prod_with_inv_hessian(
                    vec=d_main_loss_dw,
                    grad_w=d_train_loss_dw,
                    params_w=list(primary_model.parameters()),
                    alpha=1e-4,
                )

                # multiply by -1:
                d_main_dw_times_inv_hess = [-g for g in d_main_dw_times_inv_hess]

                # compute the Jacobian of d_train_loss_dw w.r.t. the auxiliary model's parameters (d/dPhi d_train_loss_dw) [W x P]
                # times the vector dmain_dw_times_inv_hess  [W x 1] to get the gradient of the main loss w.r.t. the auxiliary model's parameters
                # at W* [P x 1]
                d_main_d_phi = torch.autograd.grad(
                    d_train_loss_dw,
                    list(auxiliary_model.parameters()),
                    grad_outputs=d_main_dw_times_inv_hess,
                    allow_unused=True,
                    retain_graph=True,
                )
                # grad clipping
                if max_grad_norm is not None:
                    clip_grad_norm_(d_main_d_phi, max_norm=max_grad_norm)
                    
                # use the gradient of the main loss w.r.t. the auxiliary model's parameters to update the auxiliary model's parameters
                for p, g in zip(auxiliary_model.parameters(), d_main_d_phi, strict=False):
                    p.grad = g
                aux_optimizer.step()


        # evaluate the loss and accuracy of the primary model on the validation set on the main task
        primary_model.eval()
        val_loss = 0
        val_acc = 0
        val_main_task_acc = 0
        n_examples = 0
        n_examples_main_task = 0

        for batch_cpu in val_dl:
            batch = (t.to(device) for t in batch_cpu)
            images, labels = batch
            model_out = primary_model(images)
            cur_val_loss = get_single_class_loss(model_out=model_out, labels=labels, i_class=main_task_class)
            val_loss += cur_val_loss.sum()
            class_pred = model_out.argmax(dim=1)
            val_acc += (class_pred == labels).sum()
            val_main_task_acc += (class_pred[labels == main_task_class] == main_task_class).sum()
            n_examples_main_task += (labels == main_task_class).sum()
            n_examples += len(images)
        val_loss = val_loss / n_examples
        val_acc = val_acc / n_examples
        val_main_task_acc = val_main_task_acc / n_examples_main_task
        print(
            f"val_loss_main_task: {val_loss:.4f}, val_total_acc: {val_acc:.4f}, val_main_task_acc: {val_main_task_acc:.4f}",
        )


# ----------------------------------------------------------------------


def approx_vector_prod_with_inv_hessian(vec: torch.Tensor, grad_w, params_w, alpha=0.1):
    """
    Approximates the vector product with the inverse Hessian matrix using the truncated sum approximation.
    The Hessian is H = d(grad_w)/d(params_w) [W x W] and the vector is vec [W x 1].
    The function approximates  inv(alpha * H) @ vec [1 x W] with:
    [sum_{i=0..K}  (1 - alpha * H)^i] @ vec
    where K is the number of iterations in the truncated sum approximation.
    Since the hessian is too large for memory, we re-compute the hessian times vector product at each iteration,
    using this form:
    [sum_{i=0..K}  c_i] @ vec
    where c_i = (1 - alpha * H)^i = (1 - alpha * H) @ c_{i-1} @ = c_{i-1} - alpha *  H @ c_{i-1}, c_0 = I
    thus the sum can be computed by:
    sum_{i=0..K} b_i, where b_i = c_i @ vec =  b_{i-1} - alpha *  H @ b_{i-1}, and b_0 = vec

    Args:
        vec: a vector of size W
        grad_w: the gradient of the loss w.r.t. the model's parameters
        params_w: the model's parameters
        alpha: a scalar
    """

    truncate_iter = 4  # how many elements in the truncated sum that approximates the inverse Hessian vector product

    b = vec  # the current summed vector (at i=0).
    s = vec  # the current accumalated sum (up to i=0).

    # multiply grad_w by alpha:
    grad_w = [g * alpha for g in grad_w]

    for _ in range(1, truncate_iter):
        # Compute the Hessian times vector product:  alpha * H @ b_{i-1}
        H_b = torch.autograd.grad(
            grad_w,
            params_w,
            grad_outputs=b,
            retain_graph=True,
            allow_unused=True,
        )

        # Update b_i = b_{i-1} - alpha *  H @ b_{i-1}
        b = [curr_b - curr_h for (curr_b, curr_h) in zip(b, H_b, strict=False)]

        # update the accumalated sum: s_i = s_{i-1} + b_i
        s = [curr_s + curr_b for (curr_s, curr_b) in zip(s, b, strict=False)]

    return s


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
