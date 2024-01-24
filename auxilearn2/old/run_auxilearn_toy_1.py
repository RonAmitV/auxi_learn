# source: https://github.com/AvivNavon/AuxiLearn
# https://jovian.com/ahmadyahya11/pytorch-cnn-cifar10
import tarfile
from pathlib import Path

import torch
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor

from src.auxilearn.hypernet import MonoHyperNet, MonoLinearHyperNet, MonoNonlinearHyperNet
from src.auxilearn.optim import MetaOptimizer

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
    meta_val_ratio = 0.1  # ratio of the dataset examples to be used for updating the auxiliary parameters
    aux_lr = 1e-4  # auxiliary model's optimizer learning rate
    aux_wd = 1e-4  # auxiliary model's optimizer  weight decay
    hypergrad_every = 50
    random_seed = 42
    main_task_class = 0  # class index of the main task
    n_meta_loss_accum = 1  # Number of batches to accumulate for meta loss

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
    meta_val_size = int(meta_val_ratio * len(dataset))
    train_size = len(dataset) - val_size - meta_val_size
    train_ds, val_ds, meta_val_ds = random_split(dataset, [train_size, val_size, meta_val_size])

    # create data loaders
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, val_batch_size, num_workers=4)
    meta_val_dl = DataLoader(meta_val_ds, batch_size, shuffle=True, num_workers=4)
    meta_val_dl_iter = iter(meta_val_dl)
    meta_train_iter = iter(train_loader)

    # create primary model - a CNN that classifies images
    primary_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", weights="ResNet18_Weights.DEFAULT")
    primary_model.fc = torch.nn.Linear(512, n_classes)
    primary_model = primary_model.to(device)

    # auxiliary_model (compute the total train auxiliary loss given all the tasks losses)
    auxiliary_model_type = "linear"
    if auxiliary_model_type == "mono_nonlinear":
        auxiliary_model = MonoNonlinearHyperNet(
            input_dim=n_classes,  # total number of tasks
            main_task=main_task_class,  # class index of the main task
            hidden_sizes=[3],  # sizes of the hidden layers
        ).to(device)
    elif auxiliary_model_type == "linear":
        auxiliary_model = MonoLinearHyperNet(
            input_dim=n_classes,  # total number of tasks
            main_task=main_task_class,  # class index of the main task
        ).to(device)

    # optimizers
    primary_optimizer = torch.optim.AdamW(primary_model.parameters())

    aux_base_optimizer = torch.optim.AdamW(auxiliary_model.parameters(), lr=aux_lr, weight_decay=aux_wd)
    aux_optimizer = MetaOptimizer(aux_base_optimizer, hpo_lr=aux_lr)

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

            # update primary parameters
            train_loss.backward()
            primary_optimizer.step()

            # condition for updating auxiliary parameters
            if step % hypergrad_every == 0:
                # Compute the main task loss on the meta-validation set
                meta_val_loss = 0
                n_examples = 0
                for _ in range(n_meta_loss_accum):
                    try:
                        meta_batch = next(meta_val_dl_iter)
                    except StopIteration:
                        meta_val_dl_iter = iter(meta_val_dl)
                        meta_batch = next(meta_val_dl_iter)
                    meta_batch = (t.to(device) for t in meta_batch)
                    images, labels = meta_batch
                    model_out = primary_model(images)
                    meta_val_loss += get_single_class_loss(
                        model_out=model_out,
                        labels=labels,
                        i_class=main_task_class,
                    ).sum()  # sum over the batch
                    n_examples += len(images)
                meta_val_loss = meta_val_loss / n_examples  # mean

                #  Compute the meta-train loss
                meta_train_loss = 0
                n_examples = 0
                for _ in range(n_meta_loss_accum):
                    try:
                        meta_batch = next(meta_train_iter)
                    except StopIteration:
                        meta_train_iter = iter(train_loader)
                        meta_batch = next(meta_train_iter)
                    meta_batch = (t.to(device) for t in meta_batch)
                    images, labels = meta_batch
                    model_out = primary_model(images)

                    # get the list of train losses for each task
                    train_losses = get_tasks_losses(model_out, labels)

                    # combine losses using 'auxiliary_model' to get the total train loss
                    meta_train_loss += auxiliary_model(train_losses).sum()  # sum over the batch
                    n_examples += len(images)

                meta_train_loss = meta_train_loss / n_examples  # mean

                # update auxiliary parameters - no need to call loss.backwards() or aux_optimizer.zero_grad()
                aux_optimizer.step(
                    val_loss=meta_val_loss,
                    train_loss=meta_train_loss,
                    aux_params=list(auxiliary_model.parameters()),
                    parameters=list(primary_model.parameters()),
                )
                if isinstance(auxiliary_model, MonoHyperNet):
                    # set all the auxiliary parameters to be non-negative
                    auxiliary_model.clamp()

                if isinstance(auxiliary_model, MonoLinearHyperNet):
                    # print the auxiliary parameters
                    print("Auxiliary weights", auxiliary_model.linear.weight)

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
if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------
