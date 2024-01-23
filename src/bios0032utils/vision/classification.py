import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def train(dataloader, model, criterion, optimizer, device="cpu"):
    """
    This routine correspond to the steps taken for every epoch in the training
    schedule.

    Input: dataloader, model, criterion, optimizer and device to run model on

    Return the Average Loss and Top-1 Accuracy for the images in the given
    batch.
    """
    model.train()
    loss_list = []
    top1_list = []

    ## Loop is commented out in student version
    for data in dataloader:
        optimizer.zero_grad()
        inputs = data["img"].to(device)
        targets = data["target"].to(device)

        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()  # backward step - propagate loss
        optimizer.step()  # step

        # measure top-1 accuracy and record loss
        prec1 = accuracy(output.data, targets)[0]
        loss_list.append(loss.item())
        top1_list.append(prec1.item())

    print(
        "Train. Prec@1 {top1:.3f}\nTrain. Loss {loss:.3f}\n".format(
            top1=np.mean(top1_list), loss=np.mean(loss_list)
        )
    )

    return np.mean(loss_list), np.mean(top1_list)


@torch.no_grad()
def validate(dataloader, model, criterion, device="cpu", split="val"):
    """
    This routine correspond to the steps taken during the validation phase of a
    trained model or a model under training. Note that the gradients should not
    be updated here.

    Input: dataloader, model, criterion and device to run model on and split
    for reporting purposes (can be val/test)

    Return the Average Loss, Top-1 Accuracy for the images in the given batch
    along with the predictions. Returning predictions is particularly useful in
    the test phase when we want to dig deeper in the evaluation of the model's
    performance.
    """
    # switch to evaluate mode
    model.eval()
    loss_list = []
    top1_list = []
    predictions = []

    ## Loop is commented out in student version
    for data in dataloader:
        inputs = data["img"].to(device)
        targets = data["target"].to(device)

        output = model(inputs)
        loss = criterion(output, targets)

        # measure top-1 accuracy and record loss
        prec1 = accuracy(output.data, targets)[0]
        loss_list.append(loss.item())
        top1_list.append(prec1.item())
        # keep most probable outputs for each image as our predictions
        predictions = (
            predictions + output.max(1).indices.cpu().numpy().tolist()
        )

    print(
        "{split} prec@1 {top1:.3f}\n{split} loss {loss:.3f}\n".format(
            split=split.capitalize(),
            top1=np.mean(top1_list),
            loss=np.mean(loss_list),
        )
    )

    return np.mean(loss_list), np.mean(top1_list), predictions


def accuracy(output, targets, topk=(1,)):
    """Computes the precision@k for the specified values of k.
    Helper util to be used during training/validation.
    Input is model outputs and targets in tensor format"""
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = (
            correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        )
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def plot_confusion_matrix(plot_df):
    plt.figure(figsize=(12, 12))
    sub_df = (
        plot_df.groupby(["species", "predicted_species"])
        .img_path.count()
        .reset_index()
    )
    sub_df.rename(columns={"img_path": "correct_count"}, inplace=True)
    species_without_prediction = set(sub_df.species) - set(
        sub_df.predicted_species
    )
    for sp in species_without_prediction:
        sub_df = pd.concat(
            [
                sub_df,
                pd.DataFrame(
                    [[sp, sp, np.nan]],
                    columns=["species", "predicted_species", "correct_count"],
                ),
            ]
        )
    species_count_norm = (
        plot_df.groupby("species").img_path.count().reset_index()
    )
    species_count_norm.rename(
        columns={"img_path": "total_count"}, inplace=True
    )
    sub_df = sub_df.merge(
        species_count_norm, left_on="species", right_on="species", how="left"
    )
    sub_df["correct_perc"] = 100 * (sub_df.correct_count / sub_df.total_count)
    sub_df = sub_df.pivot("species", "predicted_species", "correct_perc")
    ax = sns.heatmap(sub_df, cmap="RdYlGn", annot=True, fmt=".2f")
    ax.set_title("Species Classification Confusion Matrix", size=20)
    ax.set_xlabel("Predicted Species", size=20)
    ax.set_ylabel("Actual Species", size=20)
    plt.setp(ax.get_yticklabels(), fontsize=10)  # type: ignore
    plt.setp(ax.get_xticklabels(), fontsize=10, rotation=90)  # type: ignore
    plt.show()
