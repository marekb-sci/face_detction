# -*- coding: utf-8 -*-
import math
import torch

def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):

    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    use_lr_scheduler=True):
    model.train()
    model.to(device)

    lr_scheduler = None
    if use_lr_scheduler and epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    losses_history = []
    # for images, targets in metric_logger.log_every(data_loader, print_freq, header):
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses_history.append({k: v.item() for k, v in loss_dict.items()})
        losses_history[-1]['total'] = sum(losses_history[-1].values())
        losses_history[-1]['batch_size'] = len(images)

        losses = sum(loss for loss in loss_dict.values())


        for loss_name, loss_value in loss_dict.items():
            if not math.isfinite(loss_value.item()):
                print(f"Loss {loss_name} is {loss_value}, stopping training")
                print(loss_dict)
                return losses_history

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        # metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    print(sum([ld['total'] for ld in losses_history]))
    return losses_history

