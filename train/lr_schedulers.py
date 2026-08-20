import torch


SCHEDULER_NAMES = {
    "exponential",
    "constant",
    "step",
    "multistep",
    "linear",
    "polynomial",
    "cosine",
    "cosine_restarts",
    "onecycle",
    "cyclic",
}


def create_lr_scheduler(
    optimizer,
    name,
    total_epochs,
    steps_per_epoch,
    lr_decay=0.999875,
    min_lr_ratio=0.01,
):
    name = str(name).lower()
    if name not in SCHEDULER_NAMES:
        raise ValueError(
            "Unknown LR scheduler %r; expected one of: %s"
            % (name, ", ".join(sorted(SCHEDULER_NAMES)))
        )
    total_epochs = max(int(total_epochs), 1)
    steps_per_epoch = max(int(steps_per_epoch), 1)
    base_lr = float(optimizer.param_groups[0]["lr"])
    min_lr_ratio = float(min_lr_ratio)
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be between 0 and 1")

    if name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(lr_decay)
        ), "epoch"
    if name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=total_epochs
        ), "epoch"
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(total_epochs // 3, 1), gamma=0.5
        ), "epoch"
    if name == "multistep":
        milestones = sorted(
            {
                max(round(total_epochs * 0.4), 1),
                max(round(total_epochs * 0.7), 1),
                max(round(total_epochs * 0.9), 1),
            }
        )
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5
        ), "epoch"
    if name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=max(min_lr_ratio, 1e-8),
            total_iters=total_epochs,
        ), "epoch"
    if name == "polynomial":
        return torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=total_epochs, power=0.9
        ), "epoch"
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=base_lr * min_lr_ratio,
        ), "epoch"
    if name == "cosine_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(total_epochs // 5, 1),
            T_mult=2,
            eta_min=base_lr * min_lr_ratio,
        ), "epoch"

    total_steps = total_epochs * steps_per_epoch
    if name == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
            cycle_momentum=False,
        ), "batch"
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr * min_lr_ratio,
        max_lr=base_lr,
        step_size_up=max(total_steps // 5, 1),
        step_size_down=max(total_steps // 5, 1),
        mode="triangular2",
        cycle_momentum=False,
    ), "batch"
