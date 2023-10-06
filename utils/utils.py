import torch


def count_params(model, logger):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            print(p.name)
    logger.info(f"Total params: {total_params}")
    logger.info(f"Trainable params: {trainable_params}")