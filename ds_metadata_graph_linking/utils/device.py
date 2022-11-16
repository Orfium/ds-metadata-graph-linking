import torch


class Devices:
    GPU = 'cuda'
    CPU = 'cpu'


def decide_device(device):
    if device == Devices.GPU and torch.cuda.is_available():
        return device

    return Devices.CPU
