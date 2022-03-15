from enum import Enum, auto

from torch.utils.data import DataLoader


class LoopPhase(Enum):
    TRAIN = auto()
    VALIDATION = auto()


class Phase:

    def __init__(self, type_: LoopPhase, dataloader: DataLoader, is_train: bool = True) -> None:
        self.type_ = type_
        self.dataloader = dataloader
        self.is_train = is_train
