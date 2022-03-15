from torch.utils.data import DataLoader


class Phase:

    def __init__(self, dataloader: DataLoader, is_train: bool = True) -> None:
        self.dataloader = dataloader
        self.is_train = is_train
