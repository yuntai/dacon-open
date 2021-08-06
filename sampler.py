# https://discuss.pytorch.org/t/how-to-use-my-own-sampler-when-i-already-use-distributedsampler/62143/21
# from https://github.com/catalyst-team/catalyst/blob/master/catalyst/data/sampler.py
from torch.utils.data import DistributedSampler
from torch.utils.data import Dataset, Sampler
from typing import Optional, Iterator, List, Union
from operator import itemgetter
import torch
import torch.distributed as dist
import math
import numpy as np
from collections import Counter

class BalanceClassSampler(Sampler):
    """Allows you to create stratified sample on unbalanced classes.
    Args:
        labels: list of class label for each elem in the dataset
        mode: Strategy to balance classes.
            Must be one of [downsampling, upsampling]
    Python API examples:
    .. code-block:: python
        import os
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from catalyst import dl
        from catalyst.data import ToTensor, BalanceClassSampler
        from catalyst.contrib.datasets import MNIST
        train_data = MNIST(os.getcwd(), train=True, download=True, transform=ToTensor())
        train_labels = train_data.targets.cpu().numpy().tolist()
        train_sampler = BalanceClassSampler(train_labels, mode=5000)
        valid_data = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
        loaders = {
            "train": DataLoader(train_data, sampler=train_sampler, batch_size=32),
            "valid": DataLoader(valid_data, batch_size=32),
        }
        model = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 10))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.02)
        runner = dl.SupervisedRunner()
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=1,
            logdir="./logs",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
        )
    """

    def __init__(self, labels: List[int], mode: Union[str, int] = "downsampling"):
        """Sampler initialisation."""
        super().__init__(labels)

        labels = np.array(labels)
        samples_per_class = {label: (labels == label).sum() for label in set(labels)}

        self.lbl2idx = {
            label: np.arange(len(labels))[labels == label].tolist() for label in set(labels)
        }

        if isinstance(mode, str):
            assert mode in ["downsampling", "upsampling"]

        if isinstance(mode, int) or mode == "upsampling":
            samples_per_class = mode if isinstance(mode, int) else max(samples_per_class.values())
        else:
            samples_per_class = min(samples_per_class.values())

        self.labels = labels
        self.samples_per_class = samples_per_class
        self.length = self.samples_per_class * len(set(labels))

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.lbl2idx):
            replace_flag = self.samples_per_class > len(self.lbl2idx[key])
            indices += np.random.choice(
                self.lbl2idx[key], self.samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)

        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length

class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


# From
# https://github.com/catalyst-team/catalyst/blob/146885c62f93f1feef360a85f5d4b37b954bcdcb/catalyst/data/dataset/torch.py#L12

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

class DynamicBalanceClassSampler(Sampler):
    """
    This kind of sampler can be used for classification tasks with significant
    class imbalance.
    The idea of this sampler that we start with the original class distribution
    and gradually move to uniform class distribution like with downsampling.
    Let's define :math: D_i = #C_i/ #C_min where :math: #C_i is a size of class
    i and :math: #C_min is a size of the rarest class, so :math: D_i define
    class distribution. Also define :math: g(n_epoch) is a exponential
    scheduler. On each epoch current :math: D_i  calculated as
    :math: current D_i  = D_i ^ g(n_epoch),
    after this data samples according this distribution.
    Notes:
         In the end of the training, epochs will contain only
         min_size_class * n_classes examples. So, possible it will not
         necessary to do validation on each epoch. For this reason use
         ControlFlowCallback.
    Examples:
        >>> import torch
        >>> import numpy as np
        >>> from catalyst.data import DynamicBalanceClassSampler
        >>> from torch.utils import data
        >>> features = torch.Tensor(np.random.random((200, 100)))
        >>> labels = np.random.randint(0, 4, size=(200,))
        >>> sampler = DynamicBalanceClassSampler(labels)
        >>> labels = torch.LongTensor(labels)
        >>> dataset = data.TensorDataset(features, labels)
        >>> loader = data.dataloader.DataLoader(dataset, batch_size=8)
        >>> for batch in loader:
        >>>     b_features, b_labels = batch
    Sampler was inspired by https://arxiv.org/abs/1901.06783
    """

    def __init__(
        self,
        labels: List[Union[int, str]],
        exp_lambda: float = 0.9,
        start_epoch: int = 0,
        max_d: Optional[int] = None,
        mode: Union[str, int] = "downsampling",
        ignore_warning: bool = False,
    ):
        """
        Args:
            labels: list of labels for each elem in the dataset
            exp_lambda: exponent figure for schedule
            start_epoch: start epoch number, can be useful for multi-stage
            experiments
            max_d: if not None, limit on the difference between the most
            frequent and the rarest classes, heuristic
            mode: number of samples per class in the end of training. Must be
            "downsampling" or number. Before change it, make sure that you
            understand how does it work
            ignore_warning: ignore warning about min class size
        """
        assert isinstance(start_epoch, int)
        assert 0 < exp_lambda < 1, "exp_lambda must be in (0, 1)"
        super().__init__(labels)
        self.exp_lambda = exp_lambda
        if max_d is None:
            max_d = np.inf
        self.max_d = max_d
        self.epoch = start_epoch
        labels = np.array(labels)
        samples_per_class = Counter(labels)
        self.min_class_size = min(samples_per_class.values())

        if self.min_class_size < 100 and not ignore_warning:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"the smallest class contains only"
                f" {self.min_class_size} examples. At the end of"
                f" training, epochs will contain only"
                f" {self.min_class_size * len(samples_per_class)}"
                f" examples"
            )

        self.original_d = {
            key: value / self.min_class_size for key, value in samples_per_class.items()
        }
        self.label2idxes = {
            label: np.arange(len(labels))[labels == label].tolist() for label in set(labels)
        }

        if isinstance(mode, int):
            self.min_class_size = mode
        else:
            assert mode == "downsampling"

        self.labels = labels
        self._update()

    def _update(self) -> None:
        """
        Update d coefficients
        Returns: None
        """
        current_d = {
            key: min(value ** self._exp_scheduler(), self.max_d)
            for key, value in self.original_d.items()
        }
        samples_per_classes = {
            key: int(value * self.min_class_size) for key, value in current_d.items()
        }
        self.samples_per_classes = samples_per_classes
        self.length = np.sum(list(samples_per_classes.values()))
        self.epoch += 1

    def _exp_scheduler(self) -> float:
        return self.exp_lambda ** self.epoch

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            indices of stratified sample
        """
        indices = []
        for key in sorted(self.label2idxes):
            samples_per_class = self.samples_per_classes[key]
            replace_flag = samples_per_class > len(self.label2idxes[key])
            indices += np.random.choice(
                self.label2idxes[key], samples_per_class, replace=replace_flag
            ).tolist()
        assert len(indices) == self.length
        np.random.shuffle(indices)
        self._update()
        return iter(indices)

    def __len__(self) -> int:
        """
        Returns:
             length of result sample
        """
        return self.length
