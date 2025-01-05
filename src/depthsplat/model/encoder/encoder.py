from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from torch import nn

from ...dataset.types import BatchedViews, DataShim
from ..types import Gaussians

T = TypeVar("T")

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf



class Encoder(nn.Module, ABC, Generic[T]):
    cfg: T

    def __init__(self, cfg: T) -> None:
        super().__init__()
        self.cfg = cfg

    @abstractmethod
    def forward(
        self,
        context: BatchedViews,
        deterministic: bool,
    ) -> Gaussians:
        pass

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x

