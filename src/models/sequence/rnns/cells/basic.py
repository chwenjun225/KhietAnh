"""Baseline simple RNN cells such as the vanilla RNN and GRU."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.nn import LinearActivation, Activation # , get_initializer
from src.models.nn.gate import Gate
from src.models.nn.orthogonal import OrthogonalLinear
from src.models.sequence.base import SequenceModule

