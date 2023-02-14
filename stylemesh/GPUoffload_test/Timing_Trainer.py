"""Trainer to automate the training."""
import logging
import sys
import time
import warnings
from typing import Any, Dict, Generator, Iterable, List, Optional, Type, Union
sys.path.append(".")

log = logging.getLogger(__name__)
# warnings to ignore in trainer
warnings.filterwarnings(
    "ignore", message="torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead"
)

from pytorch_lightning.trainer import Trainer

class Timing_Trainer(Trainer):

    def _run(self, model: "pl.LightningModule", ckpt_path: Optional[str] = None):
        start_time = time.time()
        super()._run(model)
        return time.time() - start_time