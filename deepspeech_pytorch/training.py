import json
import random

import numpy as np
import pytorch_lightning as pl
import torch.utils.data.distributed
from hydra.utils import to_absolute_path

from deepspeech_pytorch.checkpoint import FileCheckpointHandler
from deepspeech_pytorch.configs.train_config import DeepSpeechConfig
from deepspeech_pytorch.loader.data_module import DeepSpeechDataModule
from deepspeech_pytorch.logging import DeepSpeechTrainsLogger
from deepspeech_pytorch.model import DeepSpeech


def train(cfg: DeepSpeechConfig):
    # Set seeds for determinism
    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)

    with open(to_absolute_path(cfg.data.labels_path)) as label_file:
        labels = json.load(label_file)

    checkpoint_callback = FileCheckpointHandler(
        cfg=cfg.checkpointing
    )
    if cfg.checkpointing.load_auto_checkpoint:
        latest_checkpoint = checkpoint_callback.find_latest_checkpoint()
        if latest_checkpoint:
            cfg.checkpointing.continue_from = latest_checkpoint

    data_loader = DeepSpeechDataModule(
        labels=labels,
        data_cfg=cfg.data,
        normalize=True,
        multigpu=cfg.training.multigpu
    )

    model = DeepSpeech(
        labels=labels,
        model_cfg=cfg.model,
        optim_cfg=cfg.optim,
        precision=cfg.training.precision,
        spect_cfg=cfg.data.spect
    )

    if cfg.viz.trains:
        logger = DeepSpeechTrainsLogger(
            project_name=cfg.viz.project_name,
            task_name=cfg.viz.task_name,
            auto_connect_arg_parser=False,
            auto_connect_frameworks=False
        )
    else:
        logger = None

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        gpus=cfg.training.gpus,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=cfg.checkpointing.continue_from if cfg.checkpointing.continue_from else None,
        precision=cfg.training.precision.value,
        gradient_clip_val=cfg.optim.max_norm,
        replace_sampler_ddp=False,
        distributed_backend=cfg.training.multigpu.value
    )
    trainer.fit(model, data_loader)