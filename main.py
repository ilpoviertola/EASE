# ---------------------------------------------------------------
# © 2025 Ilpo Viertola. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from PyTorch Lightning,
# used under the Apache 2.0 License and from tue-mps/EoMT,
# used under the MIT License.
# ---------------------------------------------------------------

import warnings
import logging
from types import MethodType

import jsonargparse._typehints as _t
from gitignore_parser import parse_gitignore
import torch
from lightning.pytorch import cli
from lightning.pytorch.callbacks import (
    ModelSummary,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher

from training.lightning_module import LightningModule
from datasets.lightning_data_module import LightningDataModule

_orig_single = _t.raise_unexpected_value


def _raise_single(*args, exception=None, **kwargs):
    if isinstance(exception, Exception):
        raise exception
    return _orig_single(*args, exception=exception, **kwargs)


_orig_union = _t.raise_union_unexpected_value


def _raise_union(subtypes, val, vals):
    for e in reversed(vals):
        if isinstance(e, Exception):
            raise e
    return _orig_union(subtypes, val, vals)


_t.raise_unexpected_value = _raise_single
_t.raise_union_unexpected_value = _raise_union


def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            # added below to check val based on global steps instead of batches in case of iteration based val check and gradient accumulation
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


class LightningCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^The ``compute`` method of metric PanopticQuality was called before the ``update`` method.*",
        )
        warnings.filterwarnings(
            "ignore", message=r"^Grad strides do not match bucket view strides.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*functools.partial will be a method descriptor in future Python versions*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^Online softmax is disabled on the fly since Inductor decides to.*",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"^UserWarning: No device id is provided via*",
        )

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")

        parser.link_arguments(
            "data.init_args.num_classes", "model.init_args.num_classes"
        )
        parser.link_arguments(
            "data.init_args.num_classes",
            "model.init_args.network.init_args.num_classes",
        )

        parser.link_arguments(
            "data.init_args.stuff_classes", "model.init_args.stuff_classes"
        )

        parser.link_arguments("data.init_args.img_size", "model.init_args.img_size")
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.network.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        parser.link_arguments(
            "model.init_args.network.init_args.num_q",
            "model.init_args.network.init_args.query_generator.init_args.query_num",
        )
        parser.link_arguments(
            "data.init_args.read_per_frame", "model.init_args.per_frame"
        )
        parser.link_arguments(
            "model.init_args.network.init_args.rope_seq_len",
            "model.init_args.imgs_per_audio",
        )
        parser.link_arguments(
            "model.init_args.network.init_args.audio_enhance.init_args.embed_dim",
            "model.init_args.network.init_args.query_generator.init_args.embed_dim",
        )
        parser.link_arguments(
            "model.init_args.network.init_args.audio_enhance.init_args.embed_dim",
            "model.init_args.network.init_args.fuser.init_args.aud_dim",
        )
        parser.link_arguments(
            "model.init_args.network.init_args.audio_enhance.init_args.embed_dim",
            "model.init_args.network.init_args.fuser.init_args.audio_dim",
        )

    def fit(self, model, **kwargs):
        if hasattr(self.trainer.logger.experiment, "log_code"):
            is_gitignored = parse_gitignore(".gitignore")
            include_fn = lambda path: path.endswith(".py") or path.endswith(".yaml")
            self.trainer.logger.experiment.log_code(
                ".", include_fn=include_fn, exclude_fn=is_gitignored
            )

        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        if model.per_frame:
            assert (
                model.network.rope_seq_len == 1
            ), "When per_frame is True, rope_seq_len must be 1."

        self.trainer.fit(model, **kwargs)

    def validate(self, model, **kwargs):
        if model.network.masked_attn_enabled:
            print("Disabling masked attention for validation.")
            model.network.masked_attn_enabled = False
            model.metrics = model.metrics[:1]

        self.trainer.validate(model, **kwargs)

    def test(self, model, **kwargs):
        if model.network.masked_attn_enabled:
            print("Disabling masked attention for testing.")
            model.network.masked_attn_enabled = False
            model.metrics = model.metrics[:1]

        self.trainer.test(model, **kwargs)


def cli_main():
    LightningCLI(
        LightningModule,
        LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=None,
        seed_everything_default=0,
        trainer_defaults={
            "precision": "16-mixed",
            "enable_model_summary": False,
            "callbacks": [
                ModelSummary(max_depth=3),
                LearningRateMonitor(logging_interval="epoch"),
                ModelCheckpoint(
                    monitor="metrics/val_f_measure",
                    mode="max",
                    save_top_k=1,
                    filename="best-f-epoch={epoch}-val_f_measure={metrics/val_f_measure:.2f}-val_miou={metrics/val_iou_all:.2f}",
                    auto_insert_metric_name=False,
                    save_last=True,
                ),
                ModelCheckpoint(
                    monitor="metrics/val_iou_all",
                    mode="max",
                    save_top_k=1,
                    filename="best-miou-epoch={epoch}-val_f_measure={metrics/val_f_measure:.2f}-val_miou={metrics/val_iou_all:.2f}",
                    auto_insert_metric_name=False,
                    save_last=False,
                ),
            ],
            "devices": 1,
            "gradient_clip_val": 0.01,
            "gradient_clip_algorithm": "norm",
        },
    )


if __name__ == "__main__":
    cli_main()
