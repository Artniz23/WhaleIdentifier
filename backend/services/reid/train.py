import os
import pandas as pd
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
import argparse
from typing import Optional
import torch
from types import SimpleNamespace

from services.reid.classifier import SphereClassifier
from services.reid.config import Config, load_config
from services.reid.data_module import WhaleDataModule
from services.reid.utils import load_df


def train(
        df: pd.DataFrame,
        args: argparse.Namespace,
        cfg: Config,
        fold: int,
        do_inference: bool = True,
) -> Optional[float]:
    out_dir = f"{args.out_base_dir}/{args.exp_name}/{fold}"
    id_class_nums = df.individual_id.value_counts().sort_index().values
    model = SphereClassifier(cfg, id_class_nums=id_class_nums, backbone_pretrained=cfg.pretrained)
    data_module = WhaleDataModule(
        df, cfg, f"{args.in_base_dir}/train_images", fold
    )
    loggers = [pl_loggers.CSVLogger(out_dir)]
    if args.wandb_logger:
        loggers.append(
            pl_loggers.WandbLogger(
                project="kaggle-happywhale", group=args.exp_name, name=f"{args.exp_name}/{fold}", save_dir=out_dir
            )
        )
    callbacks = [LearningRateMonitor("epoch"), TQDMProgressBar(refresh_rate=10), ]
    if args.save_checkpoint:
        callbacks.append(ModelCheckpoint(
            dirpath=out_dir,
            monitor="train/mapNone",
            mode="max",
            save_last=True,
            save_top_k=1,
            verbose=True,
        ))
    trainer = Trainer(
        accelerator="gpu",
        max_epochs=cfg["max_epochs"],
        logger=loggers,
        callbacks=callbacks,
        enable_checkpointing=args.save_checkpoint,
        precision=32,
        sync_batchnorm=True,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    ckpt_path = f"{out_dir}/last.ckpt"
    if not os.path.exists(ckpt_path) or not args.load_snapshot:
        ckpt_path = None
    trainer.fit(model, ckpt_path=ckpt_path, datamodule=data_module)
    if do_inference:
        # all train data
        model.test_results_fp = f"{out_dir}/train_results.npz"
        trainer.test(model, data_module.all_dataloader())

        # custom test data
        test_csv_name = "test.csv"

        model.test_results_fp = f"{out_dir}/test_results.npz"
        df_test = load_df(args.in_base_dir, cfg, test_csv_name)
        test_data_module = WhaleDataModule(
            df_test,
            cfg,
            f"{args.in_base_dir}/test_images",
            -1
        )
        trainer.test(model, test_data_module.all_dataloader())

    if args.wandb_logger:
        wandb.finish()
    else:
        return None

torch.set_float32_matmul_precision('medium')

argsObj = SimpleNamespace(
    out_base_dir="efficient_b7_results",
    in_base_dir="kaggle_reid_dataset",
    exp_name="b7",
    load_snapshot=False,
    save_checkpoint=True,
    wandb_logger=False,
    config_path="reid_config/efficientnet_b7.yaml",
)

cfg = load_config(argsObj.config_path, "reid_config/default.yaml")
df = load_df(argsObj.in_base_dir, cfg, "train.csv")
if cfg["n_splits"] == -1:
    train(df, argsObj, cfg, -1, do_inference=True)
else:
    train(df, argsObj, cfg, 0, do_inference=True)