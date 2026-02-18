#!/usr/bin/env python3
"""Launcher de entrenamiento para GATrAutoRegressor con PyTorch Lightning."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from src.datamodule.pf_autoregressor_dataset import (
    PFPreprocessor,
    PFPreprocessorConfig,
    make_pf_splits,
)
from src.model.GATrAutoRegressor import (
    GATrAutoRegressive,
    GATrAutoRegressor,
    GATrAutoRegressorParamsSplit,
    GATrAutoRegressorParamsWhole,
)
from src.model.gatr_autoregressor_module import GATrAutoRegressorLightningModule


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(cfg)
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg.setdefault("trainer", {})
    cfg.setdefault("wandb", {})

    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.dataset_paths:
        cfg["data"]["dataset_paths"] = args.dataset_paths
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.val_ratio is not None:
        cfg["data"]["val_ratio"] = args.val_ratio
    if args.data_mode is not None:
        cfg["data"]["mode"] = args.data_mode
    if args.pin_memory is not None:
        cfg["data"]["pin_memory"] = args.pin_memory

    if args.model_mode is not None:
        cfg["model"]["mode"] = args.model_mode

    if args.max_epochs is not None:
        cfg["trainer"]["max_epochs"] = args.max_epochs
    if args.lr is not None:
        cfg["trainer"]["lr"] = args.lr
    if args.weight_decay is not None:
        cfg["trainer"]["weight_decay"] = args.weight_decay
    if args.precision is not None:
        cfg["trainer"]["precision"] = args.precision
    if args.accumulate_grad_batches is not None:
        cfg["trainer"]["accumulate_grad_batches"] = args.accumulate_grad_batches
    if args.gradient_clip_val is not None:
        cfg["trainer"]["gradient_clip_val"] = args.gradient_clip_val

    if args.checkpoint_dir is not None:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.resume_from is not None:
        cfg["resume_from"] = args.resume_from

    if args.wandb_project is not None:
        cfg["wandb"]["project"] = args.wandb_project
    if args.wandb_run_name is not None:
        cfg["wandb"]["run_name"] = args.wandb_run_name
    if args.wandb_offline is not None:
        cfg["wandb"]["offline"] = args.wandb_offline
    if args.no_wandb:
        cfg["wandb"]["disabled"] = True
    
    if args.devices:
        cfg["trainer"]["devices"] = args.devices

    return cfg


def ns(d: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**d)


def build_model(model_cfg: Dict[str, Any], trainer_cfg: Optional[Dict[str, Any]] = None) -> GATrAutoRegressor:
    mode = model_cfg.get("mode", "whole_detector")

    whole_cfg = GATrAutoRegressorParamsWhole(
        hidden_mv=model_cfg.get("hidden_mv", 32),
        hidden_s=model_cfg.get("hidden_s", 64),
        num_blocks=model_cfg.get("num_blocks", 3),
        in_s_channels=model_cfg.get("in_s_channels", 6),
        in_mv_channels=model_cfg.get("in_mv_channels", 1),
        out_mv_channels=model_cfg.get("out_mv_channels", 1),
        dropout=model_cfg.get("dropout", 0.1),
        out_s_channels=model_cfg.get("out_s_channels", None),
    )

    split_out_s = model_cfg.get("split_out_s_channels", [None, None, None, None])
    split_hidden_s = model_cfg.get("split_hidden_s", [64, 64, 64, 64])
    inferred_final_in_s = split_out_s[0] if split_out_s[0] is not None else split_hidden_s[0]

    split_cfg = GATrAutoRegressorParamsSplit(
        hidden_mv=model_cfg.get("split_hidden_mv", [32, 32, 32, 32]),
        hidden_s=model_cfg.get("split_hidden_s", [64, 64, 64, 64]),
        num_blocks=model_cfg.get("split_num_blocks", [3, 3, 3, 3]),
        in_s_channels=model_cfg.get("split_in_s_channels", [6, 6, 6, 6]),
        in_mv_channels=model_cfg.get("split_in_mv_channels", [1, 1, 1, 1]),
        out_mv_channels=model_cfg.get("split_out_mv_channels", [1, 1, 1, 1]),
        dropout=model_cfg.get("split_dropout", [0.1, 0.1, 0.1, 0.1]),
        out_s_channels=split_out_s,
        final_module_cfg=GATrAutoRegressorParamsWhole(
            hidden_mv=model_cfg.get("split_final_hidden_mv", model_cfg.get("hidden_mv", 32)),
            hidden_s=model_cfg.get("split_final_hidden_s", model_cfg.get("hidden_s", 64)),
            num_blocks=model_cfg.get("split_final_num_blocks", model_cfg.get("num_blocks", 3)),
            in_s_channels=model_cfg.get("split_final_in_s_channels", inferred_final_in_s),
            in_mv_channels=model_cfg.get("split_final_in_mv_channels", 1),
            out_mv_channels=model_cfg.get("split_final_out_mv_channels", 1),
            dropout=model_cfg.get("split_final_dropout", model_cfg.get("dropout", 0.1)),
            out_s_channels=model_cfg.get("split_final_out_s_channels", model_cfg.get("out_s_channels", None)),
        ),
    )

    ar_cfg = GATrAutoRegressive(
        hidden_mv=model_cfg.get("ar_hidden_mv", 32),
        hidden_s=model_cfg.get("ar_hidden_s", 64),
        num_blocks=model_cfg.get("ar_num_blocks", 2),
        out_mv_channels=model_cfg.get("ar_out_mv_channels", 1),
        out_s_channels=model_cfg.get("ar_out_s_channels", None),
        dropout=model_cfg.get("ar_dropout", 0.1),
    )

    params_cfg: Dict[str, Any] = {
        "autoregressive_module": ar_cfg,
        "max_steps": model_cfg.get("max_steps", 128),
        "max_ar_steps_train": model_cfg.get(
            "max_ar_steps_train",
            (trainer_cfg or {}).get("max_ar_steps_train", None),
        ),
        "debug_memory": model_cfg.get(
            "debug_memory",
            (trainer_cfg or {}).get("debug_memory", False),
        ),
        "debug_memory_interval": model_cfg.get(
            "debug_memory_interval",
            (trainer_cfg or {}).get("debug_memory_interval", 5),
        ),
    }
    if mode == "whole_detector":
        params_cfg["whole_detector"] = whole_cfg
    elif mode == "detector_split":
        params_cfg["detector_split"] = split_cfg
    else:
        raise ValueError(f"model.mode inválido: {mode}")

    return GATrAutoRegressor(mode=mode, params_cfg=params_cfg)


def build_preprocessor(pre_cfg: Dict[str, Any]) -> PFPreprocessor:
    cfg = PFPreprocessorConfig(
        normalize_coords=pre_cfg.get("normalize_coords", True),
        normalize_energy=pre_cfg.get("normalize_energy", True),
        normalize_momentum=pre_cfg.get("normalize_momentum", True),
        log_energy=pre_cfg.get("log_energy", True),
        log_momentum=pre_cfg.get("log_momentum", True),
        log_eps=pre_cfg.get("log_eps", 1e-6),
        normalize_pfo_momentum=pre_cfg.get("normalize_pfo_momentum", False),
        log_pfo_energy=pre_cfg.get("log_pfo_energy", True),
    )
    return PFPreprocessor(cfg)


def build_dataloaders(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]
    preprocessor = build_preprocessor(cfg.get("preprocessor", {}))

    train_ds, val_ds = make_pf_splits(
        paths=data_cfg["dataset_paths"],
        val_ratio=data_cfg.get("val_ratio", 0.1),
        mode=data_cfg.get("mode", "memory"),
        preprocessor=preprocessor,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("num_workers", 4) > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        persistent_workers=data_cfg.get("num_workers", 4) > 0,
    )
    return train_loader, val_loader


def run_single_forward(
    module: GATrAutoRegressorLightningModule,
    train_loader: DataLoader,
    device: torch.device,
) -> None:
    batch = next(iter(train_loader)).to(device)
    module.to(device)
    module.eval()
    module.model.eval()

    with torch.no_grad():
        mv_v_part, mv_s_part, scalars, hit_batch, pfo_true_objects = module._prepare_batch(batch)
        output = module.model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=hit_batch,
        )
        if output["pfo_momentum"] is None:
            raise RuntimeError("Forward ejecutado, pero no se generaron PFOs.")

        losses = module.loss_fn(output, pfo_true_objects, hit_batch[0])

    print("Forward OK.")
    print(f"- pfo_momentum: {tuple(output['pfo_momentum'].shape)}")
    print(f"- pfo_pid: {tuple(output['pfo_pid'].shape)}")
    print(f"- assignments: {tuple(output['assignments'].shape)}")
    print(f"- stop_probs: {tuple(output['stop_probs'].shape)}")
    print(f"- loss: {losses['loss'].item():.6f}")


def make_logger(cfg: Dict[str, Any]):
    wandb_cfg = cfg.get("wandb", {})
    if wandb_cfg.get("disabled", False):
        return CSVLogger("logs", name="gatr_autoregressor")

    try:
        from lightning.pytorch.loggers import WandbLogger
    except Exception:
        return CSVLogger("logs", name="gatr_autoregressor")

    return WandbLogger(
        project=wandb_cfg.get("project", "gatr-autoregressor"),
        name=wandb_cfg.get("run_name", "baseline-v1"),
        log_model=wandb_cfg.get("log_model", False),
        offline=wandb_cfg.get("offline", False),
    )


def make_callbacks(cfg: Dict[str, Any]):
    trainer_cfg = cfg["trainer"]
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints/gatr_autoregressor")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    callbacks: List[Any] = []
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="epoch{epoch:02d}-step{step}",
            monitor=trainer_cfg.get("early_stopping_metric", "val/loss"),
            mode="min",
            save_top_k=trainer_cfg.get("save_top_k", 3),
            save_last=True,
        )
    )
    patience = trainer_cfg.get("early_stopping_patience", 20)
    if patience and patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=trainer_cfg.get("early_stopping_metric", "val/loss"),
                mode="min",
                patience=patience,
            )
        )
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenamiento Lightning para GATrAutoRegressor")
    parser.add_argument("--config", default="configs/gatr_autoregressor_default.yaml")
    parser.add_argument("--test", action="store_true", help="Ejecuta solo un forward de verificación")

    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset-paths", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--data-mode", choices=["memory", "lazy"])
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    # parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=None)

    parser.add_argument("--model-mode", choices=["whole_detector", "detector_split"])

    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--precision")
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument("--gradient-clip-val", type=float)

    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", nargs="+", type=int, help="Device IDs (e.g., 0 1 2 3)")
    parser.add_argument("--strategy")

    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--resume-from")

    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-offline", dest="wandb_offline", action="store_true")
    parser.add_argument("--wandb-online", dest="wandb_offline", action="store_false")
    parser.set_defaults(wandb_offline=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_yaml(args.config), args)

    if not cfg["data"].get("dataset_paths"):
        raise ValueError("No hay dataset_paths en config ni por línea de comandos.")

    L.seed_everything(cfg.get("seed", 42), workers=True)

    model = build_model(cfg["model"], cfg.get("trainer", {}))
    trainer_cfg = ns(cfg["trainer"])
    module = GATrAutoRegressorLightningModule(
        model=model,
        cfg=trainer_cfg,
        plot_every_n_steps=cfg["trainer"].get("plot_every_n_steps", 50),
    )
    train_loader, val_loader = build_dataloaders(cfg)

    if args.test:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        run_single_forward(module, train_loader, device)
        return
    print("Configuración de entrenamiento:")
    print(cfg["trainer"]["devices"])
    print(cfg["trainer"]["strategy"])
    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=cfg["trainer"].get("devices", "auto"),
        strategy=cfg["trainer"].get("strategy", "auto"),
        max_epochs=cfg["trainer"].get("max_epochs", 100),
        precision=cfg["trainer"].get("precision", "32"),
        accumulate_grad_batches=cfg["trainer"].get("accumulate_grad_batches", 1),
        gradient_clip_val=cfg["trainer"].get("gradient_clip_val", 0.0),
        gradient_clip_algorithm=cfg["trainer"].get("gradient_clip_algorithm", "norm"),
        log_every_n_steps=cfg["trainer"].get("log_every_n_steps", 10),
        num_sanity_val_steps=cfg["trainer"].get("num_sanity_val_steps", 2),
        callbacks=make_callbacks(cfg),
        logger=make_logger(cfg),
    )

    trainer.fit(
        model=module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.get("resume_from", None),
    )


if __name__ == "__main__":
    main()
