"""
Lightning Module para entrenamiento de GATrAutoRegressor.

Este módulo implementa:
- Losses para momentum (dirección + magnitud), PID, charge, assignment y stop
- Máscara de validez para manejar eventos con diferente número de PFOs
- Logging de métricas y visualizaciones
- Integración con PFAutoRegressorDataset

IMPORTANTE sobre índices:
- pfo_event_idx: índice de evento en el batch (0, 0, 1, 1, 2, ...) - PyG lo incrementa
- hit_to_pfo: índice LOCAL de PFO (0, 1, 2, 0, 1, ...) - NO se incrementa
  porque el modelo autoregresivo compara con step_idx que va 0, 1, 2, ...
"""

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any, Tuple
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.model.GATrAutoRegressor import GATrAutoRegressor


def reorganize_gt_to_tb(
    gt_batch: torch.Tensor,
    gt_values: torch.Tensor,
    T: int,
    B: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reorganiza ground truth de formato flat (N_pfo, ...) a (T, B, ...).
    
    Args:
        gt_batch: (N_pfo,) índice de evento por PFO
        gt_values: (N_pfo, ...) valores a reorganizar
        T: número de steps
        B: número de eventos en batch
        device: dispositivo
    
    Returns:
        output_tb: (T, B, ...) valores reorganizados
        pfo_step_idx: (N_pfo,) índice de step por PFO
    """
    # Calcular índice de step local para cada PFO
    pfo_step_idx = torch.zeros(gt_batch.shape[0], dtype=torch.long, device=device)
    pfo_count_per_event = torch.zeros(B, dtype=torch.long, device=device)
    
    for pfo_idx in range(gt_batch.shape[0]):
        evt = gt_batch[pfo_idx].long()
        pfo_step_idx[pfo_idx] = pfo_count_per_event[evt]
        pfo_count_per_event[evt] += 1
    
    # Crear tensor de salida
    value_shape = gt_values.shape[1:] if gt_values.dim() > 1 else ()
    output_tb = torch.zeros(T, B, *value_shape, device=device, dtype=gt_values.dtype)
    
    # Filtrar PFOs que caben en T steps
    valid_pfo_mask = pfo_step_idx < T
    valid_steps = pfo_step_idx[valid_pfo_mask]
    valid_events = gt_batch[valid_pfo_mask].long()
    
    output_tb[valid_steps, valid_events] = gt_values[valid_pfo_mask]
    
    return output_tb, pfo_step_idx


class GATrAutoRegressorLoss(nn.Module):
    """
    Módulo de loss para GATrAutoRegressor.
    
    Componentes:
    - Momentum direction: 1 - cosine_similarity
    - Momentum magnitude: MSE en log-scale
    - PID: CrossEntropy
    - Charge: MSE
    - Assignment: BCE
    - Stop: BCE
    """
    
    def __init__(
        self,
        lambda_dir: float = 1.0,
        lambda_mag: float = 1.0,
        lambda_pid: float = 1.0,
        lambda_charge: float = 0.5,
        lambda_assign: float = 1.0,
        lambda_stop: float = 0.5,
    ):
        super().__init__()
        self.lambda_dir = lambda_dir
        self.lambda_mag = lambda_mag
        self.lambda_pid = lambda_pid
        self.lambda_charge = lambda_charge
        self.lambda_assign = lambda_assign
        self.lambda_stop = lambda_stop
    
    def forward(
        self,
        output: Dict[str, torch.Tensor],
        pfo_true_objects: Dict[str, torch.Tensor],
        hit_batch: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula todas las losses.
        
        Args:
            output: Salida del modelo con keys:
                - pfo_momentum: (T, B, 3)
                - pfo_p_mod: (T, B, 1)
                - pfo_pid: (T, B, 5)
                - pfo_charge: (T, B, 1)
                - assignments: (T, N, 1)
                - stop_probs: (T, B, 1)
            pfo_true_objects: GT con keys:
                - momentum: (N_pfo, 3)
                - p_mod: (N_pfo, 1) log(|p|) ya preprocesado
                - pid: (N_pfo, 5) one-hot
                - charge: (N_pfo, 1)
                - batch: (N_pfo,) índice de evento en el batch (0,0,1,1,2,...)
                - hit_to_pfo: (N_hits,) índice LOCAL de PFO por hit (0,1,2,0,1,...)
            hit_batch: (N_hits,) índice de evento por hit
        
        Returns:
            Dict con loss total y componentes individuales
        """
        device = output["pfo_momentum"].device
        T, B, _ = output["pfo_momentum"].shape
        N = output["assignments"].shape[1]
        
        gt_batch = pfo_true_objects["batch"]
        gt_momentum = pfo_true_objects["momentum"]
        gt_p_mod = pfo_true_objects["p_mod"]  # Ya en log-scale del preprocesador
        gt_pid = pfo_true_objects["pid"]
        gt_charge = pfo_true_objects["charge"]
        hit_to_pfo = pfo_true_objects["hit_to_pfo"]
        
        # =============================================
        # 1. Construir máscara de validez (T, B)
        # =============================================
        pfos_per_event = torch.bincount(gt_batch.long(), minlength=B)
        step_idx = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        valid_mask = step_idx < pfos_per_event.unsqueeze(0)  # (T, B)
        
        # =============================================
        # 2. Reorganizar GT para alinear con predicciones (T, B, ...)
        # =============================================
        gt_momentum_tb, _ = reorganize_gt_to_tb(gt_batch, gt_momentum, T, B, device)
        gt_p_mod_tb, _ = reorganize_gt_to_tb(gt_batch, gt_p_mod, T, B, device)
        gt_pid_tb, _ = reorganize_gt_to_tb(gt_batch, gt_pid, T, B, device)
        gt_charge_tb, _ = reorganize_gt_to_tb(gt_batch, gt_charge, T, B, device)
        
        # =============================================
        # 3. Loss de dirección del momentum
        # =============================================
        pred_dir = F.normalize(output["pfo_momentum"], dim=-1, eps=1e-8)
        gt_dir = F.normalize(gt_momentum_tb, dim=-1, eps=1e-8)
        
        # Cosine similarity: 1 = perfecta alineación
        cos_sim = (pred_dir * gt_dir).sum(dim=-1)  # (T, B)
        loss_dir = (1 - cos_sim)[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0, device=device)
        
        # =============================================
        # 4. Loss de magnitud del momentum (ambos en log-scale)
        # =============================================
        pred_p_mod = output["pfo_p_mod"].squeeze(-1)  # (T, B) - ya en log-scale
        gt_p_mod_flat = gt_p_mod_tb.squeeze(-1)  # (T, B) - ya en log-scale del preprocesador
        
        # MSE directo ya que ambos están en log-scale
        loss_mag = F.mse_loss(pred_p_mod[valid_mask], gt_p_mod_flat[valid_mask]) if valid_mask.any() else torch.tensor(0.0, device=device)
        
        # =============================================
        # 5. Loss de PID (CrossEntropy)
        # =============================================
        pred_pid = output["pfo_pid"]  # (T, B, 5)
        gt_pid_class = gt_pid_tb.argmax(dim=-1)  # (T, B)
        
        # Flatten para CrossEntropy
        pred_pid_flat = pred_pid[valid_mask]  # (N_valid, 5)
        gt_pid_flat = gt_pid_class[valid_mask]  # (N_valid,)
        
        loss_pid = F.cross_entropy(pred_pid_flat, gt_pid_flat) if valid_mask.any() else torch.tensor(0.0, device=device)
        
        # =============================================
        # 6. Loss de Charge (MSE)
        # =============================================
        pred_charge = output["pfo_charge"].squeeze(-1)  # (T, B)
        gt_charge_flat = gt_charge_tb.squeeze(-1)  # (T, B)
        
        loss_charge = F.mse_loss(pred_charge[valid_mask], gt_charge_flat[valid_mask]) if valid_mask.any() else torch.tensor(0.0, device=device)
        
        # =============================================
        # 7. Loss de Assignment (BCE)
        # =============================================
        # hit_to_pfo contiene índices LOCALES (0, 1, 2, ...) por evento
        # En el paso t, queremos asignar hits cuyo hit_to_pfo == t
        # Esto coincide con cómo el modelo usa step_idx en _update_residual
        
        gt_assignment = torch.zeros(T, N, 1, device=device)
        hit_events = hit_batch.long()
        
        for t in range(T):
            # Un hit pertenece al PFO t si hit_to_pfo == t (índice LOCAL)
            # Solo válido si el evento tiene al menos t+1 PFOs
            valid_for_step = t < pfos_per_event[hit_events]
            gt_assignment[t, :, 0] = ((hit_to_pfo == t) & valid_for_step).float()
        
        pred_assignment = output["assignments_logits"]  # (T, N, 1)
        
        # Crear máscara de validez para assignments
        # Un hit es válido en step t si su evento tiene al menos t+1 PFOs
        step_idx_assign = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        assignment_valid_mask = step_idx_assign < pfos_per_event[hit_events].unsqueeze(0)  # (T, N)
        
        # Aplicar máscara correctamente
        if assignment_valid_mask.any():
            pred_assign_valid = pred_assignment.squeeze(-1)[assignment_valid_mask]
            gt_assign_valid = gt_assignment.squeeze(-1)[assignment_valid_mask]
            loss_assign = F.binary_cross_entropy_with_logits(pred_assign_valid, gt_assign_valid)
        else:
            loss_assign = torch.tensor(0.0, device=device)
        
        # =============================================
        # 8. Loss de Stop (BCE)
        # =============================================
        step_idx_stop = torch.arange(T, device=device).unsqueeze(1)  # (T, 1)
        gt_stop = (step_idx_stop >= pfos_per_event.unsqueeze(0)).float().unsqueeze(-1)  # (T, B, 1)
        
        pred_stop = output["stop_logits"]  # (T, B, 1)
        loss_stop = F.binary_cross_entropy_with_logits(pred_stop, gt_stop)
        
        # =============================================
        # 9. Loss total
        # =============================================
        total_loss = (
            self.lambda_dir * loss_dir +
            self.lambda_mag * loss_mag +
            self.lambda_pid * loss_pid +
            self.lambda_charge * loss_charge +
            self.lambda_assign * loss_assign +
            self.lambda_stop * loss_stop
        )
        
        return {
            "loss": total_loss,
            "loss_dir": loss_dir,
            "loss_mag": loss_mag,
            "loss_pid": loss_pid,
            "loss_charge": loss_charge,
            "loss_assign": loss_assign,
            "loss_stop": loss_stop,
            "valid_mask": valid_mask,
        }


class GATrAutoRegressorLightningModule(L.LightningModule):
    """
    Lightning Module para entrenar GATrAutoRegressor.
    """
    
    def __init__(
        self,
        model: GATrAutoRegressor,
        cfg: Any,
        plot_every_n_steps: int = 50,
    ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.plot_every_n_steps = plot_every_n_steps
        # self.automatic_optimization = False  # For step backward
        
        # Loss module
        self.loss_fn = GATrAutoRegressorLoss(
            lambda_dir=getattr(cfg, "lambda_dir", 1.0),
            lambda_mag=getattr(cfg, "lambda_mag", 1.0),
            lambda_pid=getattr(cfg, "lambda_pid", 1.0),
            lambda_charge=getattr(cfg, "lambda_charge", 0.5),
            lambda_assign=getattr(cfg, "lambda_assign", 1.0),
            lambda_stop=getattr(cfg, "lambda_stop", 0.5),
        )
        
        self.save_hyperparameters(ignore=["model"])
    
    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=getattr(self.cfg, "weight_decay", 1e-4),
        )
        
        scheduler_cfg = getattr(self.cfg, "scheduler", None)
        warmup_pct = float(getattr(self.cfg, "warmup_pct", 0.0))
        warmup_start_factor = float(getattr(self.cfg, "warmup_start_factor", 0.1))
        max_epochs = int(getattr(self.cfg, "max_epochs", 100))
        warmup_epochs = int(max_epochs * warmup_pct)
        
        if scheduler_cfg == "cosine":
            # Si hay warmup, el coseno se aplica en las épocas restantes.
            cosine_t_max = max(1, max_epochs - warmup_epochs)
            main_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=cosine_t_max,
                eta_min=getattr(self.cfg, "lr_min", 1e-6),
            )
        else:
            main_sched = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=getattr(self.cfg, "decay_steps", 30),
                gamma=getattr(self.cfg, "decay_rate", 0.5),
            )
        
        if warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                opt,
                schedulers=[warmup_sched, main_sched],
                milestones=[warmup_epochs],
            )
        else:
            sched = main_sched
        
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "interval": "epoch",
            }
        }
    
    def _split_by_detector(
        self,
        mv_v: torch.Tensor,
        mv_s: torch.Tensor,
        scalars: torch.Tensor,
        hit_batch: torch.Tensor,
    ):
        """
        Divide los hits por tipo de detector (4 grupos).
        
        Los scalars tienen shape (N, 6) donde columnas 2:6 son one-hot del detector:
        - 0: INNER_TRACKER
        - 1: ECAL
        - 2: HCAL
        - 3: MUON_TRACKER
        
        Args:
            mv_v: (N, 3) posiciones
            mv_s: (N, 3) vectores secundarios
            scalars: (N, 6) [log_E, log_p, det_0, det_1, det_2, det_3]
            hit_batch: (N,) índice de evento por hit
        
        Returns:
            4 listas de tensores, una por detector
        """
        # Obtener tipo de detector de cada hit desde columnas 2:6
        detector_type = scalars[:, 2:6].argmax(dim=-1)  # (N,) valores 0,1,2,3
        
        mv_v_split = []
        mv_s_split = []
        scalars_split = []
        batch_split = []
        
        for det_idx in range(4):
            mask = detector_type == det_idx
            
            if mask.sum() > 0:
                mv_v_split.append(mv_v[mask])
                mv_s_split.append(mv_s[mask])
                scalars_split.append(scalars[mask])
                batch_split.append(hit_batch[mask])
            else:
                # Detector sin hits: tensor vacío con shape correcto
                mv_v_split.append(torch.empty(0, 3, device=mv_v.device, dtype=mv_v.dtype))
                mv_s_split.append(torch.empty(0, 3, device=mv_s.device, dtype=mv_s.dtype))
                scalars_split.append(torch.empty(0, scalars.shape[1], device=scalars.device, dtype=scalars.dtype))
                batch_split.append(torch.empty(0, device=hit_batch.device, dtype=hit_batch.dtype))
        
        return mv_v_split, mv_s_split, scalars_split, batch_split
    
    def _prepare_batch(self, batch):
        """
        Prepara los datos del batch para el modelo.
        
        Espera que el batch tenga (después del batching de PyG con PFAutoRegressorData):
        - mv_v_part: posiciones/direcciones de hits (N_total, 3)
        - mv_s_part: vectores secundarios (N_total, 3)
        - scalars: (E, p_mod) por hit (N_total, 2)
        - batch: índice de evento por hit (N_total,)
        - pfo_pid: one-hot PID del GT (N_pfo_total, 5)
        - pfo_momentum: momentum del GT (N_pfo_total, 3)
        - pfo_p_mod: log(|p|) ya preprocesado (N_pfo_total, 1)
        - pfo_charge: charge del GT (N_pfo_total, 1)
        - pfo_event_idx: índice de evento por PFO (N_pfo_total,) - INCREMENTADO por PyG (0,0,1,1,2,...)
        - hit_to_pfo: índice LOCAL de PFO por hit (N_total,) - NO incrementado, índices 0,1,2,... por evento
        - n_pfo: número de PFOs por evento (tensor o lista de B elementos)
        
        IMPORTANTE:
        - pfo_event_idx: tiene índices de evento del batch (0,0,1,1,2,2,...)
        - hit_to_pfo: tiene índices LOCALES (0,1,2,0,1,0,1,2,...) porque el modelo
          autoregresivo compara con step_idx que va 0, 1, 2, ...
        
        Para modo 'detector_split', divide los hits por tipo de detector.
        """
        # GT para las losses (igual para ambos modos)
        pfo_true_objects = {
            "pid": batch.pfo_pid,
            "momentum": batch.pfo_momentum,
            "p_mod": batch.pfo_p_mod if hasattr(batch, 'pfo_p_mod') else None,
            "charge": batch.pfo_charge,
            "batch": batch.pfo_event_idx,  # Índices de evento en batch (0,0,1,1,2,...)
            "hit_to_pfo": batch.hit_to_pfo,  # Índices LOCALES de PFO (0,1,2,0,1,...)
        }
        
        # Dividir según modo del modelo
        if self.model.mode == "detector_split":
            mv_v_part, mv_s_part, scalars, hit_batch = self._split_by_detector(
                batch.mv_v_part,
                batch.mv_s_part,
                batch.scalars,
                batch.batch,
            )
        else:  # whole_detector
            mv_v_part = [batch.mv_v_part]
            mv_s_part = [batch.mv_s_part]
            scalars = [batch.scalars]
            hit_batch = [batch.batch]
        
        return mv_v_part, mv_s_part, scalars, hit_batch, pfo_true_objects
    
    def build_gt_index_tb(self,
        gt_batch: torch.Tensor,  # (N_pfo,)
        T: int,
        B: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Devuelve idx_tb: (T,B) con el índice GLOBAL del pfo GT correspondiente a (t,b).
        Si un evento no tiene pfo en ese step, se pone -1.
        """
        N_pfo = gt_batch.shape[0]
        # Reutilizamos tu reorganize, pero metiendo como "values" los índices [0..N_pfo-1]
        gt_indices = torch.arange(N_pfo, device=device, dtype=torch.long)
        gt_indices_tb, _ = reorganize_gt_to_tb(gt_batch, gt_indices, T, B, device)  # (T,B)

        # Problema: reorganize_gt_to_tb rellena con 0 por defecto para "vacíos".
        # Necesitamos distinguir los vacíos. Usamos el mismo conteo que ya haces para valid_mask:
        pfos_per_event = torch.bincount(gt_batch.long(), minlength=B)
        step_idx = torch.arange(T, device=device).unsqueeze(1)   # (T,1)
        valid_mask = step_idx < pfos_per_event.unsqueeze(0)      # (T,B)

        idx_tb = gt_indices_tb.clone()
        idx_tb[~valid_mask] = -1
        return idx_tb
    
    def loss_one_step(
        self,
        step_idx: int,
        pfo_pred: Dict[str, torch.Tensor],          # dict con momentum (B,3), p_mod (B,1), pid (B,5), charge (B,1)
        assignment_logits: torch.Tensor,            # (N_hits,1)
        stop_logits: torch.Tensor,                  # (B,1)
        pfo_true_objects: Dict[str, torch.Tensor],  # GT flat
        hit_batch: torch.Tensor,                    # (N_hits,)
        pfos_per_event: torch.Tensor,               # (B,)
        gt_idx_tb: torch.Tensor,                    # (T,B) indices globales GT
    ) -> Dict[str, torch.Tensor]:
        device = stop_logits.device
        B = pfos_per_event.shape[0]

        # --- validez por evento en este step
        valid_evt = (step_idx < pfos_per_event)          # (B,)
        valid_evt_any = bool(valid_evt.any().item())

        # --- GT index global para este step (B,)
        gt_idx = gt_idx_tb[step_idx]                     # (B,) con -1 en inválidos

        # ===== 1) Momentum dir + 2) Magnitud p_mod + 3) PID + 4) Charge =====
        if valid_evt_any:
            # Selecciona sólo eventos válidos
            valid_idx = gt_idx[valid_evt]                # (n_valid,) índices globales GT

            gt_momentum = pfo_true_objects["momentum"][valid_idx]  # (n_valid,3)
            gt_p_mod    = pfo_true_objects["p_mod"][valid_idx]     # (n_valid,1) log(|p|)
            gt_pid_oh   = pfo_true_objects["pid"][valid_idx]       # (n_valid,5)
            gt_charge   = pfo_true_objects["charge"][valid_idx]    # (n_valid,1)

            pred_momentum = pfo_pred["momentum"][valid_evt]        # (n_valid,3)
            pred_p_mod    = pfo_pred["p_mod"][valid_evt]           # (n_valid,1) log(|p|)
            pred_pid      = pfo_pred["pid"][valid_evt]             # (n_valid,5) logits
            pred_charge   = pfo_pred["charge"][valid_evt]          # (n_valid,1)

            # dir loss: 1 - cos
            pred_dir = F.normalize(pred_momentum, dim=-1, eps=1e-8)
            gt_dir   = F.normalize(gt_momentum,  dim=-1, eps=1e-8)
            cos_sim  = (pred_dir * gt_dir).sum(dim=-1)             # (n_valid,)
            loss_dir = (1.0 - cos_sim).mean()

            # mag loss (log-scale)
            loss_mag = F.mse_loss(pred_p_mod.squeeze(-1), gt_p_mod.squeeze(-1))

            # pid loss
            gt_pid_class = gt_pid_oh.argmax(dim=-1)                # (n_valid,)
            loss_pid = F.cross_entropy(pred_pid, gt_pid_class)

            # charge loss
            loss_charge = F.mse_loss(pred_charge.squeeze(-1), gt_charge.squeeze(-1))

        else:
            loss_dir = torch.tensor(0.0, device=device)
            loss_mag = torch.tensor(0.0, device=device)
            loss_pid = torch.tensor(0.0, device=device)
            loss_charge = torch.tensor(0.0, device=device)

        # ===== 5) Assignment BCE =====
        # GT: un hit pertenece al step t si hit_to_pfo == t
        hit_events = hit_batch.long()  # (N_hits,)
        valid_hit = step_idx < pfos_per_event[hit_events]  # (N_hits,) hit válido si su evento tiene ese step

        if bool(valid_hit.any().item()):
            gt_assign = (pfo_true_objects["hit_to_pfo"] == step_idx) & valid_hit  # (N_hits,)
            pred = assignment_logits.squeeze(-1)[valid_hit]                        # (N_valid_hits,)
            tgt  = gt_assign.float()[valid_hit]                                    # (N_valid_hits,)
            loss_assign = F.binary_cross_entropy_with_logits(pred, tgt)
        else:
            loss_assign = torch.tensor(0.0, device=device)

        # ===== 6) Stop BCE =====
        # GT stop=1 si ya hemos pasado el nº de pfos del evento
        gt_stop = (step_idx >= pfos_per_event).float().unsqueeze(-1)               # (B,1)
        loss_stop = F.binary_cross_entropy_with_logits(stop_logits, gt_stop)

        # ===== total =====
        total = (
            self.loss_fn.lambda_dir   * loss_dir +
            self.loss_fn.lambda_mag   * loss_mag +
            self.loss_fn.lambda_pid   * loss_pid +
            self.loss_fn.lambda_charge* loss_charge +
            self.loss_fn.lambda_assign* loss_assign +
            self.loss_fn.lambda_stop  * loss_stop
        )

        return {
            "loss": total,
            "loss_dir": loss_dir.detach(),
            "loss_mag": loss_mag.detach(),
            "loss_pid": loss_pid.detach(),
            "loss_charge": loss_charge.detach(),
            "loss_assign": loss_assign.detach(),
            "loss_stop": loss_stop.detach(),
        }
    # def training_step(self, batch, batch_idx):
    #     opt = self.optimizers()
    #     opt.zero_grad(set_to_none=True)

    #     mv_v_part, mv_s_part, scalars, hit_batch, pfo_true_objects = self._prepare_batch(batch)
    #     teacher_forcing = True

    #     # Encoder (esto sí lo hacemos una vez)
    #     enc_output, batch_vec = self.model.forward_module(
    #         mv_v_part=mv_v_part,
    #         mv_s_part=mv_s_part,
    #         scalars=scalars,
    #         batch=hit_batch,
    #     )

    #     device = enc_output[0].device
    #     B = int(batch_vec.max().item()) + 1 if batch_vec.numel() > 0 else 0

    #     # cuántos PFOs GT por evento
    #     gt_batch = pfo_true_objects["batch"].long()
    #     pfos_per_event = torch.bincount(gt_batch, minlength=B)  # (B,)

    #     # nº de steps de training = max PFOS (capado)
    #     num_steps = int(pfos_per_event.max().item()) if pfos_per_event.numel() > 0 else 1
    #     num_steps = min(num_steps, self.model._max_steps())
    #     if self.model.max_ar_steps_train is not None:
    #         num_steps = min(num_steps, self.model.max_ar_steps_train)
    #     num_steps = max(num_steps, 1)

    #     # Tabla (T,B) de índices GT por step/event
    #     gt_idx_tb = self.build_gt_index_tb(gt_batch, T=num_steps, B=B, device=device)  # (T,B)

    #     # Estado autoregresivo:
    #     # En training teacher forcing, SOLO necesitas residual para enmascarar hits “ya usados”.
    #     # Pero ojo: como residual lo actualizas con GT hard mask, NO depende de predicciones -> no crea grafo entre steps.
    #     residual = self.model._init_residual(batch_vec.shape[0], device)

    #     active_events = torch.ones(B, dtype=torch.bool, device=device)  # en training no se usa para cortar

    #     # acumuladores (detach) para logging
    #     loss_acc = torch.tensor(0.0, device=device)
    #     dir_acc = mag_acc = pid_acc = charge_acc = assign_acc = stop_acc = torch.tensor(0.0, device=device)
    
    #     # Bucle autoregresivo: forward 1 step -> loss_step -> backward -> liberar grafo
    #     for t in range(num_steps):
    #         # --- Build tokens (depende de residual, que en training es GT y no tiene grad)
    #         enc_output_step = tuple(x.detach().requires_grad_(True) for x in enc_output)
    #         mv_out, scalar_out = self.model._build_hit_tokens(enc_output_step, residual)
    #         scalar_dim = scalar_out.size(1)

    #         # En training teacher forcing, _build_object_tokens ignora pfo_list predicho
    #         # (usa pfo_true_objects). Puedes pasar [] sin problema.
    #         object_tokens = self.model._build_object_tokens(
    #             pfo_list=[],
    #             training=True,
    #             pfo_true_objects=pfo_true_objects,
    #             scalar_out=scalar_out,
    #             step_idx=t,
    #         )
    #         query_token = self.model._build_query_token(t, scalar_dim=scalar_dim, scalar_device=scalar_out.device)

    #         tokens_mv, tokens_s, tokens_batch = self.model._assemble_tokens(
    #             (mv_out, scalar_out),
    #             object_tokens,
    #             query_token,
    #             batch_vec,
    #             active_events,
    #         )

    #         decoded_tokens = self.model._autoregressive_step(tokens_mv, tokens_s, tokens_batch)
    #         query_embedding = self.model._extract_query_embedding(decoded_tokens)
    #         hits_embedding = self.model._extract_hit_embeddings(decoded_tokens, batch_vec, tokens_s)

    #         # Predicciones del step t
    #         pfo = self.model._predict_pfo_properties(query_embedding, active_events, training=True)
    #         _, assignment_logits = self.model._predict_assignment(
    #             decoded_tokens, query_embedding, hits_embedding, batch_vec, residual, active_events, training=True
    #         )
    #         _, stop_logits = self.model._predict_stop(query_embedding, residual, batch_vec)

    #         # --- Loss del step t
    #         losses_t = self.loss_one_step(
    #             step_idx=t,
    #             pfo_pred=pfo,
    #             assignment_logits=assignment_logits,
    #             stop_logits=stop_logits,
    #             pfo_true_objects=pfo_true_objects,
    #             hit_batch=batch_vec,
    #             pfos_per_event=pfos_per_event,
    #             gt_idx_tb=gt_idx_tb,
    #         )

    #         # Normaliza para que el gradiente total no crezca con num_steps
    #         loss_step = losses_t["loss"] / float(num_steps)

    #         # Backward inmediatamente -> no guardas grafo de steps previos
    #         # self.manual_backward(loss_step)

    #         # --- Update residual (training: GT hard mask)
    #         # Esto mantiene la información necesaria para el siguiente step (qué hits “quedan”).
    #         # Como usa GT y no usa predicciones, no engancha el grafo entre steps.
    #         residual = self.model._update_residual(
    #             residual,
    #             assignment=torch.zeros_like(residual),  # no se usa en training, pero la firma lo pide
    #             training=True,
    #             step_idx=t,
    #             hit_to_pfo=pfo_true_objects["hit_to_pfo"],
    #         )

    #         # Por seguridad: cortar cualquier posible grafo accidental (normalmente residual ya no tiene grad)
    #         residual = residual.detach()

    #         # logging acumulado (detach)
    #         loss_acc = loss_acc + loss_step.detach()
    #         dir_acc = dir_acc + (losses_t["loss_dir"] / float(num_steps))
    #         mag_acc = mag_acc + (losses_t["loss_mag"] / float(num_steps))
    #         pid_acc = pid_acc + (losses_t["loss_pid"] / float(num_steps))
    #         charge_acc = charge_acc + (losses_t["loss_charge"] / float(num_steps))
    #         assign_acc = assign_acc + (losses_t["loss_assign"] / float(num_steps))
    #         stop_acc = stop_acc + (losses_t["loss_stop"] / float(num_steps))

    #         # libera refs grandes explícitamente (ayuda a que Python suelte tensores)
    #         del decoded_tokens, query_embedding, hits_embedding, pfo, assignment_logits, stop_logits
    #         del tokens_mv, tokens_s, tokens_batch, mv_out, scalar_out, object_tokens, query_token

    #     # Paso de optimizador al final (un step por batch como antes)
    #     opt.step()
    #     opt.zero_grad(set_to_none=True)
    #     if batch_idx % self.plot_every_n_steps == 0 and self.logger is not None:
    #         with torch.no_grad():
    #             output_full = self.model(
    #                 mv_v_part=mv_v_part,
    #                 mv_s_part=mv_s_part,
    #                 scalars=scalars,
    #                 pfo_true_objects=pfo_true_objects,
    #                 batch=hit_batch,          # IMPORTANTE: lista como espera el modelo
    #                 teacher_forcing=True,
    #             )

    #             if output_full["pfo_momentum"] is not None:
    #                 # hit_batch puede ser lista (detector_split) o lista con 1 elemento (whole_detector)
    #                 if isinstance(hit_batch, list):
    #                     all_hit_batch = torch.cat(hit_batch, dim=0)
    #                 else:
    #                     all_hit_batch = hit_batch

    #                 if isinstance(mv_v_part, list):
    #                     all_mv_v = torch.cat(mv_v_part, dim=0)
    #                 else:
    #                     all_mv_v = mv_v_part

    #                 self._log_visualizations(
    #                     output_full,
    #                     pfo_true_objects,
    #                     all_hit_batch,
    #                     all_mv_v,
    #                     prefix="train",
    #                 )
    #     # Logging como antes
    #     self.log("train/loss", loss_acc, prog_bar=True, batch_size=B, sync_dist=True)
    #     self.log("train/loss_dir", dir_acc, batch_size=B, sync_dist=True)
    #     self.log("train/loss_mag", mag_acc, batch_size=B, sync_dist=True)
    #     self.log("train/loss_pid", pid_acc, batch_size=B, sync_dist=True)
    #     self.log("train/loss_charge", charge_acc, batch_size=B, sync_dist=True)
    #     self.log("train/loss_assign", assign_acc, batch_size=B, sync_dist=True)
    #     self.log("train/loss_stop", stop_acc, batch_size=B, sync_dist=True)

    #     return loss_acc
    
    def training_step(self, batch, batch_idx):
        mv_v_part, mv_s_part, scalars, hit_batch, pfo_true_objects = self._prepare_batch(batch)
        teacher_forcing=True
        # Forward pass
        if getattr(self.cfg, "debug_memory", False):
            rank = int(self.global_rank) if hasattr(self, "global_rank") else 0
            n_hits = int(hit_batch[0].shape[0])
            n_pfo = int(pfo_true_objects["batch"].shape[0])
            b_events = int(hit_batch[0].max().item()) + 1 if hit_batch[0].numel() > 0 else 0
            max_hits_event = int(torch.bincount(hit_batch[0].long(), minlength=b_events).max().item()) if b_events > 0 else 0
            sum_pfo = int(torch.bincount(pfo_true_objects["batch"].long(), minlength=b_events).sum().item()) if b_events > 0 else n_pfo
            print(
                f"[BATCH_DEBUG][rank={rank}] batch_idx={batch_idx} B={b_events} "
                f"sum_n_hits={n_hits} max_n_hits_event={max_hits_event} sum_n_pfo={sum_pfo}"
            )

        output = self.model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=hit_batch,
            teacher_forcing=True,
        )
        
        # Handle edge case: no PFOs generated
        if output["pfo_momentum"] is None:
            self.log("train/loss", 0.0, prog_bar=True)
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute losses
        losses = self.loss_fn(output, pfo_true_objects, hit_batch[0])
        
        # Logging
        B = int(hit_batch[0].max().item()) + 1
        self.log("train/loss", losses["loss"], prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train/loss_dir", losses["loss_dir"], batch_size=B, sync_dist=True)
        self.log("train/loss_mag", losses["loss_mag"], batch_size=B, sync_dist=True)
        self.log("train/loss_pid", losses["loss_pid"], batch_size=B, sync_dist=True)
        self.log("train/loss_charge", losses["loss_charge"], batch_size=B, sync_dist=True)
        self.log("train/loss_assign", losses["loss_assign"], batch_size=B, sync_dist=True)
        self.log("train/loss_stop", losses["loss_stop"], batch_size=B, sync_dist=True)
        
        # Visualizaciones periódicas
        if batch_idx % self.plot_every_n_steps == 0 and self.logger is not None:
            # Concatenar todos los hits de diferentes detectores si están divididos
            all_mv_v = torch.cat(mv_v_part, dim=0) if isinstance(mv_v_part, list) else mv_v_part[0]
            all_hit_batch = torch.cat(hit_batch, dim=0) if isinstance(hit_batch, list) else hit_batch[0]
            self._log_visualizations(output, pfo_true_objects, all_hit_batch, all_mv_v, prefix="train")
        
        return losses["loss"]
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        mv_v_part, mv_s_part, scalars, hit_batch, pfo_true_objects = self._prepare_batch(batch)
        
        # Forward pass
        output = self.model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=hit_batch,
            teacher_forcing=True,
        )
        
        # Handle edge case
        if output["pfo_momentum"] is None:
            self.log("val/loss", 0.0, prog_bar=True)
            return torch.tensor(0.0, device=self.device)
        
        # Compute losses
        losses = self.loss_fn(output, pfo_true_objects, hit_batch[0])
        
        # Logging
        B = int(hit_batch[0].max().item()) + 1
        self.log("val/loss", losses["loss"], prog_bar=True, batch_size=B, sync_dist=True)
        self.log("val/loss_dir", losses["loss_dir"], batch_size=B, sync_dist=True)
        self.log("val/loss_mag", losses["loss_mag"], batch_size=B, sync_dist=True)
        self.log("val/loss_pid", losses["loss_pid"], batch_size=B, sync_dist=True)
        self.log("val/loss_charge", losses["loss_charge"], batch_size=B, sync_dist=True)
        self.log("val/loss_assign", losses["loss_assign"], batch_size=B, sync_dist=True)
        self.log("val/loss_stop", losses["loss_stop"], batch_size=B, sync_dist=True)
        
        # Métricas adicionales
        self._log_metrics(output, pfo_true_objects, losses["valid_mask"], prefix="val")
        
        # Visualizaciones en primer batch de validación
        if batch_idx == 0 and self.logger is not None:
            # Concatenar todos los hits de diferentes detectores si están divididos
            all_mv_v = torch.cat(mv_v_part, dim=0) if isinstance(mv_v_part, list) else mv_v_part[0]
            all_hit_batch = torch.cat(hit_batch, dim=0) if isinstance(hit_batch, list) else hit_batch[0]
            self._log_visualizations(output, pfo_true_objects, all_hit_batch, all_mv_v, prefix="val")
        
        return losses["loss"]
    
    def _log_metrics(self, output, pfo_true_objects, valid_mask, prefix="val"):
        """
        Calcula y loguea métricas adicionales.
        """
        device = output["pfo_momentum"].device
        T, B, _ = output["pfo_momentum"].shape
        
        gt_batch = pfo_true_objects["batch"]
        gt_pid = pfo_true_objects["pid"]
        gt_momentum = pfo_true_objects["momentum"]
        
        # Usar helper para reorganizar GT
        gt_pid_tb, _ = reorganize_gt_to_tb(gt_batch, gt_pid, T, B, device)
        gt_momentum_tb, _ = reorganize_gt_to_tb(gt_batch, gt_momentum, T, B, device)
        
        # PID accuracy
        pred_pid_class = output["pfo_pid"].argmax(dim=-1)  # (T, B)
        gt_pid_class = gt_pid_tb.argmax(dim=-1)  # (T, B)
        
        correct = (pred_pid_class == gt_pid_class)[valid_mask]
        pid_accuracy = correct.float().mean() if correct.numel() > 0 else torch.tensor(0.0)
        
        self.log(f"{prefix}/pid_accuracy", pid_accuracy, sync_dist=True)
        
        # Momentum error (angular)
        pred_dir = F.normalize(output["pfo_momentum"], dim=-1, eps=1e-8)
        gt_dir = F.normalize(gt_momentum_tb, dim=-1, eps=1e-8)
        
        # Ángulo en grados
        cos_sim = (pred_dir * gt_dir).sum(dim=-1).clamp(-1, 1)
        angle_error = torch.acos(cos_sim) * 180 / 3.14159
        mean_angle_error = angle_error[valid_mask].mean() if valid_mask.any() else torch.tensor(0.0)
        
        self.log(f"{prefix}/angle_error_deg", mean_angle_error, sync_dist=True)
    
    def _log_visualizations(self, output, pfo_true_objects, hit_batch, hit_positions, prefix="train"):
        """
        Genera y loguea visualizaciones avanzadas:
        - Proyecciones 2D de asignaciones (theta/phi y r/theta)
        - Matriz de confusión para PID
        - Histogramas de resolución (momentum magnitude y dirección)
        
        Args:
            output: salida del modelo
            pfo_true_objects: ground truth
            hit_batch: (N_hits,) índice de evento por hit
            hit_positions: (N_hits, 3) posiciones de los hits
            prefix: "train" o "val"
        """
        try:
            # Convertir BFloat16 a float32 para evitar problemas con matplotlib/wandb
            def to_float32(x):
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return x.detach().float()
                return x
            
            T, B, _ = output["pfo_momentum"].shape
            
            # Seleccionar un evento aleatorio
            event_id = torch.randint(0, B, (1,)).item()
            
            gt_batch = pfo_true_objects["batch"]
            n_pfos_event = (gt_batch == event_id).sum().item()
            
            # Hits de este evento
            hit_mask = (hit_batch == event_id)
            n_hits = hit_mask.sum().item()
            
            if n_hits == 0 or n_pfos_event == 0:
                return
            
            # =============================================
            # CONVERSIÓN A FLOAT32 Y PREPARACIÓN DE DATOS
            # =============================================
            hit_positions_float = to_float32(hit_positions[hit_mask])  # (n_hits, 3)
            assignments = to_float32(output["assignments"][:, hit_mask, 0])  # (T, n_hits)
            
            # Calcular coordenadas esféricas
            r = torch.norm(hit_positions_float, dim=-1).detach()  # (n_hits,)
            xy_norm = torch.norm(hit_positions_float[:, :2], dim=-1).detach()  # (n_hits,)
            
            # theta: ángulo polar (0 a pi)
            theta = torch.atan2(xy_norm, hit_positions_float[:, 2]).detach()  # (n_hits,)
            theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)
            
            # phi: ángulo azimutal (-pi a pi)
            phi = torch.atan2(hit_positions_float[:, 1], hit_positions_float[:, 0]).detach()  # (n_hits,)
            
            # Ground truth para métricas
            gt_hit_to_pfo = pfo_true_objects["hit_to_pfo"][hit_mask]  # (n_hits,)
            gt_pid = pfo_true_objects["pid"]  # (N_pfo_total, 5)
            gt_momentum = pfo_true_objects["momentum"]  # (N_pfo_total, 3)
            
            # =============================================
            # FIGURA 1: Proyección THETA/PHI con asignaciones
            # =============================================
            fig_proj_sph, ax = plt.subplots(1, 1, figsize=(10, 8))
            
            # Usar la asignación del step más probable para cada hit
            assign_probs = assignments[:n_pfos_event].detach().cpu().numpy()
            assignment_predictions = assign_probs.argmax(axis=0)  # (n_hits,)
            
            # Scatter plot con colores por asignación
            scatter = ax.scatter(
                phi.cpu().numpy(),
                theta.cpu().numpy(),
                c=assignment_predictions,
                cmap='tab10',
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            ax.set_xlabel("φ (azimuthal angle)")
            ax.set_ylabel("θ (polar angle)")
            ax.set_title(f"Event {event_id}: Hit assignments (θ/φ projection)")
            plt.colorbar(scatter, ax=ax, label="PFO step")
            
            if self.logger is not None:
                self.logger.experiment.log({
                    f"{prefix}/assignments_sphere": wandb.Image(fig_proj_sph),
                })
            plt.close(fig_proj_sph)
            
            # =============================================
            # FIGURA 2: Proyección R/THETA con asignaciones
            # =============================================
            fig_proj_r_theta, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter en coordenadas (r, theta)
            scatter1 = ax1.scatter(
                theta.cpu().numpy(),
                r.cpu().numpy(),
                c=assignment_predictions,
                cmap='tab10',
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            ax1.set_xlabel("θ (polar angle)")
            ax1.set_ylabel("r (radius)")
            ax1.set_title(f"Event {event_id}: Assignments (r/θ projection)")
            plt.colorbar(scatter1, ax=ax1, label="PFO step")
            
            # Scatter en coordenadas (phi, theta) 2D alternativa
            scatter2 = ax2.scatter(
                phi.cpu().numpy(),
                r.cpu().numpy(),
                c=assignment_predictions,
                cmap='tab10',
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            ax2.set_xlabel("φ (azimuthal angle)")
            ax2.set_ylabel("r (radius)")
            ax2.set_title(f"Event {event_id}: Assignments (r/φ projection)")
            plt.colorbar(scatter2, ax=ax2, label="PFO step")
            
            if self.logger is not None:
                self.logger.experiment.log({
                    f"{prefix}/assignments_r_theta": wandb.Image(fig_proj_r_theta),
                })
            plt.close(fig_proj_r_theta)
            
            # =============================================
            # FIGURA 3: Comparación de Asignaciones (GT vs Predicciones)
            # =============================================
            fig_assign_comp, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Ground truth: usar hit_to_pfo directamente
            # hit_to_pfo contiene índices LOCALES (0, 1, 2, ...) de los PFOs
            gt_assignments = gt_hit_to_pfo.detach().cpu().numpy()  # (n_hits,)
            
            # Scatter plot Ground Truth
            scatter_gt = ax1.scatter(
                phi.cpu().numpy(),
                theta.cpu().numpy(),
                c=gt_assignments,
                cmap='tab10',
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            ax1.set_xlabel("φ (azimuthal angle)")
            ax1.set_ylabel("θ (polar angle)")
            ax1.set_title(f"Event {event_id}: Hit Assignments - Ground Truth")
            plt.colorbar(scatter_gt, ax=ax1, label="PFO index")
            
            # Scatter plot Predicciones
            scatter_pred = ax2.scatter(
                phi.cpu().numpy(),
                theta.cpu().numpy(),
                c=assignment_predictions,
                cmap='tab10',
                s=50,
                alpha=0.7,
                edgecolors='black',
                linewidth=0.5
            )
            ax2.set_xlabel("φ (azimuthal angle)")
            ax2.set_ylabel("θ (polar angle)")
            ax2.set_title(f"Event {event_id}: Hit Assignments - Predictions")
            plt.colorbar(scatter_pred, ax=ax2, label="PFO step")
            
            fig_assign_comp.suptitle(f"Event {event_id}: Ground Truth vs Predictions", fontsize=14, fontweight='bold')
            fig_assign_comp.tight_layout()
            
            if self.logger is not None:
                self.logger.experiment.log({
                    f"{prefix}/assignments_gt_vs_pred": wandb.Image(fig_assign_comp),
                })
            plt.close(fig_assign_comp)
            
            # =============================================
            # FIGURA 3: Matriz de Confusión para PID
            # =============================================
            if n_pfos_event > 0:
                pred_pid_logits = to_float32(output["pfo_pid"][:n_pfos_event, event_id, :])  # (n_pfos, 5)
                pred_pid = pred_pid_logits.argmax(dim=-1).detach().cpu().numpy()  # (n_pfos,)
                
                # Solo contar PFOs que están en el GT
                valid_pfo_indices = (gt_batch == event_id).nonzero(as_tuple=True)[0][:n_pfos_event]
                if len(valid_pfo_indices) > 0:
                    gt_pid_event = to_float32(gt_pid[valid_pfo_indices])  # (n_pfos, 5)
                    gt_pid_class = gt_pid_event.argmax(dim=-1).detach().cpu().numpy()  # (n_pfos,)
                    
                    fig_cm, ax = plt.subplots(figsize=(8, 6))
                    cm = confusion_matrix(gt_pid_class, pred_pid, labels=[0, 1, 2, 3, 4])
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['e', 'μ', 'π', 'K', 'p'])
                    disp.plot(ax=ax, cmap='Blues')
                    ax.set_title(f"Event {event_id}: PID Confusion Matrix")
                    
                    if self.logger is not None:
                        self.logger.experiment.log({
                            f"{prefix}/pid_confusion_matrix": wandb.Image(fig_cm),
                        })
                    plt.close(fig_cm)
            
            # =============================================
            # FIGURA 4: Histogramas de Resolución de Momentum
            # =============================================
            if n_pfos_event > 0:
                pred_momentum = to_float32(output["pfo_momentum"][:n_pfos_event, event_id, :])  # (n_pfos, 3)
                pred_p_mod = torch.norm(pred_momentum, dim=-1)  # (n_pfos,)
                
                valid_pfo_indices = (gt_batch == event_id).nonzero(as_tuple=True)[0][:n_pfos_event]
                if len(valid_pfo_indices) > 0:
                    gt_momentum_event = to_float32(gt_momentum[valid_pfo_indices])  # (n_pfos, 3)
                    gt_p_mod = torch.norm(gt_momentum_event, dim=-1)  # (n_pfos,)
                    
                    fig_res, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Resolución en magnitud del momentum
                    p_resolution = (pred_p_mod - gt_p_mod) / (gt_p_mod + 1e-8)
                    p_resolution = p_resolution.detach().cpu().numpy()
                    
                    ax1.hist(p_resolution, bins=20, alpha=0.7, color='blue', edgecolor='black')
                    ax1.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Perfect')
                    ax1.set_xlabel("(p_pred - p_truth) / p_truth")
                    ax1.set_ylabel("Frequency")
                    ax1.set_title(f"Event {event_id}: Momentum Magnitude Resolution")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    # Resolución en dirección (ángulo)
                    pred_dir = pred_momentum / (pred_p_mod.unsqueeze(-1) + 1e-8)
                    gt_dir = gt_momentum_event / (gt_p_mod.unsqueeze(-1) + 1e-8)
                    
                    cos_similarity = (pred_dir * gt_dir).sum(dim=-1)
                    cos_similarity = torch.clamp(cos_similarity, -1.0, 1.0)
                    angle_resolution = torch.acos(cos_similarity) * 180 / torch.pi  # en grados
                    angle_resolution = angle_resolution.detach().cpu().numpy()
                    
                    ax2.hist(angle_resolution, bins=20, alpha=0.7, color='green', edgecolor='black')
                    ax2.set_xlabel("Angular deviation (degrees)")
                    ax2.set_ylabel("Frequency")
                    ax2.set_title(f"Event {event_id}: Momentum Direction Resolution")
                    ax2.grid(True, alpha=0.3)
                    
                    fig_res.tight_layout()
                    
                    if self.logger is not None:
                        self.logger.experiment.log({
                            f"{prefix}/momentum_resolution": wandb.Image(fig_res),
                        })
                    plt.close(fig_res)
            
            # =============================================
            # FIGURA 5: Stop Probabilities
            # =============================================
            fig_stop, ax = plt.subplots(1, 1, figsize=(8, 4))
            
            stop_probs = to_float32(output["stop_probs"][:, event_id, 0]).detach().cpu().numpy()
            steps = range(len(stop_probs))
            ax.bar(steps, stop_probs, alpha=0.7, color='steelblue')
            ax.axhline(y=0.5, color='r', linestyle='--', label='threshold')
            ax.axvline(x=n_pfos_event - 0.5, color='g', linestyle='--', label=f'GT: {n_pfos_event} PFOs')
            ax.set_xlabel("Step")
            ax.set_ylabel("Stop probability")
            ax.set_title(f"Event {event_id}: Stop probabilities")
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            if self.logger is not None:
                self.logger.experiment.log({
                    f"{prefix}/stop_probs": wandb.Image(fig_stop),
                })
            
            plt.close(fig_stop)
            
        except Exception as e:
            print(f"Warning: visualization failed with error: {e}")
            import traceback
            traceback.print_exc()
            plt.close("all")
