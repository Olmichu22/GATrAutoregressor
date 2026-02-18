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
        if scheduler_cfg == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=getattr(self.cfg, "max_epochs", 100),
                eta_min=getattr(self.cfg, "lr_min", 1e-6),
            )
        else:
            sched = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=getattr(self.cfg, "decay_steps", 30),
                gamma=getattr(self.cfg, "decay_rate", 0.5),
            )
        
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
    
    def training_step(self, batch, batch_idx):
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
            self._log_visualizations(output, pfo_true_objects, hit_batch[0], prefix="train")
        
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
            self._log_visualizations(output, pfo_true_objects, hit_batch[0], prefix="val")
        
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
    
    def _log_visualizations(self, output, pfo_true_objects, hit_batch, prefix="train"):
        """
        Genera y loguea visualizaciones.
        """
        try:
            T, B, _ = output["pfo_momentum"].shape
            
            # Seleccionar un evento aleatorio
            event_id = torch.randint(0, B, (1,)).item()
            
            gt_batch = pfo_true_objects["batch"]
            n_pfos_event = (gt_batch == event_id).sum().item()
            
            # Figura de assignments
            fig_assign, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            # Hits de este evento
            hit_mask = (hit_batch == event_id)
            n_hits = hit_mask.sum().item()
            
            if n_hits > 0 and n_pfos_event > 0:
                # Assignment predictions para este evento
                assignments = output["assignments"][:, hit_mask, 0].detach().cpu().numpy()  # (T, n_hits)
                
                im = ax.imshow(assignments[:n_pfos_event], aspect='auto', cmap='viridis', vmin=0, vmax=1)
                ax.set_xlabel("Hit index")
                ax.set_ylabel("PFO step")
                ax.set_title(f"Event {event_id}: Assignment predictions")
                plt.colorbar(im, ax=ax)
            
            if self.logger is not None:
                self.logger.experiment.log({
                    f"{prefix}/assignments": wandb.Image(fig_assign),
                }, step=self.global_step)
            
            plt.close(fig_assign)
            
            # Figura de stop probabilities
            fig_stop, ax = plt.subplots(1, 1, figsize=(8, 4))
            
            stop_probs = output["stop_probs"][:, event_id, 0].detach().cpu().numpy()
            steps = range(len(stop_probs))
            ax.bar(steps, stop_probs, alpha=0.7)
            ax.axhline(y=0.5, color='r', linestyle='--', label='threshold')
            ax.axvline(x=n_pfos_event - 0.5, color='g', linestyle='--', label=f'GT: {n_pfos_event} PFOs')
            ax.set_xlabel("Step")
            ax.set_ylabel("Stop probability")
            ax.set_title(f"Event {event_id}: Stop probabilities")
            ax.legend()
            
            if self.logger is not None:
                self.logger.experiment.log({
                    f"{prefix}/stop_probs": wandb.Image(fig_stop),
                }, step=self.global_step)
            
            plt.close(fig_stop)
            
        except Exception as e:
            print(f"Warning: visualization failed with error: {e}")
            plt.close("all")
