# models/gatr_flow_extractor.py
from gatr.interface import embed_point

import torch
import torch.nn as nn
from src.model.gatr_module import GATrBasicModule
from torch_scatter import scatter_mean
from dataclasses import dataclass, field
from typing import List, Optional

def concatenate_outputs(outputs, batch):
    n_outputs = len(outputs)
    n_batch = len(batch)
    assert n_outputs == n_batch, "Number of outputs and batch tensors must match"
    if n_outputs < 2:
        return outputs[0], batch[0] # No need to concatenate
    mv_out_final_conc = []
    scalar_out_final_conc = []
    point_conc = []
    scalar_conc = []
    batch_conc = []
    for output, batch_part in zip(outputs, batch):
        mv_out_final, scalar_out_final, point, scalar = output
        mv_out_final_conc.append(mv_out_final)
        scalar_out_final_conc.append(scalar_out_final)
        point_conc.append(point)
        scalar_conc.append(scalar)
        batch_conc.append(batch_part)
    
    mv_v_out_final = torch.cat(mv_out_final_conc, dim=0)
    scalar_out_final =  torch.cat(scalar_out_final_conc, dim=0)
    point = torch.cat(point_conc, dim=0)
    scalar = torch.cat(scalar_conc, dim=0)
    batch = torch.cat(batch_conc, dim=0)
    # Sort by batch to maintain order
    sorted_indices = torch.argsort(batch)
    mv_v_out_final = mv_v_out_final[sorted_indices]
    scalar_out_final = scalar_out_final[sorted_indices]
    point = point[sorted_indices]
    scalar = scalar[sorted_indices]
    batch = batch[sorted_indices]
    
    return mv_v_out_final, scalar_out_final, point, scalar, batch 
    

class SequentialWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(
        self,
        mv_v_part,
        mv_s_part,
        scalars,
        batch,
        embedded_geom=[None]
    ):
        outputs = self.module(
            mv_v_part[0],
            mv_s_part[0],
            scalars[0],
            batch[0],
            embedded_geom=embedded_geom[0]
        )
        return outputs, batch[0]

class ParallelWrapper(nn.Module):
    def __init__(self, modules: List[GATrBasicModule], aggregate_module: GATrBasicModule):
        """
        modules: iterable de 4 nn.Module
        """
        super().__init__()
        assert len(modules) == 4, "Se requieren exactamente 4 módulos"
        self.modules_ = nn.ModuleList(modules)
        self.aggregate_module = aggregate_module
    def forward(
        self,
        mv_v_part,
        mv_s_part,
        scalars,
        batch,
        embedded_geom=[None, None, None, None]
    ):
        outputs = []

        for i, module in enumerate(self.modules_):
            out = module(
                mv_v_part[i],
                mv_s_part[i],
                scalars[i],
                batch[i],
                embedded_geom=embedded_geom[i]
            )
            outputs.append(out)
        # This is not implemeted yet, general structure only
        mv_v_out_final, scalar_out_final, _, _, batch = concatenate_outputs(outputs, batch)
        # mv_out_final is already in GA embedded form
        outputs = self.aggregate_module(mv_v_part=None,
                                        mv_s_part=None,
                                        scalars = scalar_out_final, batch=batch, embedded_geom=mv_v_out_final
        )
        return outputs, batch

@dataclass
class GATrAutoRegressorParamsWhole:
    hidden_mv: int = 32
    hidden_s: int = 64
    num_blocks: int = 3
    in_s_channels: int = 2  # E, p_mod
    in_mv_channels: int = 1
    out_mv_channels: int = 1
    dropout: float = 0.1
    out_s_channels: Optional[int] = None
    
@dataclass
class GATrAutoRegressive:
    hidden_mv: int = 32
    hidden_s: int = 64
    num_blocks: int = 3
    out_mv_channels: int = 1
    out_s_channels: Optional[int] = None
    dropout: float = 0.1
# Creamos una dataclass para los parámetros
@dataclass
class GATrAutoRegressorParamsSplit:
    hidden_mv: List[int] = field(default_factory=lambda: [32, 32, 32, 32])
    hidden_s: List[int] = field(default_factory=lambda: [64, 64, 64, 64])
    num_blocks: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    in_s_channels: List[int] = field(default_factory=lambda: [2, 2, 2, 2])  # E, p_mod
    in_mv_channels: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    out_mv_channels: List[int] = field(default_factory=lambda: [1, 1, 1, 1])
    dropout: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1, 0.1])
    out_s_channels: List[Optional[int]] = field(default_factory=lambda: [None, None, None, None])
    final_module_cfg: GATrAutoRegressorParamsWhole = field(
        default_factory=GATrAutoRegressorParamsWhole
    )
class GATrAutoRegressor(nn.Module):
    """
    GATr-based flow model for particle shower extraction.
    """
    def __init__(self, mode:str = "detector_split", params_cfg:dict = {},): 
        super().__init__()
        if mode not in ["detector_split", "whole_detector"]:
            raise ValueError(f"Invalid mode: {mode}. Choose 'detector_split' or 'whole_detector'.")
        self.mode = mode
        # backbone GATr: procesa nodos → características latentes por nodo
        
        if self.mode == "whole_detector":
            cfg: GATrAutoRegressorParamsWhole = params_cfg.get("whole_detector")
            if cfg is None:
                raise ValueError("Missing configuration for 'whole_detector' mode.")
            
            gatrenc = GATrBasicModule(
                hidden_mv_channels = cfg.hidden_mv,
                hidden_s_channels = cfg.hidden_s,
                num_blocks = cfg.num_blocks,
                in_s_channels = cfg.in_s_channels,
                in_mv_channels = cfg.in_mv_channels,
                out_mv_channels = cfg.out_mv_channels,
                dropout = cfg.dropout,
                out_s_channels = cfg.out_s_channels
            )
            self.forward_module =  SequentialWrapper(
                gatrenc
            )
            out_s_dim = cfg.out_s_channels if cfg.out_s_channels is not None else cfg.hidden_s
            out_mv_channels = cfg.out_mv_channels
        elif self.mode == "detector_split":
            cfg: GATrAutoRegressorParamsSplit = params_cfg.get("detector_split")
            if cfg is None:
                raise ValueError("Missing configuration for 'detector_split' mode.")
            
            modules = []
            for i in range(4):
                gatrenc = GATrBasicModule(
                    hidden_mv_channels = cfg.hidden_mv[i],
                    hidden_s_channels = cfg.hidden_s[i],
                    num_blocks = cfg.num_blocks[i],
                    in_s_channels = cfg.in_s_channels[i],
                    in_mv_channels = cfg.in_mv_channels[i],
                    out_mv_channels = cfg.out_mv_channels[i],
                    dropout = cfg.dropout[i],
                    out_s_channels = cfg.out_s_channels[i]
                )
                modules.append(gatrenc)
            aggregate_module = GATrBasicModule(
                hidden_mv_channels = cfg.final_module_cfg.hidden_mv,
                hidden_s_channels = cfg.final_module_cfg.hidden_s,
                num_blocks = cfg.final_module_cfg.num_blocks,
                in_s_channels = cfg.final_module_cfg.in_s_channels,
                in_mv_channels = cfg.final_module_cfg.in_mv_channels,
                out_mv_channels = cfg.final_module_cfg.out_mv_channels,
                dropout = cfg.final_module_cfg.dropout,
                out_s_channels = cfg.final_module_cfg.out_s_channels
            )
            self.forward_module = ParallelWrapper(
                modules, aggregate_module
            )
            out_s_dim = cfg.final_module_cfg.out_s_channels if cfg.final_module_cfg.out_s_channels is not None else cfg.final_module_cfg.hidden_s
            out_mv_channels = cfg.final_module_cfg.out_mv_channels
        self.autoregressive_module = None  # Placeholder for autoregressive module initialization
        autorregresive_module_cfg: GATrAutoRegressive = params_cfg.get("autoregressive_module")
        if autorregresive_module_cfg is None:
            raise ValueError("Missing configuration for 'autoregressive_module'.")
        AR_module = GATrBasicModule(
            hidden_mv_channels = autorregresive_module_cfg.hidden_mv,
            hidden_s_channels = autorregresive_module_cfg.hidden_s,
            num_blocks = autorregresive_module_cfg.num_blocks,
            in_s_channels = out_s_dim + 1 + 3 + 5 + 2,  # scalars + residual(1) + one_hot_type(3) + one_hot_pid(5) + charge(1) + p_mod(1)
            in_mv_channels = out_mv_channels,
            out_mv_channels = autorregresive_module_cfg.out_mv_channels,
            dropout = autorregresive_module_cfg.dropout,
            out_s_channels = autorregresive_module_cfg.out_s_channels
        )
        final_s_channels = autorregresive_module_cfg.out_s_channels if autorregresive_module_cfg.out_s_channels is not None else autorregresive_module_cfg.hidden_s
        self.autoregressive_module = AR_module

        self.max_steps = int(params_cfg.get("max_steps", 128))
        max_ar_steps_train = params_cfg.get("max_ar_steps_train", None)
        self.max_ar_steps_train = int(max_ar_steps_train) if max_ar_steps_train is not None else None
        self.debug_memory = bool(params_cfg.get("debug_memory", False))
        self.debug_memory_interval = int(params_cfg.get("debug_memory_interval", 5))

        self.p_head = nn.Sequential(
            nn.Linear(final_s_channels + 1, final_s_channels),
            nn.ReLU(),
            nn.Linear(final_s_channels, 1)   # |p|
        )

        self.pid_head = nn.Sequential(
            nn.Linear(final_s_channels + 1, final_s_channels),
            nn.ReLU(),
            nn.Linear(final_s_channels, 5)   # e, mu, pi, n, gamma
        )

        self.charge_head = nn.Sequential(
            nn.Linear(final_s_channels + 1, final_s_channels),
            nn.ReLU(),
            nn.Linear(final_s_channels, 1)   # Va a ser un valor continuo
        )

        self.assignment_head = nn.Sequential(
            nn.Linear(final_s_channels + final_s_channels + 1 + 1 + 3 + 1, final_s_channels),
            # hit_point(3) + hit_scalar(1) + hit_scalar_out(S) + query_scalar_out(S) + query_scalar(1) + residual(1)
            nn.ReLU(),
            nn.Linear(final_s_channels, 1)
        )

        self.stop_head = nn.Sequential(
            nn.Linear(final_s_channels + 1 + 1, final_s_channels),
            nn.ReLU(),
            nn.Linear(final_s_channels, 1)
        )

    def forward(self,
                mv_v_part: List[torch.Tensor],
                mv_s_part: List[torch.Tensor],
                scalars: List[torch.Tensor],
                pfo_true_objects: dict,
                batch: List[torch.Tensor],
                teacher_forcing: Optional[bool] = None):

        
    #     pfo_true_objects={
    #     "pid": data.pfo_pid,
    #     "momentum": data.pfo_momentum,
    #     "charge": data.pfo_charge,
    #     "batch": data.pfo_event_idx,
    #     "hit_to_pfo": data.hit_to_pfo
    # }
        enc_output, batch = self.forward_module(
            mv_v_part,
            mv_s_part,
            scalars,
            batch
        )
        
        batch_data_length =  batch.shape[0]
        device = enc_output[0].device
        loop_teacher_forcing = self.training if teacher_forcing is None else teacher_forcing
        pfo_list, assignments, stop_probs, assignments_logits, stop_logits = self._run_autoregressive_loop(
                enc_output=enc_output,
                batch_data_length=batch_data_length,
                device = device,
                batch=batch,
                teacher_forcing=loop_teacher_forcing,
                pfo_true_objects=pfo_true_objects
            )

        return self._format_output(pfo_list, assignments, stop_probs, assignments_logits, stop_logits)

    # ============================
    # AUTOREGRESSIVE LOOP
    # ============================
    def _run_autoregressive_loop(self, enc_output, batch_data_length, device, batch, teacher_forcing: bool, pfo_true_objects):
        """
        Main autoregressive loop.
        Generates PFOs sequentially until STOP.
        
        Training: número de pasos = max PFOs en GT
        Inference: parar cuando todos los eventos hayan terminado
        """
        residual = self._init_residual(batch_data_length, device)
        pfo_list = self._init_object_sequence()
        assignments = []
        stop_probs = []
        assignments_logits_vec = []
        stop_logits_vec = []
        # Número de eventos del batch
        B = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        active_events = torch.ones(B, dtype=torch.bool, device=device)
        
        # En training, número de pasos = max PFOs por evento en GT
        if teacher_forcing:
            pfo_batch = pfo_true_objects["batch"]
            max_pfos = max(torch.bincount(pfo_batch).tolist()) if pfo_batch.numel() > 0 else 1
            num_steps = min(max_pfos, self._max_steps())
            if self.max_ar_steps_train is not None:
                num_steps = min(num_steps, self.max_ar_steps_train)
        else:
            max_pfos = -1
            num_steps = self._max_steps()

        if self.debug_memory:
            rank = int(torch.distributed.get_rank()) if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
            print(
                f"[AR_DEBUG][rank={rank}] teacher_forcing={teacher_forcing} "
                f"max_pfos={max_pfos} num_steps={num_steps} max_steps_cap={self._max_steps()}"
            )
        
        for step_idx in range(num_steps):
            # Build tokens
            mv_out, scalar_out = self._build_hit_tokens(enc_output, residual)
            scalar_dim = scalar_out.size(1) # (N, scalar_out + residual + one_hot_type + one_hot_pid + charge + p_mod)
            scalar_device = scalar_out.device
            object_tokens = self._build_object_tokens(pfo_list, teacher_forcing, pfo_true_objects, scalar_out, step_idx)
            query_token = self._build_query_token(step_idx, scalar_dim=scalar_dim, scalar_device=scalar_device)

            tokens_mv, tokens_s, tokens_batch = self._assemble_tokens(
                (mv_out, scalar_out), object_tokens, query_token, batch, active_events
            )

            if self.debug_memory and (step_idx % max(self.debug_memory_interval, 1) == 0 or step_idx == num_steps - 1):
                n_hits = int(batch.shape[0])
                n_obj = int(object_tokens[0].shape[0])
                n_query = int(torch.count_nonzero(active_events).item()) if not teacher_forcing else B
                self._log_memory_debug(
                    step_idx=step_idx,
                    num_steps=num_steps,
                    n_hits=n_hits,
                    n_obj=n_obj,
                    n_query=n_query,
                    tokens_mv=tokens_mv,
                    tokens_s=tokens_s,
                    tokens_batch=tokens_batch,
                )

            # GATr decoder step
            try:
                decoded_tokens = self._autoregressive_step( # point, scalar, scalar_out, tokens_batch
                    tokens_mv, tokens_s, tokens_batch
                )
            except torch.cuda.OutOfMemoryError:
                n_hits = int(batch.shape[0])
                n_obj = int(object_tokens[0].shape[0])
                n_query = int(torch.count_nonzero(active_events).item()) if not teacher_forcing else B
                self._log_memory_debug(
                    step_idx=step_idx,
                    num_steps=num_steps,
                    n_hits=n_hits,
                    n_obj=n_obj,
                    n_query=n_query,
                    tokens_mv=tokens_mv,
                    tokens_s=tokens_s,
                    tokens_batch=tokens_batch,
                    oom=True,
                )
                raise

            query_embedding, tokens_batch = self._extract_query_embedding(decoded_tokens)
            hits_embedding = self._extract_hit_embeddings(decoded_tokens, batch, tokens_s) # HAY QUE IMPLEMENTARLO
            # Predictions
            pfo = self._predict_pfo_properties(query_embedding, tokens_batch, active_events, teacher_forcing)
            assignment, assignment_logits = self._predict_assignment(
                query_embedding, hits_embedding, batch, residual, active_events, teacher_forcing
            )
            stop_prob, stop_logits = self._predict_stop(query_embedding, residual, batch)
            
            # actualizar eventos activos (solo relevante en inferencia)
            if not teacher_forcing:
                stop_mask = stop_prob.squeeze(-1) > 0.5
                active_events = active_events & (~stop_mask)
            # Update state
            residual = self._update_residual(
                residual,
                assignment,
                teacher_forcing,
                step_idx,
                pfo_true_objects["hit_to_pfo"] if teacher_forcing else None
            )
            assignments_logits_vec.append(assignment_logits)
            stop_logits_vec.append(stop_logits)
            self._append_pfo(pfo_list, pfo)
            assignments.append(assignment)
            stop_probs.append(stop_prob)
            if not teacher_forcing and not active_events.any():
                break
        return pfo_list, assignments, stop_probs, assignments_logits_vec, stop_logits_vec

    # ============================
    # INITIALIZATION
    # ============================
    def _extract_hit_embeddings(self, decoded_tokens, batch, token_scalar):
        point, scalar, scalar_out, tokens_batch = decoded_tokens

        # HIT type = one_hot_type[0]
        # asumimos que el one-hot está en las 3 últimas columnas antes de pid/charge
        is_hit = token_scalar[:, -10] == 1.0   # ajusta índice si cambia el layout

        hit_point = point[is_hit]
        hit_scalar = scalar[is_hit]
        hit_scalar_out = scalar_out[is_hit]

        return hit_point, hit_scalar, hit_scalar_out
    
    def _init_residual(self, batch_data_length, device):
        """
        Initialize residual per hit.
        Typically ones, or energy-weighted.
        """
        residual = torch.ones(batch_data_length, 1, device=device) # (N, 1)
        return residual
        # pass

    def _init_object_sequence(self):
        """
        Initialize empty list / container for generated PFOs.
        """
        return []

    def _max_steps(self):
        """
        Maximum number of autoregressive steps.
        Acts as a safety cap.
        """
        return self.max_steps

    def _log_memory_debug(self, step_idx, num_steps, n_hits, n_obj, n_query, tokens_mv, tokens_s, tokens_batch, oom: bool = False):
        rank = int(torch.distributed.get_rank()) if torch.distributed.is_available() and torch.distributed.is_initialized() else 0
        device = tokens_mv.device
        msg = (
            f"[AR_DEBUG][rank={rank}] step={step_idx + 1}/{num_steps} "
            f"N_hit={n_hits} N_obj={n_obj} N_query={n_query} N_total={int(tokens_mv.shape[0])} "
            f"tokens_mv={tuple(tokens_mv.shape)} tokens_s={tuple(tokens_s.shape)} tokens_batch={tuple(tokens_batch.shape)}"
        )
        if device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            msg += (
                f" allocated_mb={allocated:.1f} reserved_mb={reserved:.1f} "
                f"max_allocated_mb={max_allocated:.1f} max_reserved_mb={max_reserved:.1f}"
            )
        print(msg)
        if oom and device.type == "cuda":
            print(torch.cuda.memory_summary(device=device, abbreviated=True))

    # ============================
    # TOKEN BUILDERS
    # ============================
    def _build_hit_tokens(self, enc_output, residual):
        """
        Build HIT tokens:
        - Use enc_output mv + s # 
        - Inject residual
        - Add type embedding (HIT)
        """
        
        one_hot_type = torch.zeros(enc_output[1].size(0), 3, device=enc_output[1].device) # three one hot (IS_HIT, IS_OBJ, IS_QUERY)
        # One hot per particle (5), 2 extra features being charge and modulus of momentum
        extra_tokens_features = torch.zeros(enc_output[1].size(0), 7, device=enc_output[1].device)  # placeholder for extra features 
        one_hot_type[:, 0] = 1.0  # HIT type
        mv_out, scalar_out, _, _ = enc_output
        
        assert scalar_out.dim() == 2, "scalar_out must have shape (N, S)"
        assert residual.dim() == 2, "residual must have shape (N, 1)"
        
        scalar_out = torch.cat([scalar_out, residual], dim=-1)
        scalar_out = torch.cat([scalar_out, one_hot_type], dim=-1)
        scalar_out = torch.cat([scalar_out, extra_tokens_features], dim=-1)
        return mv_out, scalar_out

    def _build_object_tokens(
        self,
        pfo_list,
        training: bool,
        pfo_true_objects,
        scalar_out,
        step_idx: int,
    ):
        """
        Build OBJ tokens from previously generated PFOs.

        - Training: use true PFOs with teacher forcing
        - Inference: use previously generated PFOs
        - IMPORTANT: selection is PER EVENT, not global
        """

        device = scalar_out.device
        scalar_dim = scalar_out.size(1)

        # -------------------------------------------------
        # Caso base: no objetos aún
        # -------------------------------------------------
        if step_idx == 0:
            empty_mv = scalar_out.new_zeros(0, 1, 16)
            empty_s = scalar_out.new_zeros(0, scalar_dim)
            empty_batch = scalar_out.new_zeros(0, dtype=torch.long)
            return (empty_mv, empty_s), empty_batch

        # -------------------------------------------------
        # TRAINING: teacher forcing con PFOs verdaderos
        # -------------------------------------------------
        if training:
            pfo_batch = pfo_true_objects["batch"]        # [N_pfo_total]
            pfo_pid = pfo_true_objects["pid"]
            pfo_charge = pfo_true_objects["charge"]
            pfo_momentum = pfo_true_objects["momentum"]

            selected_indices = []

            # Procesar evento a evento
            for b in torch.unique(pfo_batch):
                mask_evt = (pfo_batch == b)
                evt_indices = mask_evt.nonzero(as_tuple=False).squeeze(-1)

                # Si este evento tiene al menos step_idx PFOs, usamos los anteriores
                if evt_indices.numel() >= step_idx:
                    selected_indices.extend(evt_indices[:step_idx].tolist())
                else:
                    # Si tiene menos, usamos todos los que haya
                    selected_indices.extend(evt_indices.tolist())

            if len(selected_indices) == 0:
                empty_mv = scalar_out.new_zeros(0, 1, 16)
                empty_s = scalar_out.new_zeros(0, scalar_dim)
                empty_batch = scalar_out.new_zeros(0, dtype=torch.long)
                return (empty_mv, empty_s), empty_batch

            selected_indices = torch.tensor(selected_indices, device=device)

            # Extraer propiedades
            pfo_momentum_step = pfo_momentum[selected_indices]
            # PFO Momentum module
            pfo_momentum_module_step = torch.norm(pfo_momentum_step, dim=-1, keepdim=True)
            pfo_momentum_step = pfo_momentum_step / (pfo_momentum_module_step + 1e-8)
            
            pfo_pid_step = pfo_pid[selected_indices]
            pfo_charge_step = pfo_charge[selected_indices]
            pfo_batch_step = pfo_batch[selected_indices]

            # Geometría (placeholder)
            embedded_mv_pfo = embed_point(pfo_momentum_step)  
            embedded_mv_pfo = embedded_mv_pfo.unsqueeze(1)    # (N,1,16)

            # Escalares
            scalar_feat_out = torch.zeros(
                embedded_mv_pfo.size(0), scalar_dim, device=device
            )
            scalar_feat_out[:, -9] = 1.0   # OBJ type one-hot (from -10 to -8)
            scalar_feat_out[:, -7:-2] = pfo_pid_step.float() # pid one hot from -7 to -3
            scalar_feat_out[:, -2:-1] = pfo_charge_step.float() # charge at -2
            scalar_feat_out[:, -1] = pfo_momentum_module_step.squeeze(-1).float() # |p| at -1
            return (embedded_mv_pfo, scalar_feat_out), pfo_batch_step

        # -------------------------------------------------
        # INFERENCE: usar PFOs generados hasta ahora
        # -------------------------------------------------
        else:
            if len(pfo_list) == 0:
                empty_mv = scalar_out.new_zeros(0, 1, 16)
                empty_s = scalar_out.new_zeros(0, scalar_dim)
                empty_batch = scalar_out.new_zeros(0, dtype=torch.long)
                return (empty_mv, empty_s), empty_batch

            # Cada PFO generado ya debe traer su batch
            # pfo["momentum"] tiene shape (B_step, 3); concatenar mantiene (N_obj, 3).
            # Con stack aquí se obtenía (T_prev, B_step, 3) y rompía el alineamiento con PID/charge.
            pfo_momentum_step = torch.cat(
                [pfo["momentum"] for pfo in pfo_list], dim=0
            ).to(device)
            # PFO Momentum module
            pfo_momentum_module_step = torch.norm(pfo_momentum_step, dim=-1, keepdim=True)
            pfo_momentum_step = pfo_momentum_step / (pfo_momentum_module_step + 1e-8)
            
            pfo_charge_step = torch.cat(
                [pfo["charge"] for pfo in pfo_list], dim=0
            ).to(device)

            pfo_batch_step = torch.cat(
                [pfo["batch"] for pfo in pfo_list], dim=0
            ).to(device)
            # pfo_pid_step with softmax activation
            pfo_pid_step = torch.softmax(torch.cat(
                [pfo["pid"] for pfo in pfo_list], dim=0
            ).to(device), dim=-1)
        

            embedded_mv_pfo = embed_point(pfo_momentum_step)
            embedded_mv_pfo = embedded_mv_pfo.unsqueeze(1)

            scalar_feat_out = torch.zeros(
                embedded_mv_pfo.size(0), scalar_dim, device=device
            )
            if scalar_feat_out.size(0) != pfo_pid_step.size(0):
                raise RuntimeError(
                    f"Object-token size mismatch: geom={scalar_feat_out.size(0)} "
                    f"pid={pfo_pid_step.size(0)}"
                )
            scalar_feat_out[:, -9] = 1.0   # OBJ type one-hot (from -10 to -8)
            scalar_feat_out[:, -7:-2] = pfo_pid_step.float() # pid one hot from -7 to -3
            scalar_feat_out[:, -2:-1] = pfo_charge_step.float() # charge at -2
            scalar_feat_out[:, -1] = pfo_momentum_module_step.squeeze(-1).float() # |p| at -1

            return (embedded_mv_pfo, scalar_feat_out), pfo_batch_step



    def _build_query_token(self, step_idx: int, scalar_dim: int =  None, scalar_device=None):
        """
        Build QUERY token for current step.
        Contains only type embedding.
        """
        query_mv = torch.zeros(1, 1, 16, device=scalar_device)  # (1,1,16)
        query_s = torch.zeros(1, scalar_dim, device=scalar_device)  # (1, scalar_dim)
        query_s[0, -8] = 1.0  # QUERY type one-hot
        return (query_mv, query_s)

    def _assemble_tokens(self, hit_tokens, object_tokens, query_token, batch, active_events):
        """
        Concatenate HIT + OBJ + QUERY tokens and build the corresponding batch vector.

        Inputs
        ------
        hit_tokens:
            (hit_mv, hit_s)
            hit_mv:    (N_hits, 1, 16)  (or (N_hits,16) -> will be unsqueezed)
            hit_s:     (N_hits, S)
        object_tokens:
            ((obj_mv, obj_s), obj_batch)
            obj_mv:    (N_obj, 1, 16)
            obj_s:     (N_obj, S)
            obj_batch: (N_obj,)  event id per object token
        query_token:
            (q_mv, q_s) where currently you build a single query (1,1,16) and (1,S)
            Here we will replicate it to have ONE QUERY PER EVENT.
        batch:
            (N_hits,) event id per hit token

        Returns
        -------
        tokens_mv:    (N_hits + N_obj + B, 1, 16)
        tokens_s:     (N_hits + N_obj + B, S)
        tokens_batch: (N_hits + N_obj + B,)
        """
        # -------------------------
        # Unpack
        # -------------------------
        hit_mv, hit_s = hit_tokens
        (obj_mv, obj_s), obj_batch = object_tokens
        q_mv, q_s = query_token

        device = hit_s.device
        dtype_s = hit_s.dtype

        # -------------------------
        # Normalize shapes
        # -------------------------
        # mv expected (N,1,16). If (N,16), convert.
        if hit_mv.dim() == 2:
            hit_mv = hit_mv.unsqueeze(1)
        if obj_mv.dim() == 2:
            obj_mv = obj_mv.unsqueeze(1)
        if q_mv.dim() == 2:
            q_mv = q_mv.unsqueeze(1)

        # Ensure everything is on the same device
        obj_mv = obj_mv.to(device)
        obj_s = obj_s.to(device)
        obj_batch = obj_batch.to(device)

        q_mv = q_mv.to(device)
        q_s = q_s.to(device)

        batch = batch.to(device)

        # -------------------------
        # Build one QUERY per event
        # -------------------------
        # Number of events in the current batch
        active_idx = torch.nonzero(active_events, as_tuple=False).squeeze(-1)
        
        if active_idx.numel() > 0:
            q_mv_rep = q_mv.expand(active_idx.numel(), -1, -1).contiguous()
            q_s_rep = q_s.expand(active_idx.numel(), -1).contiguous()
            q_batch = active_idx.to(batch.dtype)
        else:
            q_mv_rep = hit_mv.new_zeros(0, 1, 16)
            q_s_rep = hit_s.new_zeros(0, hit_s.size(1))
            q_batch = batch.new_zeros(0)

        # -------------------------
        # Concatenate tokens
        # -------------------------
        tokens_mv = torch.cat([hit_mv, obj_mv, q_mv_rep], dim=0)
        tokens_s = torch.cat([hit_s, obj_s, q_s_rep.to(dtype_s)], dim=0)
        tokens_batch = torch.cat([batch, obj_batch, q_batch], dim=0)

        return tokens_mv, tokens_s, tokens_batch


    # ============================
    # DECODER STEP
    # ============================
    def _autoregressive_step(self, tokens_mv, tokens_s, tokens_batch):
        """
        Run one GATr decoder step over all tokens.
        """
        # As tokens_mv are embedded, mv_v_part and mv_s_part should be zero tensors
        _, scalar_out, point, scalar = self.autoregressive_module(mv_v_part=None, mv_s_part=None,
                                                              scalars=tokens_s,
                                                              batch=tokens_batch,
                                                              embedded_geom=tokens_mv)
        return (point, scalar, scalar_out, tokens_batch)

    def _extract_query_embedding(self, decoded_tokens):
        """
        Extract embedding corresponding to QUERY token.
        """
        point, scalar, scalar_out, tokens_batch = decoded_tokens
        # QUERY token is the last token per event
        query_indices = []
        for b in torch.sort(torch.unique(tokens_batch)).values:
            mask_evt = (tokens_batch == b)
            evt_indices = mask_evt.nonzero(as_tuple=False).squeeze(-1)
            query_idx = evt_indices[-1]  # Último token es QUERY
            query_indices.append(query_idx.item())
        query_indices = torch.tensor(query_indices, device=point.device)
        query_point = point[query_indices]    # (B, 3)
        query_scalar = scalar[query_indices]  # (B, S)
        query_scalar_out = scalar_out[query_indices]  # (B, S_out)
        return (query_point, query_scalar, query_scalar_out), tokens_batch

    # ============================
    # HEADS
    # ============================
    def _predict_pfo_properties(self, query_embedding, tokens_batch, active_events, training: bool):
        query_point, query_scalar, query_scalar_out = query_embedding

        direction = query_point / (query_point.norm(dim=-1, keepdim=True) + 1e-8)
        scalar_cond = torch.cat([query_scalar_out, query_scalar], dim=-1)

        p_mod = torch.nn.functional.softplus(self.p_head(scalar_cond))
        momentum = direction * p_mod
        pid_logits = self.pid_head(scalar_cond)
        charge_logits = self.charge_head(scalar_cond)

        # eventos correspondientes a los query tokens
        pfo_batch = torch.sort(torch.unique(tokens_batch)).values  # (B,)

        # En training: devolver TODOS los eventos (la loss manejará qué es válido)
        # En inferencia: filtrar por eventos activos
        if training:
            return {
                "batch": pfo_batch,
                "momentum": momentum,
                "p_mod": p_mod,
                "pid": pid_logits,
                "charge": charge_logits,
            }
        else:
            active_mask = active_events[pfo_batch]
            return {
                "batch": pfo_batch[active_mask],
                "momentum": momentum[active_mask],
                "p_mod": p_mod[active_mask],
                "pid": pid_logits[active_mask],
                "charge": charge_logits[active_mask],
            }

    def _predict_assignment(self, query_embedding, dec_output_hits, batch, residual, active_events, training: bool):
        """
        Predict soft assignment of hits to current PFO.
        """
        _, query_scalar, query_scalar_out = query_embedding
        # query_scalar: (B,1)
        # query_scalar_out: (B,S)

        hit_point, hit_scalar, hit_scalar_out = dec_output_hits
        # hit_scalar: (N,S)
        # residual: (N,1)

        device = hit_scalar.device
        N = hit_scalar.size(0)

        # Expand query embeddings to hit level
        query_scalar_out_exp = query_scalar_out[batch]  # (N,S)
        query_scalar_exp = query_scalar[batch]          # (N,1)

        # Build assignment features
        assign_feat = torch.cat(
            [   hit_point, # (N,3)
                hit_scalar, # (N, 1)
                hit_scalar_out, # (N,S)
                query_scalar_out_exp, # (N,S)
                query_scalar_exp, # (N,1)
                residual # (N,1)
            ],
            dim=-1
        )

        # Predict contribution
        logits = self.assignment_head(assign_feat)  # (N,1)
        assignment = torch.sigmoid(logits)

        # Mask with residual (very important)
        assignment = assignment * residual
        
        # anular hits de eventos ya parados (solo en inferencia)
        if not training:
            hit_active = active_events[batch]  # (N,)
            assignment = assignment * hit_active.unsqueeze(-1)
        return assignment, logits


    def _predict_stop(self, query_embedding, residual, batch):
        """
        Predict STOP probability per event.
        """
        _, query_scalar, query_scalar_out = query_embedding
        # query_scalar: (B,1)
        # query_scalar_out: (B,S)

        # Residual per event
        residual_evt = scatter_mean(residual, batch, dim=0)  # (B,1)

        stop_feat = torch.cat(
            [
                query_scalar_out,
                query_scalar,
                residual_evt
            ],
            dim=-1
        )

        stop_logits = self.stop_head(stop_feat)  # (B,1)
        stop_prob = torch.sigmoid(stop_logits)

        return stop_prob, stop_logits


    # ============================
    # STATE UPDATE
    # ============================
    def _update_residual(
        self,
        residual,
        assignment,
        training: bool,
        step_idx: int,
        hit_to_pfo: Optional[torch.Tensor] = None,
    ):
        """
        Update residual per hit.

        Training:
        - Use GT hit_to_pfo
        - Fully remove hits belonging to current PFO (hard masking)

        Inference:
        - Use soft predicted assignment
        - residual <- residual * (1 - assignment)

        Parameters
        ----------
        residual : (N,1)
        assignment : (N,1)
        step_idx : int
            Index of the current autoregressive PFO
        hit_to_pfo : (N,) or None
            Ground-truth hit → pfo index (global per event)
        """

        residual = residual.clamp(0.0, 1.0)

        # -------------------------
        # TRAINING: hard residual update using GT
        # -------------------------
        if training:
            if hit_to_pfo is None:
                raise ValueError("hit_to_pfo must be provided during training")

            # hits belonging to the current PFO
            hit_mask = (hit_to_pfo == step_idx)  # (N,)

            # clone to avoid in-place autograd issues
            new_residual = residual.clone()
            new_residual[hit_mask] = 0.0

            return new_residual

        # -------------------------
        # INFERENCE: soft residual update
        # -------------------------
        assignment = assignment.clamp(0.0, 1.0)

        new_residual = residual * (1.0 - assignment)
        new_residual = new_residual.clamp(0.0, 1.0)

        return new_residual

    def _append_pfo(self, pfo_list, pfo):
        """
        Append newly generated PFO to the sequence.
        """
        pfo_list.append(pfo)

    def _should_stop(self, stop_prob, threshold: float = 0.5):
        """
        stop_prob: (B,1) o (B,)
        Devuelve True si TODOS los eventos del batch quieren parar.
        """
        if stop_prob.dim() == 2:
            stop_prob = stop_prob.squeeze(-1)

        return bool((stop_prob > threshold).all().item())

    # ============================
    # OUTPUT
    # ============================
    def _format_output(self, pfo_list, assignments, stop_probs, assignments_logits, stop_logits):
        """
        pfo_list: list length T, cada elemento:
        {
            "momentum": (B,3),
            "p_mod": (B,1),
            "pid": (B,5),
            "charge": (B,1),
        }

        assignments: list length T, cada elemento (N_hits,1)
        stop_probs: list length T, cada elemento (B,1)
        assignments_logits: list length T, cada elemento (N_hits,1)
        stop_logits: list length T, cada elemento (B,1)
        """
        if len(pfo_list) == 0:
            # caso extremo: no se generó nada
            return {
                "pfo_momentum": None,
                "pfo_p_mod": None,
                "pfo_pid": None,
                "pfo_charge": None,
                "assignments": None,
                "stop_probs": None,
                "assignments_logits": None,
                "stop_logits": None,
            }
        # for p in pfo_list:
            # print(p.keys())
        pfo_momentum = torch.stack([p["momentum"] for p in pfo_list], dim=0)        # (T,B,3)
        pfo_p_mod = torch.stack([p["p_mod"] for p in pfo_list], dim=0)              # (T,B,1)
        pfo_pid_logits = torch.stack([p["pid"] for p in pfo_list], dim=0)    # (T,B,5)
        pfo_charge_logits = torch.stack([p["charge"] for p in pfo_list], dim=0)  # (T,B,3)
        pfo_batch = torch.stack([p["batch"] for p in pfo_list], dim=0)                # (T,B)
        assignments_t = None
        assignments_logits_t = None
        if len(assignments) > 0:
            assignments_t = torch.stack(assignments, dim=0)  # (T,N,1)
            assignments_logits_t = torch.stack(assignments_logits, dim=0)  # (T,N,1)
        
        stop_probs_t = None
        stop_logits_t = None
        if len(stop_probs) > 0:
            stop_probs_t = torch.stack(stop_probs, dim=0)  # (T,B,1)
            stop_logits_t = torch.stack(stop_logits, dim=0)  # (T,B,1)
        return {
            "pfo_momentum": pfo_momentum,
            "pfo_p_mod": pfo_p_mod,
            "pfo_pid": pfo_pid_logits,
            "pfo_charge": pfo_charge_logits,
            "pfo_batch": pfo_batch,
            "assignments": assignments_t,
            "stop_probs": stop_probs_t,
            "assignments_logits": assignments_logits_t,
            "stop_logits": stop_logits_t,
        }
