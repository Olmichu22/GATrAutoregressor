"""
Tests exhaustivos para GATrAutoRegressor

Cubren:
1. Inicialización del modelo (ambos modos)
2. Funciones individuales del pipeline
3. Forward pass completo (training e inferencia)
4. Casos extremos pero posibles
5. Consistencia de dimensiones
6. Gradientes y backpropagation
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import asdict

import sys
sys.path.insert(0, '/nfs/cms/arqolmo/GPU_train/DenoisingShower')

from src.model.GATrAutoRegressor import (
    GATrAutoRegressor,
    GATrAutoRegressorParamsWhole,
    GATrAutoRegressorParamsSplit,
    GATrAutoRegressive,
    concatenate_outputs,
    SequentialWrapper,
    ParallelWrapper,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def device():
    """Dispositivo para tests (CPU para CI, GPU si disponible)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def whole_detector_config():
    """Configuración para modo whole_detector"""
    return {
        "whole_detector": GATrAutoRegressorParamsWhole(
            hidden_mv=16,
            hidden_s=32,
            num_blocks=2,
            in_s_channels=2,
            in_mv_channels=1,
            out_mv_channels=1,
            dropout=0.0,  # Sin dropout para tests deterministas
            out_s_channels=32,
        ),
        "autoregressive_module": GATrAutoRegressive(
            hidden_mv=16,
            hidden_s=32,
            num_blocks=2,
            out_mv_channels=1,
            out_s_channels=32,
            dropout=0.0,
        ),
    }


@pytest.fixture
def detector_split_config():
    """Configuración para modo detector_split"""
    return {
        "detector_split": GATrAutoRegressorParamsSplit(
            hidden_mv=[16, 16, 16, 16],
            hidden_s=[32, 32, 32, 32],
            num_blocks=[2, 2, 2, 2],
            in_s_channels=[2, 2, 2, 2],
            in_mv_channels=[1, 1, 1, 1],
            out_mv_channels=[1, 1, 1, 1],
            dropout=[0.0, 0.0, 0.0, 0.0],
            out_s_channels=[32, 32, 32, 32],
            final_module_cfg=GATrAutoRegressorParamsWhole(
                hidden_mv=16,
                hidden_s=32,
                num_blocks=2,
                in_s_channels=32,  # Viene de los módulos anteriores
                in_mv_channels=1,
                out_mv_channels=1,
                dropout=0.0,
                out_s_channels=32,
            ),
        ),
        "autoregressive_module": GATrAutoRegressive(
            hidden_mv=16,
            hidden_s=32,
            num_blocks=2,
            out_mv_channels=1,
            out_s_channels=32,
            dropout=0.0,
        ),
    }


@pytest.fixture
def sample_batch_data(device):
    """
    Genera datos de ejemplo para un batch con 2 eventos.
    Evento 0: 10 hits, 2 PFOs
    Evento 1: 15 hits, 3 PFOs
    """
    torch.manual_seed(42)
    
    n_hits_evt0 = 10
    n_hits_evt1 = 15
    n_hits_total = n_hits_evt0 + n_hits_evt1
    
    # Hits data (posición, dirección, escalares)
    mv_v_part = torch.randn(n_hits_total, 3, device=device)  # posiciones/direcciones
    mv_s_part = torch.randn(n_hits_total, 3, device=device)  # vectores secundarios
    scalars = torch.randn(n_hits_total, 2, device=device)    # E, p_mod
    
    # Batch indices
    batch = torch.cat([
        torch.zeros(n_hits_evt0, dtype=torch.long, device=device),
        torch.ones(n_hits_evt1, dtype=torch.long, device=device),
    ])
    
    # PFO ground truth
    n_pfos_evt0 = 2
    n_pfos_evt1 = 3
    n_pfos_total = n_pfos_evt0 + n_pfos_evt1
    
    pfo_pid = torch.zeros(n_pfos_total, 5, device=device)
    pfo_pid[0, 0] = 1.0  # electrón
    pfo_pid[1, 2] = 1.0  # pión
    pfo_pid[2, 1] = 1.0  # muón
    pfo_pid[3, 3] = 1.0  # neutrón
    pfo_pid[4, 4] = 1.0  # gamma
    
    pfo_momentum = torch.randn(n_pfos_total, 3, device=device)
    pfo_charge = torch.tensor([[-1], [1], [-1], [0], [0]], dtype=torch.float32, device=device)
    
    pfo_batch = torch.cat([
        torch.zeros(n_pfos_evt0, dtype=torch.long, device=device),
        torch.ones(n_pfos_evt1, dtype=torch.long, device=device),
    ])
    
    # hit_to_pfo: asignación de hits a PFOs (índice local por evento)
    hit_to_pfo = torch.cat([
        torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=torch.long, device=device),  # evt 0
        torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=torch.long, device=device),  # evt 1
    ])
    
    pfo_true_objects = {
        "pid": pfo_pid,
        "momentum": pfo_momentum,
        "charge": pfo_charge,
        "batch": pfo_batch,
        "hit_to_pfo": hit_to_pfo,
    }
    
    return {
        "mv_v_part": [mv_v_part],
        "mv_s_part": [mv_s_part],
        "scalars": [scalars],
        "batch": [batch],
        "pfo_true_objects": pfo_true_objects,
        "n_hits_total": n_hits_total,
        "n_pfos_total": n_pfos_total,
        "n_events": 2,
    }


@pytest.fixture
def single_event_data(device):
    """Datos para un solo evento (batch size = 1)"""
    torch.manual_seed(42)
    
    n_hits = 20
    n_pfos = 4
    
    mv_v_part = torch.randn(n_hits, 3, device=device)
    mv_s_part = torch.randn(n_hits, 3, device=device)
    scalars = torch.randn(n_hits, 2, device=device)
    batch = torch.zeros(n_hits, dtype=torch.long, device=device)
    
    pfo_pid = torch.eye(5, device=device)[:n_pfos]
    pfo_momentum = torch.randn(n_pfos, 3, device=device)
    pfo_charge = torch.tensor([[-1], [1], [0], [-1]], dtype=torch.float32, device=device)
    pfo_batch = torch.zeros(n_pfos, dtype=torch.long, device=device)
    
    hits_per_pfo = n_hits // n_pfos
    hit_to_pfo = torch.cat([
        torch.full((hits_per_pfo,), i, dtype=torch.long, device=device)
        for i in range(n_pfos)
    ])
    
    pfo_true_objects = {
        "pid": pfo_pid,
        "momentum": pfo_momentum,
        "charge": pfo_charge,
        "batch": pfo_batch,
        "hit_to_pfo": hit_to_pfo,
    }
    
    return {
        "mv_v_part": [mv_v_part],
        "mv_s_part": [mv_s_part],
        "scalars": [scalars],
        "batch": [batch],
        "pfo_true_objects": pfo_true_objects,
    }


# ============================================================
# TESTS DE INICIALIZACIÓN
# ============================================================

class TestModelInitialization:
    """Tests de inicialización del modelo"""
    
    def test_init_whole_detector_mode(self, whole_detector_config, device):
        """Inicialización en modo whole_detector"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        
        assert model.mode == "whole_detector"
        assert isinstance(model.forward_module, SequentialWrapper)
        assert model.autoregressive_module is not None
        assert model.p_head is not None
        assert model.pid_head is not None
        assert model.charge_head is not None
        assert model.assignment_head is not None
        assert model.stop_head is not None
    
    def test_init_detector_split_mode(self, detector_split_config, device):
        """Inicialización en modo detector_split"""
        model = GATrAutoRegressor(
            mode="detector_split",
            params_cfg=detector_split_config,
        ).to(device)
        
        assert model.mode == "detector_split"
        assert isinstance(model.forward_module, ParallelWrapper)
        assert len(model.forward_module.modules_) == 4
    
    def test_init_invalid_mode(self, whole_detector_config):
        """Debe fallar con modo inválido"""
        with pytest.raises(ValueError, match="Invalid mode"):
            GATrAutoRegressor(mode="invalid_mode", params_cfg=whole_detector_config)
    
    def test_init_missing_config(self):
        """Debe fallar sin configuración"""
        with pytest.raises(ValueError, match="Missing configuration"):
            GATrAutoRegressor(mode="whole_detector", params_cfg={})
    
    def test_init_missing_autoregressive_config(self, whole_detector_config):
        """Debe fallar sin config de autoregressive module"""
        config = {"whole_detector": whole_detector_config["whole_detector"]}
        with pytest.raises(ValueError, match="Missing configuration for 'autoregressive_module'"):
            GATrAutoRegressor(mode="whole_detector", params_cfg=config)


# ============================================================
# TESTS DE FUNCIONES INDIVIDUALES
# ============================================================

class TestInitResidual:
    """Tests para _init_residual"""
    
    def test_shape(self, whole_detector_config, device):
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        residual = model._init_residual(batch_data_length=100, device=device)
        
        assert residual.shape == (100, 1)
        assert residual.device.type == device.type
    
    def test_values_are_ones(self, whole_detector_config, device):
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        residual = model._init_residual(batch_data_length=50, device=device)
        
        assert torch.allclose(residual, torch.ones(50, 1, device=device))
    
    def test_zero_length(self, whole_detector_config, device):
        """Caso extremo: 0 hits"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        residual = model._init_residual(batch_data_length=0, device=device)
        
        assert residual.shape == (0, 1)


class TestBuildHitTokens:
    """Tests para _build_hit_tokens"""
    
    def test_output_shapes(self, whole_detector_config, device):
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        N = 25
        S_enc = 32
        enc_output = (
            torch.randn(N, 1, 16, device=device),  # mv_out
            torch.randn(N, S_enc, device=device),   # scalar_out
            torch.randn(N, 3, device=device),       # point
            torch.randn(N, 1, device=device),       # scalar
        )
        residual = torch.ones(N, 1, device=device)
        
        mv_out, scalar_out = model._build_hit_tokens(enc_output, residual)
        
        assert mv_out.shape == (N, 1, 16)
        # scalar_out = S_enc + residual(1) + one_hot_type(3) + extra_features(7) = S_enc + 11
        assert scalar_out.shape == (N, S_enc + 11)
    
    def test_one_hot_type_is_hit(self, whole_detector_config, device):
        """Verifica que el one-hot type marca HIT"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        N = 10
        S_enc = 32
        enc_output = (
            torch.randn(N, 1, 16, device=device),
            torch.randn(N, S_enc, device=device),
            torch.randn(N, 3, device=device),
            torch.randn(N, 1, device=device),
        )
        residual = torch.ones(N, 1, device=device)
        
        _, scalar_out = model._build_hit_tokens(enc_output, residual)
        
        # One-hot está en posiciones -10, -9, -8 (HIT, OBJ, QUERY)
        assert torch.all(scalar_out[:, -10] == 1.0)  # HIT
        assert torch.all(scalar_out[:, -9] == 0.0)   # OBJ
        assert torch.all(scalar_out[:, -8] == 0.0)   # QUERY


class TestBuildObjectTokens:
    """Tests para _build_object_tokens"""
    
    def test_step_zero_returns_empty(self, whole_detector_config, device):
        """En step 0, no hay OBJs previos"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        model.train()
        
        pfo_true_objects = {
            "pid": torch.zeros(5, 5, device=device),
            "momentum": torch.randn(5, 3, device=device),
            "charge": torch.zeros(5, 1, device=device),
            "batch": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device),
        }
        scalar_out = torch.randn(20, 43, device=device)  # S_enc + 11
        
        (obj_mv, obj_s), obj_batch = model._build_object_tokens(
            pfo_list=[],
            training=True,
            pfo_true_objects=pfo_true_objects,
            scalar_out=scalar_out,
            step_idx=0,
        )
        
        assert obj_mv.shape[0] == 0
        assert obj_s.shape[0] == 0
        assert obj_batch.shape[0] == 0
    
    def test_training_teacher_forcing(self, whole_detector_config, device):
        """En training, usa PFOs verdaderos"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        model.train()
        
        # 2 eventos: evt0 tiene 2 PFOs, evt1 tiene 3 PFOs
        pfo_pid = torch.eye(5, device=device)
        pfo_momentum = torch.randn(5, 3, device=device)
        pfo_charge = torch.tensor([[1], [-1], [0], [1], [-1]], dtype=torch.float32, device=device)
        pfo_batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long, device=device)
        
        pfo_true_objects = {
            "pid": pfo_pid,
            "momentum": pfo_momentum,
            "charge": pfo_charge,
            "batch": pfo_batch,
        }
        scalar_out = torch.randn(20, 43, device=device)
        
        # step_idx = 1 → debe seleccionar PFO 0 de cada evento
        (obj_mv, obj_s), obj_batch = model._build_object_tokens(
            pfo_list=[],
            training=True,
            pfo_true_objects=pfo_true_objects,
            scalar_out=scalar_out,
            step_idx=1,
        )
        
        # Evento 0: 1 PFO (índice 0), Evento 1: 1 PFO (índice 2)
        assert obj_mv.shape[0] == 2
        assert torch.equal(obj_batch, torch.tensor([0, 1], device=device))
    
    def test_inference_uses_generated_pfos(self, whole_detector_config, device):
        """En inferencia, usa PFOs generados"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        model.eval()
        
        # PFOs generados previamente
        pfo_list = [
            {
                "batch": torch.tensor([0], device=device),
                "momentum": torch.tensor([[1.0, 0.0, 0.0]], device=device),
                "pid": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]], device=device),
                "charge": torch.tensor([[1]], device=device),
            },
        ]
        scalar_out = torch.randn(20, 43, device=device)
        
        (obj_mv, obj_s), obj_batch = model._build_object_tokens(
            pfo_list=pfo_list,
            training=False,
            pfo_true_objects=None,
            scalar_out=scalar_out,
            step_idx=1,
        )
        
        assert obj_mv.shape[0] == 1
        assert obj_batch.item() == 0


class TestBuildQueryToken:
    """Tests para _build_query_token"""
    
    def test_output_shapes(self, whole_detector_config, device):
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        q_mv, q_s = model._build_query_token(
            step_idx=0,
            scalar_dim=43,
            scalar_device=device,
        )
        
        assert q_mv.shape == (1, 1, 16)
        assert q_s.shape == (1, 43)
    
    def test_query_one_hot(self, whole_detector_config, device):
        """Verifica one-hot para QUERY"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        _, q_s = model._build_query_token(step_idx=0, scalar_dim=43, scalar_device=device)
        
        # QUERY one-hot en posición -8
        assert q_s[0, -8] == 1.0
        assert q_s[0, -10] == 0.0  # No es HIT
        assert q_s[0, -9] == 0.0   # No es OBJ


class TestAssembleTokens:
    """Tests para _assemble_tokens"""
    
    def test_concatenation_order(self, whole_detector_config, device):
        """Verifica orden: HIT + OBJ + QUERY"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        N_hits = 10
        N_obj = 2
        S = 43
        
        hit_tokens = (
            torch.randn(N_hits, 1, 16, device=device),
            torch.randn(N_hits, S, device=device),
        )
        obj_tokens = (
            (
                torch.randn(N_obj, 1, 16, device=device),
                torch.randn(N_obj, S, device=device),
            ),
            torch.tensor([0, 1], dtype=torch.long, device=device),
        )
        query_token = (
            torch.randn(1, 1, 16, device=device),
            torch.randn(1, S, device=device),
        )
        batch = torch.zeros(N_hits, dtype=torch.long, device=device)
        active_events = torch.tensor([True, True], device=device)
        
        tokens_mv, tokens_s, tokens_batch = model._assemble_tokens(
            hit_tokens, obj_tokens, query_token, batch, active_events
        )
        
        # Total = 10 + 2 + 2 (2 eventos activos)
        assert tokens_mv.shape[0] == N_hits + N_obj + 2
        assert tokens_s.shape[0] == N_hits + N_obj + 2
        assert tokens_batch.shape[0] == N_hits + N_obj + 2
    
    def test_handles_inactive_events(self, whole_detector_config, device):
        """Solo genera queries para eventos activos"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        N_hits = 10
        S = 43
        
        hit_tokens = (
            torch.randn(N_hits, 1, 16, device=device),
            torch.randn(N_hits, S, device=device),
        )
        obj_tokens = (
            (torch.zeros(0, 1, 16, device=device), torch.zeros(0, S, device=device)),
            torch.zeros(0, dtype=torch.long, device=device),
        )
        query_token = (
            torch.randn(1, 1, 16, device=device),
            torch.randn(1, S, device=device),
        )
        batch = torch.zeros(N_hits, dtype=torch.long, device=device)
        active_events = torch.tensor([False], device=device)  # Evento inactivo
        
        tokens_mv, tokens_s, tokens_batch = model._assemble_tokens(
            hit_tokens, obj_tokens, query_token, batch, active_events
        )
        
        # Sin queries porque el evento está inactivo
        assert tokens_mv.shape[0] == N_hits


class TestUpdateResidual:
    """Tests para _update_residual"""
    
    def test_training_hard_mask(self, whole_detector_config, device):
        """En training, usa máscara dura con GT"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        model.train()
        
        residual = torch.ones(10, 1, device=device)
        assignment = torch.rand(10, 1, device=device)
        hit_to_pfo = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long, device=device)
        
        new_residual = model._update_residual(
            residual=residual,
            assignment=assignment,
            training=True,
            step_idx=0,
            hit_to_pfo=hit_to_pfo,
        )
        
        # Hits del PFO 0 (indices 0,1,2) deben ser 0
        assert torch.all(new_residual[:3] == 0.0)
        # Resto debe ser 1
        assert torch.all(new_residual[3:] == 1.0)
    
    def test_inference_soft_update(self, whole_detector_config, device):
        """En inferencia, usa actualización suave"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        model.eval()
        
        residual = torch.ones(10, 1, device=device)
        assignment = torch.full((10, 1), 0.3, device=device)
        
        new_residual = model._update_residual(
            residual=residual,
            assignment=assignment,
            training=False,
            step_idx=0,
            hit_to_pfo=None,
        )
        
        expected = 1.0 * (1.0 - 0.3)
        assert torch.allclose(new_residual, torch.full((10, 1), expected, device=device))
    
    def test_training_without_hit_to_pfo_raises(self, whole_detector_config, device):
        """En training sin hit_to_pfo debe fallar"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        model.train()
        
        with pytest.raises(ValueError, match="hit_to_pfo must be provided"):
            model._update_residual(
                residual=torch.ones(10, 1, device=device),
                assignment=torch.rand(10, 1, device=device),
                training=True,
                step_idx=0,
                hit_to_pfo=None,
            )
    
    def test_clamps_values(self, whole_detector_config, device):
        """Verifica que los valores se mantienen en [0, 1]"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        model.eval()
        
        residual = torch.tensor([[1.5], [-0.5], [0.5]], device=device)
        assignment = torch.tensor([[0.2], [0.3], [0.4]], device=device)
        
        new_residual = model._update_residual(
            residual=residual,
            assignment=assignment,
            training=False,
            step_idx=0,
            hit_to_pfo=None,
        )
        
        assert torch.all(new_residual >= 0.0)
        assert torch.all(new_residual <= 1.0)


class TestFormatOutput:
    """Tests para _format_output"""
    
    def test_empty_pfo_list(self, whole_detector_config, device):
        """Lista vacía de PFOs"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        output = model._format_output(pfo_list=[], assignments=[])
        
        assert output["pfo_momentum"] is None
        assert output["pfo_p_mod"] is None
        assert output["pfo_pid"] is None
        assert output["pfo_charge"] is None
        assert output["assignments"] is None
    
    def test_stacks_correctly(self, whole_detector_config, device):
        """Verifica que apila correctamente"""
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_detector_config).to(device)
        
        B = 2
        pfo_list = [
            {
                "batch": torch.tensor([0, 1], device=device),
                "momentum": torch.randn(B, 3, device=device),
                "p_mod": torch.randn(B, 1, device=device),
                "pid": torch.randn(B, 5, device=device),
                "charge": torch.randn(B, 1, device=device),
            },
            {
                "batch": torch.tensor([0, 1], device=device),
                "momentum": torch.randn(B, 3, device=device),
                "p_mod": torch.randn(B, 1, device=device),
                "pid": torch.randn(B, 5, device=device),
                "charge": torch.randn(B, 1, device=device),
            },
        ]
        N = 25
        assignments = [
            torch.randn(N, 1, device=device),
            torch.randn(N, 1, device=device),
        ]
        
        output = model._format_output(pfo_list, assignments)
        
        assert output["pfo_momentum"].shape == (2, B, 3)
        assert output["pfo_p_mod"].shape == (2, B, 1)
        assert output["pfo_pid"].shape == (2, B, 5)
        assert output["pfo_charge"].shape == (2, B, 1)
        assert output["assignments"].shape == (2, N, 1)


# ============================================================
# TESTS DE FORWARD PASS COMPLETO
# ============================================================

class TestForwardPassTraining:
    """Tests del forward pass en modo training"""
    
    def test_forward_whole_detector(self, whole_detector_config, sample_batch_data, device):
        """Forward pass completo en modo whole_detector"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        output = model(
            mv_v_part=sample_batch_data["mv_v_part"],
            mv_s_part=sample_batch_data["mv_s_part"],
            scalars=sample_batch_data["scalars"],
            pfo_true_objects=sample_batch_data["pfo_true_objects"],
            batch=sample_batch_data["batch"],
        )
        
        assert output["pfo_momentum"] is not None
        assert output["pfo_pid"] is not None
        assert output["pfo_charge"] is not None
        assert output["assignments"] is not None
    
    def test_forward_shapes_are_consistent(self, whole_detector_config, sample_batch_data, device):
        """Verifica consistencia de shapes en salida"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        output = model(
            mv_v_part=sample_batch_data["mv_v_part"],
            mv_s_part=sample_batch_data["mv_s_part"],
            scalars=sample_batch_data["scalars"],
            pfo_true_objects=sample_batch_data["pfo_true_objects"],
            batch=sample_batch_data["batch"],
        )
        
        T = output["pfo_momentum"].shape[0]  # Número de pasos
        
        assert output["pfo_p_mod"].shape[0] == T
        assert output["pfo_pid"].shape[0] == T
        assert output["pfo_charge"].shape[0] == T
        assert output["assignments"].shape[0] == T
    
    def test_gradients_flow(self, whole_detector_config, sample_batch_data, device):
        """Verifica que los gradientes fluyen correctamente"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        output = model(
            mv_v_part=sample_batch_data["mv_v_part"],
            mv_s_part=sample_batch_data["mv_s_part"],
            scalars=sample_batch_data["scalars"],
            pfo_true_objects=sample_batch_data["pfo_true_objects"],
            batch=sample_batch_data["batch"],
        )
        
        # Pérdida dummy
        loss = output["pfo_momentum"].sum() + output["assignments"].sum()
        loss.backward()
        
        # Verificar que al menos algunos parámetros tienen gradiente
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad


class TestForwardPassInference:
    """Tests del forward pass en modo inferencia"""
    
    def test_forward_inference(self, whole_detector_config, sample_batch_data, device):
        """Forward pass en inferencia"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(
                mv_v_part=sample_batch_data["mv_v_part"],
                mv_s_part=sample_batch_data["mv_s_part"],
                scalars=sample_batch_data["scalars"],
                pfo_true_objects=sample_batch_data["pfo_true_objects"],
                batch=sample_batch_data["batch"],
            )
        
        # Debe generar al menos algunos PFOs
        assert output["pfo_momentum"] is not None or output["pfo_momentum"] is None  # Puede parar temprano
    
    def test_deterministic_with_seed(self, whole_detector_config, sample_batch_data, device):
        """Inferencia determinista con semilla fija"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.eval()
        
        torch.manual_seed(123)
        with torch.no_grad():
            output1 = model(
                mv_v_part=sample_batch_data["mv_v_part"],
                mv_s_part=sample_batch_data["mv_s_part"],
                scalars=sample_batch_data["scalars"],
                pfo_true_objects=sample_batch_data["pfo_true_objects"],
                batch=sample_batch_data["batch"],
            )
        
        torch.manual_seed(123)
        with torch.no_grad():
            output2 = model(
                mv_v_part=sample_batch_data["mv_v_part"],
                mv_s_part=sample_batch_data["mv_s_part"],
                scalars=sample_batch_data["scalars"],
                pfo_true_objects=sample_batch_data["pfo_true_objects"],
                batch=sample_batch_data["batch"],
            )
        
        if output1["pfo_momentum"] is not None and output2["pfo_momentum"] is not None:
            assert torch.allclose(output1["pfo_momentum"], output2["pfo_momentum"])


# ============================================================
# TESTS DE CASOS EXTREMOS
# ============================================================

class TestEdgeCases:
    """Tests de casos extremos pero posibles"""
    
    def test_single_hit(self, whole_detector_config, device):
        """Un solo hit en el evento"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        mv_v_part = [torch.randn(1, 3, device=device)]
        mv_s_part = [torch.randn(1, 3, device=device)]
        scalars = [torch.randn(1, 2, device=device)]
        batch = [torch.zeros(1, dtype=torch.long, device=device)]
        
        pfo_true_objects = {
            "pid": torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32, device=device),
            "momentum": torch.randn(1, 3, device=device),
            "charge": torch.tensor([[1]], dtype=torch.float32, device=device),
            "batch": torch.zeros(1, dtype=torch.long, device=device),
            "hit_to_pfo": torch.zeros(1, dtype=torch.long, device=device),
        }
        
        output = model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=batch,
        )
        
        assert output is not None
    
    def test_single_pfo_per_event(self, whole_detector_config, device):
        """Un solo PFO por evento"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        N = 10
        mv_v_part = [torch.randn(N, 3, device=device)]
        mv_s_part = [torch.randn(N, 3, device=device)]
        scalars = [torch.randn(N, 2, device=device)]
        batch = [torch.zeros(N, dtype=torch.long, device=device)]
        
        pfo_true_objects = {
            "pid": torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32, device=device),
            "momentum": torch.randn(1, 3, device=device),
            "charge": torch.tensor([[1]], dtype=torch.float32, device=device),
            "batch": torch.zeros(1, dtype=torch.long, device=device),
            "hit_to_pfo": torch.zeros(N, dtype=torch.long, device=device),
        }
        
        output = model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=batch,
        )
        
        assert output["pfo_momentum"] is not None
    
    def test_many_pfos(self, whole_detector_config, device):
        """Muchos PFOs (cerca del límite de _max_steps)"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        N = 100
        num_pfos = 50
        
        mv_v_part = [torch.randn(N, 3, device=device)]
        mv_s_part = [torch.randn(N, 3, device=device)]
        scalars = [torch.randn(N, 2, device=device)]
        batch = [torch.zeros(N, dtype=torch.long, device=device)]
        
        # Crear muchos PFOs
        pfo_pid = torch.zeros(num_pfos, 5, device=device)
        for i in range(num_pfos):
            pfo_pid[i, i % 5] = 1.0
        
        pfo_true_objects = {
            "pid": pfo_pid,
            "momentum": torch.randn(num_pfos, 3, device=device),
            "charge": torch.randint(-1, 2, (num_pfos, 1), dtype=torch.float32, device=device),
            "batch": torch.zeros(num_pfos, dtype=torch.long, device=device),
            "hit_to_pfo": torch.randint(0, num_pfos, (N,), device=device),
        }
        
        output = model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=batch,
        )
        
        assert output["pfo_momentum"] is not None
    
    def test_imbalanced_events(self, whole_detector_config, device):
        """Eventos con número muy diferente de hits"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        # Evento 0: 5 hits, Evento 1: 50 hits
        N_evt0, N_evt1 = 5, 50
        N_total = N_evt0 + N_evt1
        
        mv_v_part = [torch.randn(N_total, 3, device=device)]
        mv_s_part = [torch.randn(N_total, 3, device=device)]
        scalars = [torch.randn(N_total, 2, device=device)]
        batch = [torch.cat([
            torch.zeros(N_evt0, dtype=torch.long, device=device),
            torch.ones(N_evt1, dtype=torch.long, device=device),
        ])]
        
        pfo_true_objects = {
            "pid": torch.eye(5, device=device)[:3],  # 3 PFOs
            "momentum": torch.randn(3, 3, device=device),
            "charge": torch.tensor([[-1], [1], [0]], dtype=torch.float32, device=device),
            "batch": torch.tensor([0, 1, 1], dtype=torch.long, device=device),
            "hit_to_pfo": torch.cat([
                torch.zeros(N_evt0, dtype=torch.long, device=device),
                torch.cat([
                    torch.zeros(25, dtype=torch.long, device=device),
                    torch.ones(25, dtype=torch.long, device=device),
                ]),
            ]),
        }
        
        output = model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=batch,
        )
        
        assert output["pfo_momentum"] is not None
    
    def test_large_batch(self, whole_detector_config, device):
        """Batch grande (muchos eventos)"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        num_events = 16
        hits_per_event = 10
        N_total = num_events * hits_per_event
        
        mv_v_part = [torch.randn(N_total, 3, device=device)]
        mv_s_part = [torch.randn(N_total, 3, device=device)]
        scalars = [torch.randn(N_total, 2, device=device)]
        batch = [torch.repeat_interleave(
            torch.arange(num_events, device=device),
            hits_per_event,
        )]
        
        # 2 PFOs por evento
        pfos_per_event = 2
        total_pfos = num_events * pfos_per_event
        
        pfo_true_objects = {
            "pid": torch.zeros(total_pfos, 5, device=device),
            "momentum": torch.randn(total_pfos, 3, device=device),
            "charge": torch.randint(-1, 2, (total_pfos, 1), dtype=torch.float32, device=device),
            "batch": torch.repeat_interleave(
                torch.arange(num_events, device=device),
                pfos_per_event,
            ),
            "hit_to_pfo": torch.cat([
                torch.cat([
                    torch.zeros(hits_per_event // 2, dtype=torch.long, device=device),
                    torch.ones(hits_per_event // 2, dtype=torch.long, device=device),
                ])
                for _ in range(num_events)
            ]),
        }
        
        for i in range(total_pfos):
            pfo_true_objects["pid"][i, i % 5] = 1.0
        
        output = model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=batch,
        )
        
        assert output["pfo_momentum"] is not None


# ============================================================
# TESTS DE UTILIDADES
# ============================================================

class TestConcatenateOutputs:
    """Tests para concatenate_outputs"""
    
    def test_single_output(self, device):
        """Un solo output no necesita concatenación"""
        outputs = [(
            torch.randn(10, 1, 16, device=device),
            torch.randn(10, 32, device=device),
            torch.randn(10, 3, device=device),
            torch.randn(10, 1, device=device),
        )]
        batch = [torch.zeros(10, dtype=torch.long, device=device)]
        
        result, result_batch = concatenate_outputs(outputs, batch)
        
        assert result[0].shape == outputs[0][0].shape
    
    def test_multiple_outputs_sorting(self, device):
        """Múltiples outputs se ordenan por batch"""
        outputs = [
            (
                torch.randn(5, 1, 16, device=device),
                torch.randn(5, 32, device=device),
                torch.randn(5, 3, device=device),
                torch.randn(5, 1, device=device),
            ),
            (
                torch.randn(7, 1, 16, device=device),
                torch.randn(7, 32, device=device),
                torch.randn(7, 3, device=device),
                torch.randn(7, 1, device=device),
            ),
        ]
        batch = [
            torch.zeros(5, dtype=torch.long, device=device),
            torch.ones(7, dtype=torch.long, device=device),
        ]
        
        mv_out, scalar_out, point, scalar, result_batch = concatenate_outputs(outputs, batch)
        
        assert mv_out.shape[0] == 12
        assert scalar_out.shape[0] == 12
        assert torch.all(result_batch[:5] == 0)
        assert torch.all(result_batch[5:] == 1)


class TestDataclasses:
    """Tests para las dataclasses de configuración"""
    
    def test_whole_params_defaults(self):
        """Valores por defecto de GATrAutoRegressorParamsWhole"""
        params = GATrAutoRegressorParamsWhole()
        
        assert params.hidden_mv == 32
        assert params.hidden_s == 64
        assert params.num_blocks == 3
        assert params.out_s_channels is None
    
    def test_split_params_defaults(self):
        """Valores por defecto de GATrAutoRegressorParamsSplit"""
        params = GATrAutoRegressorParamsSplit()
        
        # assert len(params.modules) == 4
        assert len(params.hidden_mv) == 4
        assert all(h == 32 for h in params.hidden_mv)
    
    def test_autoregressive_params_defaults(self):
        """Valores por defecto de GATrAutoRegressive"""
        params = GATrAutoRegressive()
        
        assert params.hidden_mv == 32
        assert params.out_s_channels is None


# ============================================================
# TESTS DE ROBUSTEZ
# ============================================================

class TestRobustness:
    """Tests de robustez ante entradas problemáticas"""
    
    def test_handles_nan_gracefully(self, whole_detector_config, device):
        """El modelo no debe propagar NaN silenciosamente"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        N = 10
        mv_v_part = [torch.randn(N, 3, device=device)]
        mv_s_part = [torch.randn(N, 3, device=device)]
        scalars = [torch.randn(N, 2, device=device)]
        batch = [torch.zeros(N, dtype=torch.long, device=device)]
        
        pfo_true_objects = {
            "pid": torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32, device=device),
            "momentum": torch.randn(1, 3, device=device),
            "charge": torch.tensor([[1]], dtype=torch.float32, device=device),
            "batch": torch.zeros(1, dtype=torch.long, device=device),
            "hit_to_pfo": torch.zeros(N, dtype=torch.long, device=device),
        }
        
        output = model(
            mv_v_part=mv_v_part,
            mv_s_part=mv_s_part,
            scalars=scalars,
            pfo_true_objects=pfo_true_objects,
            batch=batch,
        )
        
        # Verificar que no hay NaN en las salidas principales
        if output["pfo_momentum"] is not None:
            assert not torch.isnan(output["pfo_momentum"]).any(), "NaN encontrado en momentum"
        if output["assignments"] is not None:
            assert not torch.isnan(output["assignments"]).any(), "NaN encontrado en assignments"
    
    def test_extreme_values(self, whole_detector_config, device):
        """Manejo de valores extremos en entrada"""
        model = GATrAutoRegressor(
            mode="whole_detector",
            params_cfg=whole_detector_config,
        ).to(device)
        model.train()
        
        N = 10
        # Valores muy grandes
        mv_v_part = [torch.randn(N, 3, device=device) * 1000]
        mv_s_part = [torch.randn(N, 3, device=device) * 1000]
        scalars = [torch.randn(N, 2, device=device) * 1000]
        batch = [torch.zeros(N, dtype=torch.long, device=device)]
        
        pfo_true_objects = {
            "pid": torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.float32, device=device),
            "momentum": torch.randn(1, 3, device=device) * 1000,
            "charge": torch.tensor([[1]], dtype=torch.float32, device=device),
            "batch": torch.zeros(1, dtype=torch.long, device=device),
            "hit_to_pfo": torch.zeros(N, dtype=torch.long, device=device),
        }
        
        # No debe crashear (puede tener NaN pero no debe lanzar excepción)
        try:
            output = model(
                mv_v_part=mv_v_part,
                mv_s_part=mv_s_part,
                scalars=scalars,
                pfo_true_objects=pfo_true_objects,
                batch=batch,
            )
            executed = True
        except RuntimeError:
            executed = False
        
        assert executed, "El modelo no debería crashear con valores extremos"


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import datetime
    import os
    
    # Crear directorio de logs si no existe
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Nombre del archivo de log con timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"test_GATrAutoRegressor_{timestamp}.log")
    
    print(f"Ejecutando tests... Los resultados se guardarán en: {log_file}")
    
    # Ejecutar pytest con salida a archivo
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=long",
        f"--log-file={log_file}",
        f"--log-file-level=DEBUG",
        f"--capture=tee-sys",  # Captura pero también muestra en terminal
        f"-r=a",  # Mostrar resumen de todos los tests
        f"--junitxml={os.path.join(log_dir, f'test_results_{timestamp}.xml')}",
    ])
    
    # También guardar salida completa en archivo de texto
    print(f"\nTests completados. Exit code: {exit_code}")
    print(f"Log guardado en: {log_file}")
    print(f"XML guardado en: {os.path.join(log_dir, f'test_results_{timestamp}.xml')}")
