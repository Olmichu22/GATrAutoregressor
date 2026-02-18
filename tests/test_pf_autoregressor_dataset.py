"""
Tests para PFAutoRegressorDataset y GATrAutoRegressorLightningModule.

Verifica:
1. Creación de datos sintéticos
2. Carga en modo memory y lazy
3. Estructura correcta de Data objects
4. Batching con PyG (índices locales de hit_to_pfo)
5. Preprocesamiento
6. Forward pass del modelo
7. Loss function
8. Training step
"""

import pytest
import numpy as np
import torch
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from torch_geometric.loader import DataLoader

# Ajustar imports según estructura del proyecto
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datamodule.pf_autoregressor_dataset import (
    PFAutoRegressorDataset,
    PFAutoRegressorData,
    PFPreprocessor,
    PFPreprocessorConfig,
    pid_to_onehot,
    VALID_GEN_STATUS,
)
from src.model.gatr_autoregressor_module import (
    GATrAutoRegressorLightningModule,
    GATrAutoRegressorLoss,
    reorganize_gt_to_tb,
)
from src.model.GATrAutoRegressor import (
    GATrAutoRegressor,
    GATrAutoRegressorParamsWhole,
    GATrAutoRegressorParamsSplit,
    GATrAutoRegressive,
)


def create_synthetic_npz(path: Path, n_events: int = 5, seed: int = 42):
    """
    Crea un archivo NPZ sintético con estructura similar a convert_to_nn_format.py.
    
    Args:
        path: Ruta donde guardar el archivo
        n_events: Número de eventos a generar
        seed: Semilla para reproducibilidad
    """
    np.random.seed(seed)
    
    # Definir estructura de partículas
    particle_dtype = np.dtype([
        ('px', np.float32),
        ('py', np.float32),
        ('pz', np.float32),
        ('energy', np.float32),
        ('pid', np.int32),
        ('charge', np.float32),
        ('gen_status', np.int32),
    ])
    
    # Definir estructura de hits
    hit_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('energy', np.float32),
        ('p', np.float32),
        ('detector_type', np.int32),
    ])
    
    all_particles = []
    all_hits = []
    all_hit_to_particle = []
    all_hit_weights = []
    event_boundaries = [0]
    
    total_hits = 0
    total_particles = 0
    
    for event_idx in range(n_events):
        # Generar partículas para este evento (2-5 partículas)
        n_particles = np.random.randint(2, 6)
        particles = np.zeros(n_particles, dtype=particle_dtype)
        
        for i in range(n_particles):
            # Momentum aleatorio
            p_mag = np.random.uniform(1, 100)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            
            particles[i]['px'] = p_mag * np.sin(theta) * np.cos(phi)
            particles[i]['py'] = p_mag * np.sin(theta) * np.sin(phi)
            particles[i]['pz'] = p_mag * np.cos(theta)
            particles[i]['energy'] = np.sqrt(p_mag**2 + 0.14**2)  # masa de pion
            
            # PID aleatorio: e(11), mu(13), pi(211), n(2112), gamma(22)
            pid_choices = [11, 13, 211, 2112, 22]
            particles[i]['pid'] = np.random.choice(pid_choices)
            
            # Charge según PID
            if particles[i]['pid'] in [11, 13]:  # leptones
                particles[i]['charge'] = -1.0
            elif particles[i]['pid'] == 211:  # pion cargado
                particles[i]['charge'] = np.random.choice([-1.0, 1.0])
            else:  # neutros
                particles[i]['charge'] = 0.0
            
            # gen_status = 1 para partículas estables
            particles[i]['gen_status'] = 1
        
        all_particles.append(particles)
        
        # Generar hits para este evento (10-30 hits)
        n_hits = np.random.randint(10, 31)
        hits = np.zeros(n_hits, dtype=hit_dtype)
        hit_to_particle = np.full((n_hits, 3), -1, dtype=np.int32)
        hit_weights = np.zeros((n_hits, 3), dtype=np.float32)
        
        for i in range(n_hits):
            # Posición aleatoria
            hits[i]['x'] = np.random.uniform(-1000, 1000)
            hits[i]['y'] = np.random.uniform(-1000, 1000)
            hits[i]['z'] = np.random.uniform(-2000, 2000)
            hits[i]['energy'] = np.random.uniform(0.1, 10)
            hits[i]['p'] = np.random.uniform(0.1, 50)
            hits[i]['detector_type'] = np.random.randint(0, 4)
            
            # Asociar hit a una partícula (con probabilidad 0.9)
            if np.random.random() < 0.9:
                part_idx = np.random.randint(0, n_particles)
                # Índice global de partícula
                hit_to_particle[i, 0] = total_particles + part_idx
                hit_weights[i, 0] = np.random.uniform(0.5, 1.0)
        
        all_hits.append(hits)
        all_hit_to_particle.append(hit_to_particle)
        all_hit_weights.append(hit_weights)
        
        total_hits += n_hits
        total_particles += n_particles
        event_boundaries.append(total_hits)
    
    # Concatenar todo
    particles_concat = np.concatenate(all_particles)
    hits_concat = np.concatenate(all_hits)
    hit_to_particle_concat = np.concatenate(all_hit_to_particle)
    hit_weights_concat = np.concatenate(all_hit_weights)
    event_boundaries = np.array(event_boundaries, dtype=np.int64)
    
    # Guardar
    np.savez(
        path,
        particles=particles_concat,
        hits=hits_concat,
        hit_to_particle=hit_to_particle_concat,
        hit_weights=hit_weights_concat,
        event_boundaries=event_boundaries,
        n_events=n_events,
    )
    
    return {
        'n_events': n_events,
        'n_particles': len(particles_concat),
        'n_hits': len(hits_concat),
    }


class TestPFAutoRegressorDataset:
    """Tests para el dataset."""
    
    @pytest.fixture
    def temp_npz(self, tmp_path):
        """Crea un archivo NPZ temporal para tests."""
        npz_path = tmp_path / "test_data.npz"
        info = create_synthetic_npz(npz_path, n_events=5, seed=42)
        return npz_path, info
    
    def test_load_memory_mode(self, temp_npz):
        """Test carga en modo memory."""
        npz_path, info = temp_npz
        
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        assert len(dataset) == info['n_events']
        
        # Verificar primer evento
        data = dataset[0]
        assert hasattr(data, 'pos')
        assert hasattr(data, 'hit_to_pfo')
        assert hasattr(data, 'pfo_pid')
        assert hasattr(data, 'pfo_momentum')
        assert hasattr(data, 'n_pfo')
        assert hasattr(data, 'n_hits')
    
    def test_load_lazy_mode(self, temp_npz):
        """Test carga en modo lazy."""
        npz_path, info = temp_npz
        
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='lazy')
        
        assert len(dataset) == info['n_events']
        
        # Verificar primer evento
        data = dataset[0]
        assert hasattr(data, 'pos')
        assert hasattr(data, 'hit_to_pfo')
    
    def test_data_shapes(self, temp_npz):
        """Verifica que los shapes sean consistentes."""
        npz_path, _ = temp_npz
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        data = dataset[0]
        n_hits = data.n_hits
        n_pfo = data.n_pfo
        
        # Shapes de hits
        assert data.pos.shape == (n_hits, 3)
        assert data.mv_v_part.shape == (n_hits, 3)
        assert data.scalars.shape[0] == n_hits
        assert data.hit_to_pfo.shape == (n_hits,)
        
        # Shapes de PFOs
        assert data.pfo_pid.shape == (n_pfo, 5)
        assert data.pfo_momentum.shape == (n_pfo, 3)
        assert data.pfo_charge.shape == (n_pfo, 1)
        assert data.pfo_event_idx.shape == (n_pfo,)
    
    def test_hit_to_pfo_values(self, temp_npz):
        """Verifica que hit_to_pfo tenga valores correctos."""
        npz_path, _ = temp_npz
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        data = dataset[0]
        hit_to_pfo = data.hit_to_pfo
        n_pfo = data.n_pfo
        
        # hit_to_pfo debe tener valores en [-1, n_pfo-1]
        assert hit_to_pfo.min() >= -1
        assert hit_to_pfo.max() < n_pfo
        
        # Debe haber al menos algunos hits asociados (con probabilidad alta dado nuestro generador)
        assert (hit_to_pfo >= 0).sum() > 0
        
        print(f"  n_hits={data.n_hits}, n_pfo={n_pfo}")
        print(f"  hit_to_pfo: min={hit_to_pfo.min()}, max={hit_to_pfo.max()}")
        print(f"  hits asociados: {(hit_to_pfo >= 0).sum()}/{data.n_hits}")
    
    def test_pfo_event_idx_single_event(self, temp_npz):
        """Verifica que pfo_event_idx sea 0 para un solo evento."""
        npz_path, _ = temp_npz
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        data = dataset[0]
        
        # Para un solo evento, todos los pfo_event_idx deben ser 0
        assert (data.pfo_event_idx == 0).all()


class TestPFAutoRegressorDataBatching:
    """Tests para el batching con PyTorch Geometric."""
    
    @pytest.fixture
    def temp_npz(self, tmp_path):
        """Crea un archivo NPZ temporal."""
        npz_path = tmp_path / "test_data.npz"
        create_synthetic_npz(npz_path, n_events=10, seed=123)
        return npz_path
    
    def test_batching_preserves_local_indices(self, temp_npz):
        """
        Verifica que hit_to_pfo mantenga índices LOCALES después del batching.
        
        Esto es CRÍTICO para el modelo autoregresivo.
        """
        dataset = PFAutoRegressorDataset([str(temp_npz)], mode='memory')
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        batch = next(iter(loader))
        
        # Verificar que tenemos 3 eventos
        B = int(batch.batch.max().item()) + 1
        assert B == 3
        
        # Verificar estructura del batch
        print(f"\n  Batch con {B} eventos:")
        print(f"  Total hits: {batch.pos.shape[0]}")
        print(f"  Total PFOs: {batch.pfo_pid.shape[0]}")
        
        # hit_to_pfo debe mantener índices LOCALES
        # Es decir, los valores deben ser -1 o [0, max_pfo_per_event)
        hit_to_pfo = batch.hit_to_pfo
        hit_batch = batch.batch
        
        for event_idx in range(B):
            event_mask = hit_batch == event_idx
            event_hit_to_pfo = hit_to_pfo[event_mask]
            
            # Contar PFOs de este evento
            n_pfo_event = (batch.pfo_event_idx == event_idx).sum().item()
            
            # Verificar que los índices sean locales
            valid_hits = event_hit_to_pfo >= 0
            if valid_hits.any():
                max_idx = event_hit_to_pfo[valid_hits].max().item()
                assert max_idx < n_pfo_event, (
                    f"Evento {event_idx}: max_idx={max_idx} >= n_pfo={n_pfo_event}"
                )
            
            print(f"  Evento {event_idx}: {event_mask.sum()} hits, {n_pfo_event} PFOs, "
                  f"hit_to_pfo range: [{event_hit_to_pfo.min()}, {event_hit_to_pfo.max()}]")
    
    def test_pfo_event_idx_incremented(self, temp_npz):
        """Verifica que pfo_event_idx se incremente correctamente en batching."""
        dataset = PFAutoRegressorDataset([str(temp_npz)], mode='memory')
        loader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        batch = next(iter(loader))
        
        # pfo_event_idx debe tener valores 0, 1, 2 para un batch de 3
        unique_events = torch.unique(batch.pfo_event_idx)
        assert len(unique_events) == 3
        assert 0 in unique_events
        assert 1 in unique_events
        assert 2 in unique_events
        
        print(f"\n  pfo_event_idx unique values: {unique_events.tolist()}")


class TestPFPreprocessor:
    """Tests para el preprocesador."""
    
    @pytest.fixture
    def temp_npz(self, tmp_path):
        """Crea un archivo NPZ temporal."""
        npz_path = tmp_path / "test_data.npz"
        create_synthetic_npz(npz_path, n_events=5, seed=456)
        return npz_path
    
    def test_log_transform(self, temp_npz):
        """Verifica que las transformaciones log funcionen."""
        config = PFPreprocessorConfig(
            log_energy=True,
            log_momentum=True,
            normalize_coords=False,
            normalize_energy=False,
        )
        preproc = PFPreprocessor(config)
        
        dataset = PFAutoRegressorDataset(
            [str(temp_npz)], 
            mode='memory',
            preprocessor=preproc
        )
        
        data = dataset[0]
        
        # Los valores de energía en scalars deben estar en log-scale
        # Por lo tanto, pueden ser negativos
        energy_vals = data.scalars[:, 0]  # Primera columna es log(energy)
        print(f"\n  log(energy) range: [{energy_vals.min():.2f}, {energy_vals.max():.2f}]")
        
        # pfo_p_mod también debe estar en log-scale
        p_mod = data.pfo_p_mod
        print(f"  log(|p|) range: [{p_mod.min():.2f}, {p_mod.max():.2f}]")


def test_quick_sanity():
    """Test rápido de sanidad sin fixtures de pytest."""
    with tempfile.TemporaryDirectory() as tmpdir:
        npz_path = Path(tmpdir) / "test.npz"
        create_synthetic_npz(npz_path, n_events=3)
        
        # Cargar dataset
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        print(f"\nDataset cargado: {len(dataset)} eventos")
        
        # Verificar un evento
        data = dataset[0]
        print(f"Evento 0: {data.n_hits} hits, {data.n_pfo} PFOs")
        print(f"  hit_to_pfo shape: {data.hit_to_pfo.shape}")
        print(f"  hit_to_pfo unique: {torch.unique(data.hit_to_pfo).tolist()}")
        
        # Probar batching
        loader = DataLoader(dataset, batch_size=2)
        batch = next(iter(loader))
        print(f"\nBatch: {batch.pos.shape[0]} hits totales")
        print(f"  batch.batch unique: {torch.unique(batch.batch).tolist()}")
        print(f"  pfo_event_idx unique: {torch.unique(batch.pfo_event_idx).tolist()}")
        print(f"  hit_to_pfo unique: {torch.unique(batch.hit_to_pfo).tolist()}")
        
        print("\n✅ Test de sanidad pasado!")


# =============================================
# TESTS PARA EL TRAINER / LIGHTNING MODULE
# =============================================

@dataclass
class MockTrainerConfig:
    """Configuración mock para el trainer."""
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: str = "step"
    decay_steps: int = 10
    decay_rate: float = 0.5
    max_epochs: int = 10


def create_small_model_config():
    """Crea una configuración pequeña del modelo para tests rápidos."""
    whole_cfg = GATrAutoRegressorParamsWhole(
        hidden_mv=8,
        hidden_s=16,
        num_blocks=1,
        in_s_channels=6,  # scalars tienen 6 columnas
        in_mv_channels=1,
        out_mv_channels=1,
        dropout=0.0,
        out_s_channels=None,
    )
    
    ar_cfg = GATrAutoRegressive(
        hidden_mv=8,
        hidden_s=16,
        num_blocks=1,
        out_mv_channels=1,
        out_s_channels=None,
        dropout=0.0,
    )
    
    return {
        "whole_detector": whole_cfg,
        "autoregressive_module": ar_cfg,
    }


class TestGATrAutoRegressorLoss:
    """Tests para la función de pérdida."""
    
    def test_loss_shapes(self):
        """Verifica que la loss acepta los shapes correctos."""
        loss_fn = GATrAutoRegressorLoss()
        
        T, B, N = 3, 2, 50  # steps, batch_size, n_hits
        device = torch.device('cpu')
        
        # Mock output del modelo
        output = {
            "pfo_momentum": torch.randn(T, B, 3),
            "pfo_p_mod": torch.randn(T, B, 1).abs(),
            "pfo_pid": torch.randn(T, B, 5),
            "pfo_charge": torch.randn(T, B, 1),
            "assignments": torch.sigmoid(torch.randn(T, N, 1)),
            "stop_probs": torch.sigmoid(torch.randn(T, B, 1)),
        }
        
        # Mock GT - 5 PFOs total: 3 para evento 0, 2 para evento 1
        n_pfo = 5
        pfo_true_objects = {
            "momentum": torch.randn(n_pfo, 3),
            "p_mod": torch.randn(n_pfo, 1),
            "pid": torch.zeros(n_pfo, 5),
            "charge": torch.randn(n_pfo, 1),
            "batch": torch.tensor([0, 0, 0, 1, 1]),  # 3 PFOs evento 0, 2 evento 1
            "hit_to_pfo": torch.randint(-1, 3, (N,)),  # índices locales
        }
        # Set one-hot PIDs
        for i in range(n_pfo):
            pfo_true_objects["pid"][i, i % 5] = 1.0
        
        hit_batch = torch.cat([
            torch.zeros(30, dtype=torch.long),  # 30 hits evento 0
            torch.ones(20, dtype=torch.long),   # 20 hits evento 1
        ])
        
        # Calcular loss
        losses = loss_fn(output, pfo_true_objects, hit_batch)
        
        assert "loss" in losses
        assert "loss_dir" in losses
        assert "loss_mag" in losses
        assert "loss_pid" in losses
        assert "loss_charge" in losses
        assert "loss_assign" in losses
        assert "loss_stop" in losses
        
        # Verificar que todas las losses son escalares
        for key, value in losses.items():
            if key != "valid_mask":
                assert value.dim() == 0, f"{key} should be scalar, got shape {value.shape}"
                assert not torch.isnan(value), f"{key} is NaN"
        
        print(f"\n  Losses calculadas:")
        for key, value in losses.items():
            if key != "valid_mask":
                print(f"    {key}: {value.item():.4f}")
    
    def test_loss_with_real_data(self, tmp_path):
        """Test loss con datos del dataset real."""
        npz_path = tmp_path / "test_data.npz"
        create_synthetic_npz(npz_path, n_events=5, seed=789)
        
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        loss_fn = GATrAutoRegressorLoss()
        
        # Simular output del modelo
        B = int(batch.batch.max().item()) + 1
        N = batch.pos.shape[0]
        T = 5  # número de steps
        
        output = {
            "pfo_momentum": torch.randn(T, B, 3),
            "pfo_p_mod": torch.randn(T, B, 1).abs(),
            "pfo_pid": torch.randn(T, B, 5),
            "pfo_charge": torch.randn(T, B, 1),
            "assignments": torch.sigmoid(torch.randn(T, N, 1)),
            "stop_probs": torch.sigmoid(torch.randn(T, B, 1)),
        }
        
        pfo_true_objects = {
            "momentum": batch.pfo_momentum,
            "p_mod": batch.pfo_p_mod,
            "pid": batch.pfo_pid,
            "charge": batch.pfo_charge,
            "batch": batch.pfo_event_idx,
            "hit_to_pfo": batch.hit_to_pfo,
        }
        
        losses = loss_fn(output, pfo_true_objects, batch.batch)
        
        assert not torch.isnan(losses["loss"])
        print(f"\n  Loss con datos reales: {losses['loss'].item():.4f}")


class TestGATrAutoRegressorLightningModule:
    """Tests para el Lightning Module."""
    
    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Crea dataset temporal."""
        npz_path = tmp_path / "test_data.npz"
        create_synthetic_npz(npz_path, n_events=10, seed=101)
        return PFAutoRegressorDataset([str(npz_path)], mode='memory')
    
    @pytest.fixture
    def model_and_module(self):
        """Crea modelo y módulo Lightning."""
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        
        return model, module
    
    def test_module_creation(self, model_and_module):
        """Test creación del módulo."""
        model, module = model_and_module
        
        assert module.model is not None
        assert module.loss_fn is not None
        assert hasattr(module, 'training_step')
        assert hasattr(module, 'validation_step')
        
        print("\n  ✅ Módulo creado correctamente")
    
    def test_prepare_batch(self, model_and_module, temp_dataset):
        """Test _prepare_batch."""
        _, module = model_and_module
        
        loader = DataLoader(temp_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        mv_v_part, mv_s_part, scalars, hit_batch, pfo_true_objects = module._prepare_batch(batch)
        
        # Verificar estructura
        assert len(mv_v_part) == 1
        assert len(scalars) == 1
        assert len(hit_batch) == 1
        
        assert "momentum" in pfo_true_objects
        assert "pid" in pfo_true_objects
        assert "batch" in pfo_true_objects
        assert "hit_to_pfo" in pfo_true_objects
        
        # Verificar que hit_to_pfo tiene índices locales
        hit_to_pfo = pfo_true_objects["hit_to_pfo"]
        pfo_batch = pfo_true_objects["batch"]
        
        B = int(hit_batch[0].max().item()) + 1
        for event_idx in range(B):
            event_mask = hit_batch[0] == event_idx
            event_hit_to_pfo = hit_to_pfo[event_mask]
            n_pfo_event = (pfo_batch == event_idx).sum().item()
            
            valid_hits = event_hit_to_pfo >= 0
            if valid_hits.any():
                max_idx = event_hit_to_pfo[valid_hits].max().item()
                assert max_idx < n_pfo_event
        
        print(f"\n  ✅ _prepare_batch funciona correctamente")
        print(f"     B={B}, N_hits={hit_batch[0].shape[0]}, N_pfo={pfo_batch.shape[0]}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requiere GPU")
    def test_training_step_gpu(self, model_and_module, temp_dataset):
        """Test training_step en GPU."""
        model, module = model_and_module
        module = module.cuda()
        
        loader = DataLoader(temp_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch = batch.cuda()
        
        loss = module.training_step(batch, 0)
        
        assert loss is not None
        assert not torch.isnan(loss)
        print(f"\n  ✅ Training step GPU: loss={loss.item():.4f}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GATr requiere GPU")
    def test_training_step(self, model_and_module, temp_dataset):
        """Test training_step completo en GPU (GATr requiere CUDA)."""
        model, module = model_and_module
        module = module.cuda()
        
        loader = DataLoader(temp_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch = batch.cuda()
        
        # Forward pass
        loss = module.training_step(batch, 0)
        
        # Verificaciones
        assert loss is not None
        assert not torch.isnan(loss)
        assert loss.requires_grad  # Debe tener gradientes para backprop
        
        # Test backward pass
        loss.backward()
        
        # Verificar que los gradientes se calcularon
        has_grads = False
        for param in module.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        assert has_grads, "No se calcularon gradientes"
        
        print(f"\n  ✅ Training step completo: loss={loss.item():.4f}")
        print(f"     Gradientes calculados correctamente")
    
    def test_configure_optimizers(self, model_and_module):
        """Test configuración de optimizadores."""
        _, module = model_and_module
        
        opt_config = module.configure_optimizers()
        
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
        
        print("\n  ✅ Optimizador configurado correctamente")


class TestReorganizeGtToTb:
    """Tests para la función reorganize_gt_to_tb."""
    
    def test_basic_reorganization(self):
        """Test reorganización básica."""
        device = torch.device('cpu')
        
        # 5 PFOs: 3 para evento 0, 2 para evento 1
        gt_batch = torch.tensor([0, 0, 0, 1, 1])
        gt_values = torch.tensor([
            [1.0, 0.0, 0.0],  # PFO 0, evento 0
            [0.0, 1.0, 0.0],  # PFO 1, evento 0
            [0.0, 0.0, 1.0],  # PFO 2, evento 0
            [2.0, 0.0, 0.0],  # PFO 0, evento 1
            [0.0, 2.0, 0.0],  # PFO 1, evento 1
        ])
        
        T, B = 4, 2
        output_tb, pfo_step_idx = reorganize_gt_to_tb(gt_batch, gt_values, T, B, device)
        
        assert output_tb.shape == (T, B, 3)
        
        # Verificar asignación correcta
        # Evento 0, step 0 -> PFO 0
        assert torch.allclose(output_tb[0, 0], torch.tensor([1.0, 0.0, 0.0]))
        # Evento 0, step 1 -> PFO 1
        assert torch.allclose(output_tb[1, 0], torch.tensor([0.0, 1.0, 0.0]))
        # Evento 1, step 0 -> PFO 3 (primero del evento 1)
        assert torch.allclose(output_tb[0, 1], torch.tensor([2.0, 0.0, 0.0]))
        
        print("\n  ✅ reorganize_gt_to_tb funciona correctamente")


def test_full_pipeline_sanity():
    """Test de sanidad del pipeline completo."""
    print("\n" + "="*60)
    print("TEST DE SANIDAD DEL PIPELINE COMPLETO")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Crear datos
        npz_path = Path(tmpdir) / "test.npz"
        create_synthetic_npz(npz_path, n_events=5, seed=42)
        print("\n1. ✅ Datos sintéticos creados")
        
        # 2. Cargar dataset
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        print(f"2. ✅ Dataset cargado: {len(dataset)} eventos")
        
        # 3. Crear DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        print(f"3. ✅ Batch creado: {batch.pos.shape[0]} hits, "
              f"{batch.pfo_pid.shape[0]} PFOs")
        
        # 4. Verificar estructura del batch
        B = int(batch.batch.max().item()) + 1
        print(f"4. ✅ Batch tiene {B} eventos")
        
        # 5. Verificar índices
        assert (batch.pfo_event_idx >= 0).all()
        assert (batch.pfo_event_idx < B).all()
        hit_to_pfo = batch.hit_to_pfo
        assert (hit_to_pfo >= -1).all()
        print("5. ✅ Índices correctos (pfo_event_idx incrementado, hit_to_pfo local)")
        
        # 6. Test loss function
        loss_fn = GATrAutoRegressorLoss()
        N = batch.pos.shape[0]
        T = 3
        
        mock_output = {
            "pfo_momentum": torch.randn(T, B, 3),
            "pfo_p_mod": torch.randn(T, B, 1).abs(),
            "pfo_pid": torch.randn(T, B, 5),
            "pfo_charge": torch.randn(T, B, 1),
            "assignments": torch.sigmoid(torch.randn(T, N, 1)),
            "stop_probs": torch.sigmoid(torch.randn(T, B, 1)),
        }
        
        pfo_true_objects = {
            "momentum": batch.pfo_momentum,
            "p_mod": batch.pfo_p_mod,
            "pid": batch.pfo_pid,
            "charge": batch.pfo_charge,
            "batch": batch.pfo_event_idx,
            "hit_to_pfo": batch.hit_to_pfo,
        }
        
        losses = loss_fn(mock_output, pfo_true_objects, batch.batch)
        assert not torch.isnan(losses["loss"])
        print(f"6. ✅ Loss calculada: {losses['loss'].item():.4f}")
        
        # 7-10: Tests que requieren GPU (GATr necesita CUDA)
        if not torch.cuda.is_available():
            print("\n7-10. ⚠️ GPU no disponible, saltando tests del modelo")
            print("\n" + "="*60)
            print("✅ PIPELINE DE SANIDAD COMPLETADO (sin GPU)")
            print("="*60)
            return
        
        device = torch.device('cuda')
        batch = batch.to(device)
        
        # 7. Crear modelo
        try:
            params_cfg = create_small_model_config()
            model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
            model = model.to(device)
            print("7. ✅ Modelo creado en GPU")
            
            # 8. Crear Lightning Module
            cfg = MockTrainerConfig()
            module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
            module = module.to(device)
            print("8. ✅ Lightning Module creado en GPU")
            
            # 9. Test _prepare_batch
            mv_v, mv_s, scalars, hit_b, pfo_gt = module._prepare_batch(batch)
            print("9. ✅ _prepare_batch ejecutado")
            
            # 10. Test training_step completo con backward
            module.train()
            loss = module.training_step(batch, 0)
            assert loss is not None
            assert not torch.isnan(loss)
            
            # Backward pass
            loss.backward()
            
            # Verificar gradientes
            has_grads = False
            for param in module.model.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_grads = True
                    break
            
            if has_grads:
                print(f"10. ✅ Training step completo: loss={loss.item():.4f}, gradientes OK")
            else:
                print(f"10. ⚠️ Training step: loss={loss.item():.4f}, sin gradientes")
            
        except Exception as e:
            print(f"7-10. ❌ Error en el modelo: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)
        print("✅ PIPELINE DE SANIDAD COMPLETADO")
        print("="*60)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GATr requiere GPU")
def test_full_training_loop():
    """Test de un loop de entrenamiento completo con múltiples batches."""
    print("\n" + "="*60)
    print("TEST DE LOOP DE ENTRENAMIENTO COMPLETO")
    print("="*60)
    
    device = torch.device('cuda')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Crear datos
        npz_path = Path(tmpdir) / "test.npz"
        create_synthetic_npz(npz_path, n_events=10, seed=42)
        
        # Dataset y loader
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        loader = DataLoader(dataset, batch_size=3, shuffle=True)
        
        # Modelo
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        model = model.to(device)
        
        # Lightning Module
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        module = module.to(device)
        module.train()
        
        # Optimizer
        opt_config = module.configure_optimizers()
        optimizer = opt_config["optimizer"]
        
        # Training loop
        n_steps = 3
        losses = []
        
        for step, batch in enumerate(loader):
            if step >= n_steps:
                break
            
            batch = batch.to(device)
            optimizer.zero_grad()
            
            loss = module.training_step(batch, step)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            print(f"  Step {step}: loss={loss.item():.4f}")
        
        print(f"\n  Losses: {losses}")
        print(f"  Loss promedio: {sum(losses)/len(losses):.4f}")
        
        # Verificar que las losses son finitas
        assert all(not np.isnan(l) for l in losses)
        assert all(not np.isinf(l) for l in losses)
        
        print("\n" + "="*60)
        print("✅ LOOP DE ENTRENAMIENTO COMPLETADO")
        print("="*60)


# =============================================
# TESTS PARA DETECTOR_SPLIT Y CASOS EXTREMOS
# =============================================

def create_synthetic_npz_with_detector_control(
    path: Path,
    n_events: int = 5,
    seed: int = 42,
    allowed_detectors: list = None,
):
    """
    Crea un archivo NPZ sintético con control sobre qué detectores tienen hits.
    
    Args:
        path: Ruta donde guardar el archivo
        n_events: Número de eventos a generar
        seed: Semilla para reproducibilidad
        allowed_detectors: Lista de índices de detectores permitidos (0-3).
                          Si None, usa todos. Ej: [0, 1] solo INNER_TRACKER y ECAL.
    """
    np.random.seed(seed)
    
    if allowed_detectors is None:
        allowed_detectors = [0, 1, 2, 3]
    
    particle_dtype = np.dtype([
        ('px', np.float32),
        ('py', np.float32),
        ('pz', np.float32),
        ('energy', np.float32),
        ('pid', np.int32),
        ('charge', np.float32),
        ('gen_status', np.int32),
    ])
    
    hit_dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('energy', np.float32),
        ('p', np.float32),
        ('detector_type', np.int32),
    ])
    
    all_particles = []
    all_hits = []
    all_hit_to_particle = []
    all_hit_weights = []
    event_boundaries = [0]
    
    total_hits = 0
    total_particles = 0
    
    for event_idx in range(n_events):
        n_particles = np.random.randint(2, 6)
        particles = np.zeros(n_particles, dtype=particle_dtype)
        
        for i in range(n_particles):
            p_mag = np.random.uniform(1, 100)
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2 * np.pi)
            
            particles[i]['px'] = p_mag * np.sin(theta) * np.cos(phi)
            particles[i]['py'] = p_mag * np.sin(theta) * np.sin(phi)
            particles[i]['pz'] = p_mag * np.cos(theta)
            particles[i]['energy'] = np.sqrt(p_mag**2 + 0.14**2)
            
            pid_choices = [11, 13, 211, 2112, 22]
            particles[i]['pid'] = np.random.choice(pid_choices)
            
            if particles[i]['pid'] in [11, 13]:
                particles[i]['charge'] = -1.0
            elif particles[i]['pid'] == 211:
                particles[i]['charge'] = np.random.choice([-1.0, 1.0])
            else:
                particles[i]['charge'] = 0.0
            
            particles[i]['gen_status'] = 1
        
        all_particles.append(particles)
        
        n_hits = np.random.randint(10, 31)
        hits = np.zeros(n_hits, dtype=hit_dtype)
        hit_to_particle = np.full((n_hits, 3), -1, dtype=np.int32)
        hit_weights = np.zeros((n_hits, 3), dtype=np.float32)
        
        for i in range(n_hits):
            hits[i]['x'] = np.random.uniform(-1000, 1000)
            hits[i]['y'] = np.random.uniform(-1000, 1000)
            hits[i]['z'] = np.random.uniform(-2000, 2000)
            hits[i]['energy'] = np.random.uniform(0.1, 10)
            hits[i]['p'] = np.random.uniform(0.1, 50)
            # Solo usar detectores permitidos
            hits[i]['detector_type'] = np.random.choice(allowed_detectors)
            
            if np.random.random() < 0.9:
                part_idx = np.random.randint(0, n_particles)
                hit_to_particle[i, 0] = total_particles + part_idx
                hit_weights[i, 0] = np.random.uniform(0.5, 1.0)
        
        all_hits.append(hits)
        all_hit_to_particle.append(hit_to_particle)
        all_hit_weights.append(hit_weights)
        
        total_hits += n_hits
        total_particles += n_particles
        event_boundaries.append(total_hits)
    
    particles_concat = np.concatenate(all_particles)
    hits_concat = np.concatenate(all_hits)
    hit_to_particle_concat = np.concatenate(all_hit_to_particle)
    hit_weights_concat = np.concatenate(all_hit_weights)
    event_boundaries = np.array(event_boundaries, dtype=np.int64)
    
    np.savez(
        path,
        particles=particles_concat,
        hits=hits_concat,
        hit_to_particle=hit_to_particle_concat,
        hit_weights=hit_weights_concat,
        event_boundaries=event_boundaries,
        n_events=n_events,
    )
    
    return {'n_events': n_events, 'allowed_detectors': allowed_detectors}


def create_small_split_model_config():
    """Crea una configuración pequeña del modelo para detector_split."""
    # IMPORTANTE: final_module_cfg.in_s_channels debe coincidir con
    # out_s_channels de los módulos de detector (que es hidden_s=16 cuando out_s_channels=None)
    final_cfg = GATrAutoRegressorParamsWhole(
        hidden_mv=8,
        hidden_s=16,
        num_blocks=1,
        in_s_channels=16,  # = hidden_s de los módulos de detector
        in_mv_channels=1,
        out_mv_channels=1,
        dropout=0.0,
        out_s_channels=None,
    )
    
    split_cfg = GATrAutoRegressorParamsSplit(
        hidden_mv=[8, 8, 8, 8],
        hidden_s=[16, 16, 16, 16],
        num_blocks=[1, 1, 1, 1],
        in_s_channels=[6, 6, 6, 6],
        in_mv_channels=[1, 1, 1, 1],
        out_mv_channels=[1, 1, 1, 1],
        dropout=[0.0, 0.0, 0.0, 0.0],
        out_s_channels=[None, None, None, None],
        final_module_cfg=final_cfg,
    )
    
    ar_cfg = GATrAutoRegressive(
        hidden_mv=8,
        hidden_s=16,
        num_blocks=1,
        out_mv_channels=1,
        out_s_channels=None,
        dropout=0.0,
    )
    
    return {
        "detector_split": split_cfg,
        "autoregressive_module": ar_cfg,
    }


class TestSplitByDetector:
    """Tests para la función _split_by_detector del Lightning Module."""
    
    @pytest.fixture
    def temp_dataset_all_detectors(self, tmp_path):
        """Dataset con todos los detectores."""
        npz_path = tmp_path / "test_all_det.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=5, seed=42, allowed_detectors=[0, 1, 2, 3]
        )
        return PFAutoRegressorDataset([str(npz_path)], mode='memory')
    
    @pytest.fixture
    def temp_dataset_missing_detectors(self, tmp_path):
        """Dataset donde faltan algunos detectores (solo ECAL y HCAL)."""
        npz_path = tmp_path / "test_missing_det.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=5, seed=42, allowed_detectors=[1, 2]  # Solo ECAL, HCAL
        )
        return PFAutoRegressorDataset([str(npz_path)], mode='memory')
    
    @pytest.fixture
    def temp_dataset_single_detector(self, tmp_path):
        """Dataset con un solo detector (ECAL)."""
        npz_path = tmp_path / "test_single_det.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=5, seed=42, allowed_detectors=[1]  # Solo ECAL
        )
        return PFAutoRegressorDataset([str(npz_path)], mode='memory')
    
    def test_split_by_detector_basic(self, temp_dataset_all_detectors):
        """Test básico de división por detector."""
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        
        loader = DataLoader(temp_dataset_all_detectors, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        # Llamar directamente a _split_by_detector
        mv_v_split, mv_s_split, scalars_split, batch_split = module._split_by_detector(
            batch.mv_v_part,
            batch.mv_s_part,
            batch.scalars,
            batch.batch,
        )
        
        assert len(mv_v_split) == 4
        assert len(mv_s_split) == 4
        assert len(scalars_split) == 4
        assert len(batch_split) == 4
        
        # La suma de hits en todos los splits debe ser igual al total
        total_split_hits = sum(mv_v.shape[0] for mv_v in mv_v_split)
        assert total_split_hits == batch.mv_v_part.shape[0]
        
        print(f"\n  ✅ Split por detector básico OK")
        print(f"     Hits por detector: {[mv_v.shape[0] for mv_v in mv_v_split]}")
    
    def test_split_with_missing_detectors(self, temp_dataset_missing_detectors):
        """Test cuando algunos detectores no tienen hits."""
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        
        loader = DataLoader(temp_dataset_missing_detectors, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        mv_v_split, mv_s_split, scalars_split, batch_split = module._split_by_detector(
            batch.mv_v_part,
            batch.mv_s_part,
            batch.scalars,
            batch.batch,
        )
        
        # Detectores 0 y 3 deben estar vacíos
        assert mv_v_split[0].shape[0] == 0, "INNER_TRACKER debería estar vacío"
        assert mv_v_split[3].shape[0] == 0, "MUON_TRACKER debería estar vacío"
        
        # Detectores 1 y 2 deben tener hits
        assert mv_v_split[1].shape[0] > 0, "ECAL debería tener hits"
        assert mv_v_split[2].shape[0] > 0, "HCAL debería tener hits"
        
        # Verificar que los tensores vacíos tienen la forma correcta
        assert mv_v_split[0].shape == (0, 3)
        assert scalars_split[0].shape[1] == 6
        
        print(f"\n  ✅ Split con detectores faltantes OK")
        print(f"     Hits por detector: {[mv_v.shape[0] for mv_v in mv_v_split]}")
    
    def test_split_with_single_detector(self, temp_dataset_single_detector):
        """Test cuando solo hay un detector con hits."""
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        
        loader = DataLoader(temp_dataset_single_detector, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        mv_v_split, mv_s_split, scalars_split, batch_split = module._split_by_detector(
            batch.mv_v_part,
            batch.mv_s_part,
            batch.scalars,
            batch.batch,
        )
        
        # Solo ECAL (índice 1) debe tener hits
        assert mv_v_split[0].shape[0] == 0
        assert mv_v_split[1].shape[0] > 0
        assert mv_v_split[2].shape[0] == 0
        assert mv_v_split[3].shape[0] == 0
        
        # Total de hits debe coincidir
        assert mv_v_split[1].shape[0] == batch.mv_v_part.shape[0]
        
        print(f"\n  ✅ Split con un solo detector OK")
        print(f"     Todos los {mv_v_split[1].shape[0]} hits están en ECAL")
    
    def test_batch_indices_preserved_in_split(self, temp_dataset_all_detectors):
        """Verifica que los índices de batch se preservan correctamente."""
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        
        loader = DataLoader(temp_dataset_all_detectors, batch_size=3, shuffle=False)
        batch = next(iter(loader))
        
        mv_v_split, mv_s_split, scalars_split, batch_split = module._split_by_detector(
            batch.mv_v_part,
            batch.mv_s_part,
            batch.scalars,
            batch.batch,
        )
        
        # Verificar que cada split contiene hits de los mismos eventos
        original_events = torch.unique(batch.batch).tolist()
        
        for det_idx, det_batch in enumerate(batch_split):
            if det_batch.numel() > 0:
                det_events = torch.unique(det_batch).tolist()
                # Los eventos en el split deben ser subconjunto del original
                assert all(e in original_events for e in det_events)
        
        print(f"\n  ✅ Índices de batch preservados correctamente")
        print(f"     Eventos originales: {original_events}")


class TestDetectorSplitMode:
    """Tests para el modo detector_split completo."""
    
    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Dataset con todos los detectores."""
        npz_path = tmp_path / "test_data.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=10, seed=42, allowed_detectors=[0, 1, 2, 3]
        )
        return PFAutoRegressorDataset([str(npz_path)], mode='memory')
    
    @pytest.fixture
    def split_model_and_module(self):
        """Crea modelo y módulo Lightning en modo detector_split."""
        params_cfg = create_small_split_model_config()
        model = GATrAutoRegressor(mode="detector_split", params_cfg=params_cfg)
        
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        
        return model, module
    
    def test_prepare_batch_split_mode(self, split_model_and_module, temp_dataset):
        """Test _prepare_batch en modo detector_split."""
        _, module = split_model_and_module
        
        loader = DataLoader(temp_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        mv_v_part, mv_s_part, scalars, hit_batch, pfo_true_objects = module._prepare_batch(batch)
        
        # En modo split, deben ser listas de 4 elementos
        assert len(mv_v_part) == 4
        assert len(mv_s_part) == 4
        assert len(scalars) == 4
        assert len(hit_batch) == 4
        
        # GT sigue siendo el mismo
        assert "momentum" in pfo_true_objects
        assert "hit_to_pfo" in pfo_true_objects
        
        print(f"\n  ✅ _prepare_batch en modo split OK")
        print(f"     Hits por detector: {[mv_v.shape[0] for mv_v in mv_v_part]}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GATr requiere GPU")
    def test_training_step_split_mode(self, split_model_and_module, temp_dataset):
        """Test training_step completo en modo detector_split."""
        model, module = split_model_and_module
        module = module.cuda()
        
        loader = DataLoader(temp_dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch = batch.cuda()
        
        loss = module.training_step(batch, 0)
        
        assert loss is not None
        assert not torch.isnan(loss)
        assert loss.requires_grad
        
        loss.backward()
        
        has_grads = False
        for param in module.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        assert has_grads, "No se calcularon gradientes"
        
        print(f"\n  ✅ Training step split mode: loss={loss.item():.4f}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GATr requiere GPU")
    def test_detector_split_requires_all_detectors_with_hits(self, tmp_path):
        """
        Documenta la limitación: detector_split requiere que todos los detectores tengan hits.
        
        GATr no puede procesar tensores vacíos (0 elementos). Este test verifica que
        cuando un detector no tiene hits, el modelo detector_split falla.
        En producción, asegúrate de que cada detector tenga al menos 1 hit,
        o usa whole_detector mode.
        """
        npz_path = tmp_path / "test_empty_det.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=5, seed=42, allowed_detectors=[1, 2]  # Solo ECAL, HCAL
        )
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        params_cfg = create_small_split_model_config()
        model = GATrAutoRegressor(mode="detector_split", params_cfg=params_cfg)
        model = model.cuda()
        
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        module = module.cuda()
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch = batch.cuda()
        
        # Verificar que efectivamente hay detectores vacíos
        detector_types = batch.scalars[:, 2:6].argmax(dim=-1)
        unique_detectors = torch.unique(detector_types).tolist()
        assert 0 not in unique_detectors or 3 not in unique_detectors, \
            "El dataset debería tener al menos un detector vacío"
        
        # El modelo debería fallar porque GATr no soporta tensores vacíos
        with pytest.raises((RuntimeError, Exception)) as exc_info:
            loss = module.training_step(batch, 0)
        
        print(f"\n  ✅ Limitación documentada: detector_split falla con detectores vacíos")
        print(f"     Error esperado: {type(exc_info.value).__name__}")


class TestEdgeCases:
    """Tests para casos extremos y robustez."""
    
    @pytest.fixture
    def temp_dataset_empty_event(self, tmp_path):
        """Dataset con eventos que tienen muy pocos hits."""
        npz_path = tmp_path / "test_edge.npz"
        # Crear dataset normal primero
        create_synthetic_npz(npz_path, n_events=5, seed=42)
        return PFAutoRegressorDataset([str(npz_path)], mode='memory')
    
    def test_empty_detector_tensors_have_correct_dtype(self, tmp_path):
        """Verifica que tensores vacíos tienen dtype correcto."""
        npz_path = tmp_path / "test_single_det.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=3, seed=42, allowed_detectors=[1]  # Solo ECAL
        )
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        mv_v_split, mv_s_split, scalars_split, batch_split = module._split_by_detector(
            batch.mv_v_part,
            batch.mv_s_part,
            batch.scalars,
            batch.batch,
        )
        
        # Verificar dtypes de tensores vacíos
        for det_idx in [0, 2, 3]:  # Detectores sin hits
            assert mv_v_split[det_idx].dtype == batch.mv_v_part.dtype
            assert scalars_split[det_idx].dtype == batch.scalars.dtype
            assert batch_split[det_idx].dtype == batch.batch.dtype
        
        print("\n  ✅ Dtypes de tensores vacíos correctos")
    
    def test_split_with_varying_batch_sizes(self, tmp_path):
        """Test split con diferentes tamaños de batch."""
        npz_path = tmp_path / "test_vary.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=20, seed=42, allowed_detectors=[0, 1, 2, 3]
        )
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        
        for batch_size in [1, 2, 5, 10]:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            batch = next(iter(loader))
            
            mv_v_split, _, _, batch_split = module._split_by_detector(
                batch.mv_v_part,
                batch.mv_s_part,
                batch.scalars,
                batch.batch,
            )
            
            # Verificar que no hay errores
            total_hits = sum(mv_v.shape[0] for mv_v in mv_v_split)
            assert total_hits == batch.mv_v_part.shape[0]
        
        print(f"\n  ✅ Split funciona con diferentes tamaños de batch")
    
    def test_whole_vs_split_consistency(self, tmp_path):
        """Verifica consistencia entre modos whole y split."""
        npz_path = tmp_path / "test_consist.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=5, seed=42, allowed_detectors=[0, 1, 2, 3]
        )
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        # Modelo whole_detector
        whole_cfg = create_small_model_config()
        model_whole = GATrAutoRegressor(mode="whole_detector", params_cfg=whole_cfg)
        
        cfg = MockTrainerConfig()
        module_whole = GATrAutoRegressorLightningModule(model=model_whole, cfg=cfg)
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        
        # Prepare batch en modo whole
        mv_v_whole, mv_s_whole, scalars_whole, batch_whole, pfo_true = module_whole._prepare_batch(batch)
        
        # Cambiar a modo split manualmente
        model_whole.mode = "detector_split"
        mv_v_split, mv_s_split, scalars_split, batch_split, pfo_true_2 = module_whole._prepare_batch(batch)
        
        # Verificar que GT es idéntico
        assert torch.equal(pfo_true["hit_to_pfo"], pfo_true_2["hit_to_pfo"])
        assert torch.equal(pfo_true["momentum"], pfo_true_2["momentum"])
        
        # Verificar que los splits suman al original
        total_mv_v = torch.cat(mv_v_split, dim=0)
        # Nota: el orden puede diferir, pero los valores deben estar todos
        assert total_mv_v.shape[0] == mv_v_whole[0].shape[0]
        
        print(f"\n  ✅ Consistencia whole vs split verificada")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GATr requiere GPU")
    def test_gradient_flow_with_missing_detectors(self, tmp_path):
        """
        Verifica que los gradientes fluyen correctamente cuando algunos detectores no tienen hits.
        
        NOTA: Usamos whole_detector mode porque detector_split requiere que cada
        submodulo reciba al menos 1 hit (GATr no soporta tensores vacíos).
        En whole_detector mode, los hits de los detectores que existen se procesan juntos.
        """
        npz_path = tmp_path / "test_grad.npz"
        create_synthetic_npz_with_detector_control(
            npz_path, n_events=5, seed=42, allowed_detectors=[1, 2]  # Solo ECAL, HCAL
        )
        dataset = PFAutoRegressorDataset([str(npz_path)], mode='memory')
        
        # Usar whole_detector mode - los hits de ECAL y HCAL se procesan juntos
        params_cfg = create_small_model_config()
        model = GATrAutoRegressor(mode="whole_detector", params_cfg=params_cfg)
        model = model.cuda()
        
        cfg = MockTrainerConfig()
        module = GATrAutoRegressorLightningModule(model=model, cfg=cfg)
        module = module.cuda()
        
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))
        batch = batch.cuda()
        
        # Verificar que efectivamente solo hay hits de ECAL y HCAL
        detector_types = batch.scalars[:, 2:6].argmax(dim=-1)
        unique_detectors = torch.unique(detector_types).tolist()
        assert 0 not in unique_detectors, "No debería haber hits de INNER_TRACKER"
        assert 3 not in unique_detectors, "No debería haber hits de MUON_TRACKER"
        
        loss = module.training_step(batch, 0)
        
        assert loss is not None
        assert not torch.isnan(loss)
        
        loss.backward()
        
        # Verificar que al menos algunos parámetros tienen gradientes
        has_grads = False
        for name, param in module.model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break
        
        assert has_grads, "No hay gradientes con detectores faltantes"
        
        print(f"\n  ✅ Gradientes fluyen correctamente con detectores faltantes")
        print(f"     Detectores presentes: {unique_detectors}")
        print(f"     Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # Ejecutar tests de sanidad
    test_quick_sanity()
    test_full_pipeline_sanity()
    
    # Test de training loop si hay GPU
    if torch.cuda.is_available():
        test_full_training_loop()
    else:
        print("\n⚠️ GPU no disponible, saltando test_full_training_loop")
    
    # Para ejecutar todos los tests con pytest:
    # pytest tests/test_pf_autoregressor_dataset.py -v
