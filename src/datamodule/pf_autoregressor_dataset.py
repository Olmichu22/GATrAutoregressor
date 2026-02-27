"""
Dataset para el modelo GATrAutoRegressor.

Carga datos generados por convert_to_nn_format.py (archivos .npz) y los
prepara para el entrenamiento del modelo autoregresivo de Particle Flow.

Uso:
    # Modo en memoria (más rápido, más uso de RAM)
    dataset = PFAutoRegressorDataset(npz_paths, mode='memory')
    
    # Modo lazy (menor uso de RAM, lee archivos bajo demanda)
    dataset = PFAutoRegressorDataset(npz_paths, mode='lazy')
    
    # Con preprocesamiento personalizado
    preproc = PFPreprocessor(
        normalize_coords=True,
        normalize_energy=True,
        log_energy=True,
        log_momentum=False,
    )
    dataset = PFAutoRegressorDataset(npz_paths, preprocessor=preproc)
"""

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import List, Optional, Dict, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
import argparse
import csv


# =============================================
# Subclase de Data para manejo correcto de batching
# =============================================

class PFAutoRegressorData(Data):
    """
    Subclase de Data que define correctamente cómo manejar
    el batching de atributos relacionados con PFOs.
    
    En PyTorch Geometric, __inc__ define cuánto incrementar
    los índices durante el batching, y __cat_dim__ define
    la dimensión de concatenación.
    
    IMPORTANTE:
    - pfo_event_idx: se incrementa por 1 para indicar evento del batch (0,0,1,1,2,...)
    - hit_to_pfo: NO se incrementa, mantiene índices LOCALES (0,1,2,0,1,0,1,2,...)
      porque el modelo autoregresivo compara con step_idx que es local
    """
    
    def __inc__(self, key: str, value, *args, **kwargs) -> int:
        """
        Define el incremento para índices durante el batching.
        
        Para pfo_event_idx: incrementar por 1 (índice de evento en batch)
        Para hit_to_pfo: NO incrementar (índices locales por evento)
        """
        if key == 'hit_to_pfo':
            # hit_to_pfo contiene índices LOCALES de PFO dentro de cada evento
            # NO debe incrementarse porque el modelo compara con step_idx (0, 1, 2, ...)
            return 0
        elif key == 'pfo_event_idx':
            # pfo_event_idx siempre es 0 para un evento individual
            # Incrementar por 1 para indicar a qué evento del batch pertenece
            return 1
        else:
            return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key: str, value, *args, **kwargs) -> int:
        """
        Define la dimensión de concatenación para cada atributo.
        """
        # Los atributos escalares por evento se concatenan en dim=0
        if key in ['n_hits', 'n_pfo']:
            return None  # Se concatenan en una lista o tensor 1D
        return super().__cat_dim__(key, value, *args, **kwargs)


# Solo nos interesan partículas con gen_status == 1 (estables finales)
VALID_GEN_STATUS = {1}

# Tipos de detectores (para referencia)
DETECTOR_TYPES = {
    'INNER_TRACKER': 0,
    'ECAL': 1,
    'HCAL': 2,
    'MUON_TRACKER': 3
}

# Mapeo de PIDs comunes a índices para clasificación
# El modelo usa 5 clases según GATrAutoRegressor
PID_TO_CLASS = {
    11: 0,    # e-
    -11: 0,   # e+
    13: 1,    # μ-
    -13: 1,   # μ+
    22: 2,    # γ
    211: 3,   # π+ (hadrones cargados)
    -211: 3,  # π-
    321: 3,   # K+
    -321: 3,  # K-
    2212: 3,  # p
    -2212: 3, # p̄
    111: 4,   # π0 (hadrones neutros)
    2112: 4,  # n
    -2112: 4, # n̄
    130: 4,   # K0_L
    310: 4,   # K0_S
}
DEFAULT_CLASS = 4  # Clase por defecto para PIDs no mapeados


def pid_to_onehot(pid: int, num_classes: int = 5) -> np.ndarray:
    """Convierte un PID a vector one-hot."""
    class_idx = PID_TO_CLASS.get(abs(pid), DEFAULT_CLASS)
    onehot = np.zeros(num_classes, dtype=np.float32)
    onehot[class_idx] = 1.0
    return onehot


# =============================================
# Módulo de Preprocesamiento
# =============================================

@dataclass
class PFPreprocessorConfig:
    """
    Configuración para el preprocesador de datos PF.
    
    Todas las opciones de preprocesamiento están aisladas aquí para
    facilitar la experimentación y reproducibilidad.
    """
    # Normalización de coordenadas espaciales (z-score)
    normalize_coords: bool = True
    
    # Normalización de energía y momento (z-score después de log si aplica)
    normalize_energy: bool = True
    normalize_momentum: bool = True
    
    # Transformación logarítmica (siempre se aplica antes de normalización)
    log_energy: bool = True      # SIEMPRE True por defecto para energía
    log_momentum: bool = True    # Logaritmo del módulo del momento
    
    # Epsilon para evitar log(0)
    log_eps: float = 1e-6
    
    # Normalización del momentum de PFOs
    normalize_pfo_momentum: bool = False  # Si normalizar (px, py, pz) de PFOs
    log_pfo_energy: bool = True           # SIEMPRE True - log de |p| de PFOs


class PFPreprocessor:
    """
    Preprocesador para datos de Particle Flow.
    
    Encapsula todas las transformaciones de datos:
    - Transformaciones logarítmicas
    - Normalización z-score
    - Cálculo de estadísticas
    
    El orden de operaciones es:
    1. Log transform (si está habilitado)
    2. Z-score normalization (si está habilitado)
    
    Uso:
        preproc = PFPreprocessor(config)
        preproc.fit(hits_data)  # Calcula estadísticas
        processed = preproc.transform_hits(hits, energy, p_mod)
    """
    
    def __init__(self, config: Optional[PFPreprocessorConfig] = None):
        self.config = config or PFPreprocessorConfig()
        self.stats = None
        self._fitted = False
    
    def fit(self, all_hits: np.ndarray) -> 'PFPreprocessor':
        """
        Calcula estadísticas de normalización a partir de los datos.
        
        Args:
            all_hits: Array estructurado con todos los hits
        
        Returns:
            self para encadenamiento
        """
        cfg = self.config
        
        # Extraer valores
        x = all_hits['x'].astype(np.float32)
        y = all_hits['y'].astype(np.float32)
        z = all_hits['z'].astype(np.float32)
        energy = all_hits['energy'].astype(np.float32)
        p_mod = all_hits['p'].astype(np.float32)
        
        # Aplicar log antes de calcular estadísticas si está configurado
        if cfg.log_energy:
            energy = np.log(energy + cfg.log_eps)
        if cfg.log_momentum:
            p_mod = np.log(p_mod + cfg.log_eps)
        
        self.stats = {
            # Coordenadas
            'x_mean': float(np.mean(x)),
            'x_std': float(np.std(x)) + 1e-6,
            'y_mean': float(np.mean(y)),
            'y_std': float(np.std(y)) + 1e-6,
            'z_mean': float(np.mean(z)),
            'z_std': float(np.std(z)) + 1e-6,
            # Energía (después de log si aplica)
            'energy_mean': float(np.mean(energy)),
            'energy_std': float(np.std(energy)) + 1e-6,
            # Momento (después de log si aplica)
            'p_mean': float(np.mean(p_mod)),
            'p_std': float(np.std(p_mod)) + 1e-6,
        }
        
        self._fitted = True
        logging.info(f"Preprocesador ajustado con config: log_energy={cfg.log_energy}, "
                    f"log_momentum={cfg.log_momentum}")
        logging.info(f"Estadísticas calculadas: {self.stats}")
        
        return self
    
    def fit_from_sample(self, sample_hits: List[np.ndarray]) -> 'PFPreprocessor':
        """
        Calcula estadísticas a partir de una muestra de hits.
        Útil para modo lazy donde no queremos cargar todo en memoria.
        
        Args:
            sample_hits: Lista de arrays de hits
        
        Returns:
            self
        """
        cfg = self.config
        
        x_vals = np.concatenate([h['x'] for h in sample_hits])
        y_vals = np.concatenate([h['y'] for h in sample_hits])
        z_vals = np.concatenate([h['z'] for h in sample_hits])
        e_vals = np.concatenate([h['energy'] for h in sample_hits])
        p_vals = np.concatenate([h['p'] for h in sample_hits])
        
        # Aplicar log antes de calcular estadísticas
        if cfg.log_energy:
            e_vals = np.log(e_vals.astype(np.float32) + cfg.log_eps)
        if cfg.log_momentum:
            p_vals = np.log(p_vals.astype(np.float32) + cfg.log_eps)
        
        self.stats = {
            'x_mean': float(np.mean(x_vals)),
            'x_std': float(np.std(x_vals)) + 1e-6,
            'y_mean': float(np.mean(y_vals)),
            'y_std': float(np.std(y_vals)) + 1e-6,
            'z_mean': float(np.mean(z_vals)),
            'z_std': float(np.std(z_vals)) + 1e-6,
            'energy_mean': float(np.mean(e_vals)),
            'energy_std': float(np.std(e_vals)) + 1e-6,
            'p_mean': float(np.mean(p_vals)),
            'p_std': float(np.std(p_vals)) + 1e-6,
        }
        
        self._fitted = True
        logging.info(f"Preprocesador ajustado (sample) con config: log_energy={cfg.log_energy}")
        
        return self
    
    def transform_coords(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforma coordenadas espaciales.
        
        Args:
            x, y, z: Arrays de coordenadas
        
        Returns:
            x, y, z transformados
        """
        if not self._fitted:
            raise RuntimeError("Preprocesador no ajustado. Llama a fit() primero.")
        
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        z = z.astype(np.float32)
        
        if self.config.normalize_coords and self.stats:
            x = (x - self.stats['x_mean']) / self.stats['x_std']
            y = (y - self.stats['y_mean']) / self.stats['y_std']
            z = (z - self.stats['z_mean']) / self.stats['z_std']
        
        return x, y, z
    
    def transform_hit_energy(self, energy: np.ndarray) -> np.ndarray:
        """
        Transforma energía de hits: log + normalización opcional.
        
        Args:
            energy: Array de energías
        
        Returns:
            Energía transformada
        """
        if not self._fitted:
            raise RuntimeError("Preprocesador no ajustado. Llama a fit() primero.")
        
        energy = energy.astype(np.float32)
        
        # Siempre aplicar log a la energía
        if self.config.log_energy:
            energy = np.log(energy + self.config.log_eps)
        
        # Normalización z-score opcional
        if self.config.normalize_energy and self.stats:
            energy = (energy - self.stats['energy_mean']) / self.stats['energy_std']
        
        return energy
    
    def transform_hit_momentum(self, p_mod: np.ndarray) -> np.ndarray:
        """
        Transforma módulo del momento de hits.
        
        Args:
            p_mod: Array de módulos de momento
        
        Returns:
            Momento transformado
        """
        if not self._fitted:
            raise RuntimeError("Preprocesador no ajustado. Llama a fit() primero.")
        
        p_mod = p_mod.astype(np.float32)
        
        if self.config.log_momentum:
            p_mod = np.log(p_mod + self.config.log_eps)
        
        if self.config.normalize_momentum and self.stats:
            p_mod = (p_mod - self.stats['p_mean']) / self.stats['p_std']
        
        return p_mod
    
    def transform_pfo_momentum(
        self,
        px: np.ndarray,
        py: np.ndarray,
        pz: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Transforma momentum de PFOs.
        
        Retorna el momentum original (px, py, pz) y opcionalmente el log del módulo.
        
        Args:
            px, py, pz: Componentes del momento
        
        Returns:
            px, py, pz, log_p_mod (o p_mod sin log si no está habilitado)
        """
        px = px.astype(np.float32)
        py = py.astype(np.float32)
        pz = pz.astype(np.float32)
        
        # Calcular módulo
        p_mod = np.sqrt(px**2 + py**2 + pz**2)
        
        # Aplicar log al módulo del momento de PFOs
        if self.config.log_pfo_energy:
            log_p_mod = np.log(p_mod + self.config.log_eps)
        else:
            log_p_mod = p_mod
        
        # Normalizar componentes si está configurado
        if self.config.normalize_pfo_momentum:
            # Normalizar a vector unitario
            norm = p_mod + 1e-8
            px = px / norm
            py = py / norm
            pz = pz / norm
        
        return px, py, pz, log_p_mod
    
    def get_state(self) -> Dict:
        """Retorna el estado del preprocesador para serialización."""
        return {
            'config': {
                'normalize_coords': self.config.normalize_coords,
                'normalize_energy': self.config.normalize_energy,
                'normalize_momentum': self.config.normalize_momentum,
                'log_energy': self.config.log_energy,
                'log_momentum': self.config.log_momentum,
                'log_eps': self.config.log_eps,
                'normalize_pfo_momentum': self.config.normalize_pfo_momentum,
                'log_pfo_energy': self.config.log_pfo_energy,
            },
            'stats': self.stats,
            'fitted': self._fitted,
        }
    
    def load_state(self, state: Dict) -> 'PFPreprocessor':
        """Carga el estado del preprocesador."""
        self.config = PFPreprocessorConfig(**state['config'])
        self.stats = state['stats']
        self._fitted = state['fitted']
        return self


class PFAutoRegressorDataset(Dataset):
    """
    Dataset para entrenamiento de GATrAutoRegressor.
    
    Carga eventos desde archivos .npz generados por convert_to_nn_format.py
    y devuelve objetos Data de PyTorch Geometric con:
    
    - mv_v_part: posiciones de los hits (x, y, z) → se usarán para embed_point
    - mv_s_part: placeholder para vectores secundarios (inicialmente zeros)
    - scalars: (energy, p_mod, detector_type_onehot) por hit
    - batch: índice de evento por hit (para batching de PyG)
    
    Para el ground truth (PFOs):
    - pfo_pid: one-hot encoding del PID
    - pfo_momentum: (px, py, pz) 
    - pfo_p_mod: log(|p|) del PFO
    - pfo_charge: carga de la partícula
    - pfo_event_idx: índice de evento por PFO
    - hit_to_pfo: índice de PFO (local al evento) por hit
    
    Args:
        npz_paths: Lista de rutas a archivos .npz
        mode: 'memory' para cargar todo en memoria, 'lazy' para leer bajo demanda
        preprocessor: Instancia de PFPreprocessor (si None, se crea uno por defecto)
        max_particles_per_hit: Número máximo de partículas asociadas por hit a considerar
    """
    
    def __init__(
        self,
        npz_paths: Union[str, List[str]],
        mode: str = 'memory',
        preprocessor: Optional[PFPreprocessor] = None,
        max_particles_per_hit: int = 1,  # Usamos la asociación principal
    ):
        super().__init__()
        
        if isinstance(npz_paths, str):
            npz_paths = [npz_paths]
        
        self.npz_paths = [Path(p) for p in npz_paths]
        self.mode = mode
        self.max_particles_per_hit = max_particles_per_hit
        
        # Preprocesador (crear uno por defecto si no se proporciona)
        self.preprocessor = preprocessor or PFPreprocessor()
        
        # Verificar que los archivos existen
        for path in self.npz_paths:
            if not path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {path}")
        
        # Índice global: mapea idx → (file_idx, event_idx_in_file)
        self.global_index = []
        
        if mode == 'memory':
            self._load_all_in_memory()
        else:
            self._build_lazy_index()
    
    def _load_all_in_memory(self):
        """Carga todos los datos en memoria."""
        logging.info(f"Cargando {len(self.npz_paths)} archivos en memoria...")
        
        self.data_cache = []  # Lista de dicts con datos por archivo
        
        for file_idx, path in enumerate(self.npz_paths):
            logging.info(f"  Cargando {path.name}...")
            data = np.load(path, allow_pickle=True)
            
            file_data = {
                'particles': data['particles'],
                'hits': data['hits'],
                'hit_to_particle': data['hit_to_particle'],
                'hit_weights': data['hit_weights'],
                'event_boundaries': data['event_boundaries'],
                'n_events': int(data['n_events']),
            }
            
            self.data_cache.append(file_data)
            
            # Construir índice global
            for event_idx in range(file_data['n_events']):
                self.global_index.append((file_idx, event_idx))
        
        self.N = len(self.global_index)
        
        # Ajustar preprocesador con todos los hits
        if not self.preprocessor._fitted:
            all_hits = np.concatenate([d['hits'] for d in self.data_cache])
            self.preprocessor.fit(all_hits)
        
        logging.info(f"Cargados {self.N} eventos en memoria.")
    
    def _build_lazy_index(self):
        """Construye índice sin cargar datos en memoria."""
        logging.info(f"Construyendo índice para {len(self.npz_paths)} archivos (modo lazy)...")
        
        self.file_sizes = []
        
        for file_idx, path in enumerate(self.npz_paths):
            with np.load(path, allow_pickle=True) as data:
                n_events = int(data['n_events'])
                self.file_sizes.append(n_events)
                
                for event_idx in range(n_events):
                    self.global_index.append((file_idx, event_idx))
        
        self.N = len(self.global_index)
        self._lazy_file_handles = {}  # Cache de archivos abiertos
        
        # Ajustar preprocesador con sample si no está ajustado
        if not self.preprocessor._fitted:
            self._fit_preprocessor_lazy()
        
        logging.info(f"Índice construido: {self.N} eventos.")
    
    def _fit_preprocessor_lazy(self):
        """Ajusta el preprocesador con una muestra de datos en modo lazy."""
        sample_indices = np.random.choice(
            self.N, 
            size=min(100, self.N), 
            replace=False
        )
        
        sample_hits = []
        for idx in sample_indices:
            file_idx, event_idx = self.global_index[idx]
            event_data = self._load_event_lazy(file_idx, event_idx)
            sample_hits.append(event_data['hits'])
        
        self.preprocessor.fit_from_sample(sample_hits)
    
    def _load_event_lazy(self, file_idx: int, event_idx: int) -> Dict:
        """Carga un evento específico en modo lazy."""
        path = self.npz_paths[file_idx]
        
        # Usar handle cacheado si está disponible
        if file_idx not in self._lazy_file_handles:
            self._lazy_file_handles[file_idx] = np.load(path, allow_pickle=True)
        
        data = self._lazy_file_handles[file_idx]
        
        boundaries = data['event_boundaries']
        hit_start = boundaries[event_idx]
        hit_end = boundaries[event_idx + 1]
        
        return {
            'particles': data['particles'],
            'hits': data['hits'][hit_start:hit_end],
            'hit_to_particle': data['hit_to_particle'][hit_start:hit_end],
            'hit_weights': data['hit_weights'][hit_start:hit_end],
            'hit_start': hit_start,
            'hit_end': hit_end,
        }
    
    def _load_event_memory(self, file_idx: int, event_idx: int) -> Dict:
        """Carga un evento específico desde memoria."""
        file_data = self.data_cache[file_idx]
        
        boundaries = file_data['event_boundaries']
        hit_start = boundaries[event_idx]
        hit_end = boundaries[event_idx + 1]
        
        return {
            'particles': file_data['particles'],
            'hits': file_data['hits'][hit_start:hit_end],
            'hit_to_particle': file_data['hit_to_particle'][hit_start:hit_end],
            'hit_weights': file_data['hit_weights'][hit_start:hit_end],
            'hit_start': hit_start,
            'hit_end': hit_end,
        }
    
    def len(self) -> int:
        return self.N
    
    def get(self, idx: int) -> Data:
        """
        Obtiene un evento como objeto Data de PyTorch Geometric.
        
        El Data contiene:
        - pos: (N_hits, 3) posiciones (preprocesadas)
        - mv_v_part: (N_hits, 3) = pos (para embedding geométrico)
        - mv_s_part: (N_hits, 3) zeros placeholder
        - scalars: (N_hits, 6) [log_energy, log_p_mod, det_type_0, det_type_1, det_type_2, det_type_3]
        - pfo_pid: (N_pfo, 5) one-hot PID
        - pfo_momentum: (N_pfo, 3) momentum (px, py, pz)
        - pfo_p_mod: (N_pfo, 1) log(|p|) del PFO
        - pfo_charge: (N_pfo, 1) carga
        - pfo_event_idx: (N_pfo,) siempre 0 (un solo evento)
        - hit_to_pfo: (N_hits,) índice LOCAL de PFO por hit
            * Mismo tamaño que pos, scalars, etc.
            * Valor -1 para hits no asociados a ningún PFO válido
            * Valores 0, 1, 2, ... para hits asociados al PFO correspondiente
        - n_hits: escalar, número de hits
        - n_pfo: escalar, número de PFOs
        """
        file_idx, event_idx = self.global_index[idx]
        
        if self.mode == 'memory':
            event_data = self._load_event_memory(file_idx, event_idx)
        else:
            event_data = self._load_event_lazy(file_idx, event_idx)
        
        hits = event_data['hits']
        particles_all = event_data['particles']
        hit_to_particle = event_data['hit_to_particle']
        hit_weights = event_data['hit_weights']
        
        n_hits = len(hits)
        preproc = self.preprocessor
        
        # =============================================
        # 1. Procesar hits → features de entrada
        # =============================================
        
        # Posiciones (con preprocesamiento)
        x, y, z = preproc.transform_coords(
            hits['x'], hits['y'], hits['z']
        )
        pos = np.stack([x, y, z], axis=1)  # (N_hits, 3)
        
        # Energía con log transform + normalización opcional
        energy = preproc.transform_hit_energy(hits['energy'])
        
        # Momento con log transform + normalización opcional
        p_mod = preproc.transform_hit_momentum(hits['p'])
        
        # Detector type como one-hot (4 tipos)
        det_type = hits['detector_type'].astype(np.int32)
        det_type_onehot = np.zeros((n_hits, 4), dtype=np.float32)
        det_type_onehot[np.arange(n_hits), det_type] = 1.0
        
        # Scalars: [log_energy, log_p_mod, det_type_onehot (4)]
        scalars = np.concatenate([
            energy.reshape(-1, 1),
            p_mod.reshape(-1, 1),
            det_type_onehot
        ], axis=1)  # (N_hits, 6)
        
        # =============================================
        # 2. Procesar partículas → PFOs ground truth
        # =============================================
        
        # Encontrar partículas únicas asociadas a hits con gen_status válido
        unique_particle_indices = set()
        for i in range(n_hits):
            for j in range(self.max_particles_per_hit):
                part_idx = hit_to_particle[i, j]
                if part_idx >= 0:
                    # Verificar gen_status
                    if particles_all[part_idx]['gen_status'] in VALID_GEN_STATUS:
                        unique_particle_indices.add(part_idx)
        
        # Ordenar según el orden típico de Particle Flow:
        # 1. Muones, 2. Electrones, 3. Fotones, 4. Hadrones neutros, 5. Hadrones cargados
        # Mapeo de clase PID → prioridad de ordenamiento
        PF_ORDER = {1: 0, 0: 1, 2: 2, 4: 3, 3: 4}  # muon, e, γ, had_neutral, had_charged
        
        def pfo_sort_key(part_idx):
            pid = int(particles_all[part_idx]['pid'])
            pid_class = PID_TO_CLASS.get(abs(pid), DEFAULT_CLASS)
            pf_priority = PF_ORDER.get(pid_class, 5)
            # Ordenar por prioridad PF, luego por energía descendente
            energy = particles_all[part_idx]['energy']
            return (pf_priority, -energy)
        
        unique_particle_indices = sorted(list(unique_particle_indices), key=pfo_sort_key)
        n_pfo = len(unique_particle_indices)
        
        # Crear mapeo de índice global de partícula → índice local de PFO
        particle_to_pfo = {part_idx: pfo_idx for pfo_idx, part_idx in enumerate(unique_particle_indices)}
        
        # Extraer propiedades de los PFOs
        if n_pfo > 0:
            pfo_pid = np.zeros((n_pfo, 5), dtype=np.float32)
            pfo_momentum = np.zeros((n_pfo, 3), dtype=np.float32)
            pfo_p_mod = np.zeros((n_pfo, 1), dtype=np.float32)
            pfo_charge = np.zeros((n_pfo, 1), dtype=np.float32)
            
            for pfo_idx, part_idx in enumerate(unique_particle_indices):
                part = particles_all[part_idx]
                
                pfo_pid[pfo_idx] = pid_to_onehot(int(part['pid']))
                
                # Transformar momentum de PFO
                px, py, pz, log_p = preproc.transform_pfo_momentum(
                    np.array([part['px']]),
                    np.array([part['py']]),
                    np.array([part['pz']])
                )
                pfo_momentum[pfo_idx] = [px[0], py[0], pz[0]]
                pfo_p_mod[pfo_idx] = log_p[0]
                pfo_charge[pfo_idx] = part['charge']
        else:
            # Evento sin PFOs válidos (raro pero posible)
            pfo_pid = np.zeros((1, 5), dtype=np.float32)
            pfo_momentum = np.zeros((1, 3), dtype=np.float32)
            pfo_p_mod = np.zeros((1, 1), dtype=np.float32)
            pfo_charge = np.zeros((1, 1), dtype=np.float32)
            n_pfo = 1
        
        # =============================================
        # 3. Crear mapeo hit → PFO
        # =============================================
        
        # Para cada hit, encontrar el PFO principal asociado
        hit_to_pfo = np.full(n_hits, -1, dtype=np.int64)
        
        for i in range(n_hits):
            best_pfo = -1
            best_weight = -1.0
            
            for j in range(self.max_particles_per_hit):
                part_idx = hit_to_particle[i, j]
                weight = hit_weights[i, j]
                
                if part_idx >= 0 and weight > best_weight:
                    if particles_all[part_idx]['gen_status'] in VALID_GEN_STATUS:
                        if part_idx in particle_to_pfo:
                            candidate_pfo = particle_to_pfo[part_idx]
                            if weight > best_weight:
                                best_pfo = candidate_pfo
                                best_weight = weight
            
            hit_to_pfo[i] = best_pfo
        
        # =============================================
        # 4. Construir objeto Data con subclase PFAutoRegressorData
        # =============================================
        
        # Usamos PFAutoRegressorData para que PyG maneje correctamente
        # los índices de PFO durante el batching
        data = PFAutoRegressorData(
            # Inputs del modelo
            pos=torch.from_numpy(pos),
            mv_v_part=torch.from_numpy(pos),  # Posiciones para embedding geométrico
            mv_s_part=torch.zeros(n_hits, 1, dtype=torch.float32),  # Placeholder
            scalars=torch.from_numpy(scalars),
            
            # Ground truth PFOs
            pfo_pid=torch.from_numpy(pfo_pid),
            pfo_momentum=torch.from_numpy(pfo_momentum),
            pfo_p_mod=torch.from_numpy(pfo_p_mod),  # log(|p|)
            pfo_charge=torch.from_numpy(pfo_charge),
            pfo_event_idx=torch.zeros(n_pfo, dtype=torch.long),  # Un evento = índice 0
            
            # Mapeo hit → PFO
            hit_to_pfo=torch.from_numpy(hit_to_pfo),
            
            # Metadata
            n_hits=n_hits,
            n_pfo=n_pfo,
        )
        
        return data
    
    def close(self):
        """Cierra los handles de archivos en modo lazy."""
        if hasattr(self, '_lazy_file_handles'):
            for handle in self._lazy_file_handles.values():
                if hasattr(handle, 'close'):
                    handle.close()
            self._lazy_file_handles.clear()
    
    def __del__(self):
        self.close()


def make_pf_splits(
    paths: List[str],
    val_ratio: float = 0.2,
    mode: str = 'memory',
    preprocessor: Optional[PFPreprocessor] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Crea splits de train/val para el dataset PFAutoRegressor.
    
    Args:
        paths: Lista de rutas a archivos .npz
        val_ratio: Fracción de datos para validación
        mode: 'memory' o 'lazy'
        preprocessor: Preprocesador opcional (si None, se crea uno por defecto)
    
    Returns:
        train_dataset, val_dataset
    """
    dataset = PFAutoRegressorDataset(paths, mode=mode, preprocessor=preprocessor)
    N = dataset.len()
    
    indices = np.random.permutation(N)
    val_size = int(N * val_ratio)
    
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    
    return train_ds, val_ds


def make_pf_loaders(
    paths: List[str],
    batch_size: int = 32,
    num_workers: int = 4,
    val_ratio: float = 0.2,
    mode: str = 'memory',
    preprocessor: Optional[PFPreprocessor] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Crea DataLoaders para entrenamiento y validación.
    
    Args:
        paths: Lista de rutas a archivos .npz
        batch_size: Tamaño del batch
        num_workers: Número de workers para carga de datos
        val_ratio: Fracción de datos para validación
        mode: 'memory' o 'lazy'
        preprocessor: Preprocesador opcional
    
    Returns:
        train_loader, val_loader
    """
    train_ds, val_ds = make_pf_splits(paths, val_ratio, mode, preprocessor)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


# =============================================
# Tests
# =============================================

def test_dataset_loading(npz_path: str):
    """Test básico de carga del dataset."""
    print("=" * 60)
    print("TEST: Dataset Loading")
    print("=" * 60)
    
    # Test modo memoria
    print("\n1. Testing memory mode...")
    dataset_mem = PFAutoRegressorDataset([npz_path], mode='memory')
    print(f"   Eventos cargados: {dataset_mem.len()}")
    
    # Test modo lazy - usar el mismo preprocesador ya ajustado
    print("\n2. Testing lazy mode (con preprocesador compartido)...")
    # Reutilizar el preprocesador del modo memory para comparación justa
    shared_preprocessor = dataset_mem.preprocessor
    dataset_lazy = PFAutoRegressorDataset(
        [npz_path], 
        mode='lazy',
        preprocessor=shared_preprocessor  # Compartir preprocesador
    )
    print(f"   Eventos indexados: {dataset_lazy.len()}")
    
    # Verificar que ambos modos dan los mismos datos
    print("\n3. Comparing modes...")
    data_mem = dataset_mem.get(0)
    data_lazy = dataset_lazy.get(0)
    
    assert data_mem.n_hits == data_lazy.n_hits, "n_hits mismatch"
    assert data_mem.n_pfo == data_lazy.n_pfo, "n_pfo mismatch"
    assert torch.allclose(data_mem.pos, data_lazy.pos), "pos mismatch"
    assert torch.allclose(data_mem.scalars, data_lazy.scalars), "scalars mismatch"
    assert torch.allclose(data_mem.pfo_momentum, data_lazy.pfo_momentum), "pfo_momentum mismatch"
    
    print("   ✓ Both modes return identical data")
    
    dataset_lazy.close()
    print("\n✓ Dataset loading test passed!")


def test_data_structure(npz_path: str):
    """Test de la estructura de datos."""
    print("\n" + "=" * 60)
    print("TEST: Data Structure")
    print("=" * 60)
    
    dataset = PFAutoRegressorDataset([npz_path], mode='memory')
    data = dataset.get(0)
    
    print(f"\nEvento 0:")
    print(f"  - n_hits: {data.n_hits}")
    print(f"  - n_pfo: {data.n_pfo}")
    print(f"  - pos shape: {data.pos.shape}")
    print(f"  - mv_v_part shape: {data.mv_v_part.shape}")
    print(f"  - mv_s_part shape: {data.mv_s_part.shape}")
    print(f"  - scalars shape: {data.scalars.shape}")
    print(f"  - pfo_pid shape: {data.pfo_pid.shape}")
    print(f"  - pfo_momentum shape: {data.pfo_momentum.shape}")
    print(f"  - pfo_p_mod shape: {data.pfo_p_mod.shape}")
    print(f"  - pfo_charge shape: {data.pfo_charge.shape}")
    print(f"  - pfo_event_idx shape: {data.pfo_event_idx.shape}")
    print(f"  - hit_to_pfo shape: {data.hit_to_pfo.shape}")
    
    # Verificar shapes
    assert data.pos.shape == (data.n_hits, 3)
    assert data.mv_v_part.shape == (data.n_hits, 3)
    assert data.mv_s_part.shape == (data.n_hits, 1)
    assert data.scalars.shape == (data.n_hits, 6)
    assert data.pfo_pid.shape == (data.n_pfo, 5)
    assert data.pfo_momentum.shape == (data.n_pfo, 3)
    assert data.pfo_p_mod.shape == (data.n_pfo, 1)
    assert data.pfo_charge.shape == (data.n_pfo, 1)
    assert data.hit_to_pfo.shape == (data.n_hits,)
    
    # Verificar que pfo_pid es one-hot
    assert torch.allclose(data.pfo_pid.sum(dim=1), torch.ones(data.n_pfo))
    
    print("\n✓ Data structure test passed!")


def test_dataloader(npz_path: str):
    """Test del DataLoader con batching."""
    print("\n" + "=" * 60)
    print("TEST: DataLoader Batching")
    print("=" * 60)
    
    train_loader, val_loader = make_pf_loaders(
        [npz_path],
        batch_size=4,
        num_workers=0,
        val_ratio=0.2,
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Probar un batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch structure:")
    print(f"  - pos shape: {batch.pos.shape}")
    print(f"  - batch vector shape: {batch.batch.shape}")
    print(f"  - pfo_pid shape: {batch.pfo_pid.shape}")
    print(f"  - pfo_event_idx shape: {batch.pfo_event_idx.shape}")
    print(f"  - hit_to_pfo shape: {batch.hit_to_pfo.shape}")
    
    # Verificar que el batch vector está correctamente construido
    n_events_in_batch = batch.batch.max().item() + 1
    print(f"  - Events in batch: {n_events_in_batch}")
    
    print("\n✓ DataLoader test passed!")


def test_pfo_validity(npz_path: str):
    """Verifica que solo se incluyen PFOs con gen_status válido."""
    print("\n" + "=" * 60)
    print("TEST: PFO Validity (gen_status filter)")
    print("=" * 60)
    
    # Cargar datos raw para comparar
    raw_data = np.load(npz_path, allow_pickle=True)
    particles = raw_data['particles']
    
    # Contar partículas con gen_status válido
    valid_count = sum(1 for p in particles if p['gen_status'] in VALID_GEN_STATUS)
    total_count = len(particles)
    
    print(f"\nRaw data:")
    print(f"  - Total particles: {total_count}")
    print(f"  - Valid gen_status (1): {valid_count}")
    print(f"  - Filtered: {total_count - valid_count}")
    
    # Verificar en dataset
    dataset = PFAutoRegressorDataset([npz_path], mode='memory')
    
    total_pfo = 0
    for i in range(min(10, dataset.len())):
        data = dataset.get(i)
        total_pfo += data.n_pfo
    
    print(f"\nDataset (primeros 10 eventos):")
    print(f"  - Total PFOs extraídos: {total_pfo}")
    
    print("\n✓ PFO validity test passed!")


def test_preprocessor(npz_path: str):
    """Test del preprocesador."""
    print("\n" + "=" * 60)
    print("TEST: Preprocessor")
    print("=" * 60)
    
    # Test con configuración por defecto (log_energy=True)
    print("\n1. Testing default config (log_energy=True)...")
    config_default = PFPreprocessorConfig()
    assert config_default.log_energy == True, "log_energy should be True by default"
    assert config_default.log_pfo_energy == True, "log_pfo_energy should be True by default"
    print("   ✓ Default config is correct")
    
    # Test con configuración personalizada
    print("\n2. Testing custom config...")
    config_custom = PFPreprocessorConfig(
        normalize_coords=False,
        normalize_energy=False,
        log_energy=True,
        log_momentum=False,
    )
    preproc = PFPreprocessor(config_custom)
    
    # Cargar datos y ajustar
    raw_data = np.load(npz_path, allow_pickle=True)
    all_hits = raw_data['hits']
    preproc.fit(all_hits[:1000])  # Usar subset para rapidez
    
    print("   ✓ Preprocessor fitted successfully")
    
    # Verificar transformación de energía
    print("\n3. Testing log transform on energy...")
    test_energy = np.array([0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    transformed = preproc.transform_hit_energy(test_energy)
    
    # Verificar que se aplicó log (valores deben ser mucho menores)
    expected_log = np.log(test_energy + config_custom.log_eps).astype(np.float32)
    assert np.allclose(transformed, expected_log, rtol=1e-5), "Log transform not applied correctly"
    print(f"   Original: {test_energy}")
    print(f"   Transformed (log): {transformed}")
    print("   ✓ Log transform working correctly")
    
    # Test serialización
    print("\n4. Testing state serialization...")
    state = preproc.get_state()
    new_preproc = PFPreprocessor()
    new_preproc.load_state(state)
    assert new_preproc._fitted == True
    assert new_preproc.stats == preproc.stats
    print("   ✓ Serialization working correctly")
    
    # Test dataset con preprocesador personalizado
    print("\n5. Testing dataset with custom preprocessor...")
    dataset = PFAutoRegressorDataset(
        [npz_path], 
        mode='memory',
        preprocessor=PFPreprocessor(PFPreprocessorConfig(
            normalize_coords=True,
            log_energy=True,
            log_momentum=True,
        ))
    )
    data = dataset.get(0)
    
    # Verificar que scalars tiene valores razonables (log-transformed)
    # Energías en log scale deberían estar típicamente entre -10 y 10
    energy_vals = data.scalars[:, 0]
    assert energy_vals.min() > -20, f"Energy min too low: {energy_vals.min()}"
    assert energy_vals.max() < 20, f"Energy max too high: {energy_vals.max()}"
    print(f"   Energy range (log-transformed): [{energy_vals.min():.2f}, {energy_vals.max():.2f}]")
    
    # Verificar pfo_p_mod (debe estar en log scale)
    assert data.pfo_p_mod is not None, "pfo_p_mod should exist"
    print(f"   PFO log(|p|) range: [{data.pfo_p_mod.min():.2f}, {data.pfo_p_mod.max():.2f}]")
    print("   ✓ Dataset with custom preprocessor working correctly")
    
    print("\n✓ Preprocessor test passed!")


def _parse_event_selection(event_spec: str, max_events: int) -> List[int]:
    """
    Parsea especificación de eventos tipo "0,3,5-8".
    
    Args:
        event_spec: Cadena con índices/rangos separados por coma.
        max_events: Número total de eventos del dataset.
    
    Returns:
        Lista ordenada de índices de eventos válidos (sin duplicados).
    """
    if not event_spec or not event_spec.strip():
        raise ValueError("La selección de eventos está vacía.")
    
    selected = set()
    chunks = [c.strip() for c in event_spec.split(",") if c.strip()]
    
    for chunk in chunks:
        if "-" in chunk:
            bounds = [b.strip() for b in chunk.split("-", maxsplit=1)]
            if len(bounds) != 2:
                raise ValueError(f"Rango inválido: '{chunk}'")
            start, end = int(bounds[0]), int(bounds[1])
            if start > end:
                raise ValueError(f"Rango inválido (start > end): '{chunk}'")
            for evt_idx in range(start, end + 1):
                if evt_idx < 0 or evt_idx >= max_events:
                    raise ValueError(
                        f"Evento fuera de rango: {evt_idx}. Dataset tiene [0, {max_events - 1}]"
                    )
                selected.add(evt_idx)
        else:
            evt_idx = int(chunk)
            if evt_idx < 0 or evt_idx >= max_events:
                raise ValueError(
                    f"Evento fuera de rango: {evt_idx}. Dataset tiene [0, {max_events - 1}]"
                )
            selected.add(evt_idx)
    
    return sorted(selected)


def _load_raw_event(dataset: PFAutoRegressorDataset, global_event_idx: int) -> Dict:
    """Carga datos raw de un evento usando la misma indexación global del dataset."""
    file_idx, event_idx = dataset.global_index[global_event_idx]
    if dataset.mode == "memory":
        return dataset._load_event_memory(file_idx, event_idx)
    return dataset._load_event_lazy(file_idx, event_idx)


def _collect_event_diagnostics(dataset: PFAutoRegressorDataset, event_idx: int) -> Dict:
    """
    Extrae información detallada del evento:
    - propiedades de PFOs
    - asociaciones hit→PFO y pesos
    - Data procesado para validar consistencia
    """
    event_raw = _load_raw_event(dataset, event_idx)
    data = dataset.get(event_idx)
    
    hits = event_raw["hits"]
    particles_all = event_raw["particles"]
    hit_to_particle = event_raw["hit_to_particle"]
    hit_weights = event_raw["hit_weights"]
    n_hits = len(hits)
    
    unique_particle_indices = set()
    for i in range(n_hits):
        for j in range(dataset.max_particles_per_hit):
            part_idx = int(hit_to_particle[i, j])
            if part_idx >= 0 and particles_all[part_idx]["gen_status"] in VALID_GEN_STATUS:
                unique_particle_indices.add(part_idx)
    
    pf_order = {1: 0, 0: 1, 2: 2, 4: 3, 3: 4}
    
    def pfo_sort_key(part_idx: int):
        pid = int(particles_all[part_idx]["pid"])
        pid_class = PID_TO_CLASS.get(abs(pid), DEFAULT_CLASS)
        pf_priority = pf_order.get(pid_class, 5)
        energy = float(particles_all[part_idx]["energy"])
        return (pf_priority, -energy)
    
    unique_particle_indices = sorted(list(unique_particle_indices), key=pfo_sort_key)
    particle_to_pfo = {part_idx: pfo_idx for pfo_idx, part_idx in enumerate(unique_particle_indices)}
    
    hit_best_pfo = np.full(n_hits, -1, dtype=np.int64)
    hit_best_particle = np.full(n_hits, -1, dtype=np.int64)
    hit_best_weight = np.zeros(n_hits, dtype=np.float32)
    
    for i in range(n_hits):
        best_weight = -1.0
        best_pfo = -1
        best_particle = -1
        for j in range(dataset.max_particles_per_hit):
            part_idx = int(hit_to_particle[i, j])
            weight = float(hit_weights[i, j])
            if part_idx < 0:
                continue
            if particles_all[part_idx]["gen_status"] not in VALID_GEN_STATUS:
                continue
            if part_idx not in particle_to_pfo:
                continue
            if weight > best_weight:
                best_weight = weight
                best_particle = part_idx
                best_pfo = particle_to_pfo[part_idx]
        hit_best_pfo[i] = best_pfo
        hit_best_particle[i] = best_particle
        hit_best_weight[i] = max(best_weight, 0.0)
    
    pfo_rows = []
    for pfo_idx, part_idx in enumerate(unique_particle_indices):
        part = particles_all[part_idx]
        pid = int(part["pid"])
        px = float(part["px"])
        py = float(part["py"])
        pz = float(part["pz"])
        p_mod = float(np.sqrt(px**2 + py**2 + pz**2))
        log_p_mod = float(np.log(p_mod + dataset.preprocessor.config.log_eps))
        mask = hit_best_pfo == pfo_idx
        pfo_rows.append({
            "event_idx": event_idx,
            "pfo_idx": pfo_idx,
            "particle_idx": int(part_idx),
            "pid": pid,
            "pid_class": int(PID_TO_CLASS.get(abs(pid), DEFAULT_CLASS)),
            "charge": float(part["charge"]),
            "gen_status": int(part["gen_status"]),
            "px": px,
            "py": py,
            "pz": pz,
            "p_mod": p_mod,
            "log_p_mod": log_p_mod,
            "n_associated_hits": int(np.sum(mask)),
            "sum_assoc_weight": float(np.sum(hit_best_weight[mask])) if np.any(mask) else 0.0,
            "max_assoc_weight": float(np.max(hit_best_weight[mask])) if np.any(mask) else 0.0,
        })
    
    assoc_rows = []
    for i in range(n_hits):
        assoc_rows.append({
            "event_idx": event_idx,
            "hit_idx": i,
            "x": float(hits["x"][i]),
            "y": float(hits["y"][i]),
            "z": float(hits["z"][i]),
            "energy": float(hits["energy"][i]),
            "p": float(hits["p"][i]),
            "detector_type": int(hits["detector_type"][i]),
            "pfo_idx": int(hit_best_pfo[i]),
            "particle_idx": int(hit_best_particle[i]),
            "assoc_weight": float(hit_best_weight[i]),
        })
    
    return {
        "data": data,
        "hits": hits,
        "pfo_rows": pfo_rows,
        "assoc_rows": assoc_rows,
        "hit_best_pfo": hit_best_pfo,
        "hit_best_weight": hit_best_weight,
    }


def _write_csv_rows(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    """Escribe filas en CSV con encabezado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_pfo_projections(
    out_path: Path,
    event_idx: int,
    pfo_row: Dict,
    hits: np.ndarray,
    hit_best_pfo: np.ndarray,
    hit_best_weight: np.ndarray,
) -> None:
    """Genera un plot con dos subplots (XY e YZ) para un PFO."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib no está disponible. Instálalo para generar plots."
        ) from exc
    
    pfo_idx = int(pfo_row["pfo_idx"])
    assoc_mask = hit_best_pfo == pfo_idx
    
    x = hits["x"].astype(np.float32)
    y = hits["y"].astype(np.float32)
    z = hits["z"].astype(np.float32)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    
    # XY
    axes[0].scatter(x, y, s=8, c="lightgray", alpha=0.35, label="All hits")
    if np.any(assoc_mask):
        sc_xy = axes[0].scatter(
            x[assoc_mask], y[assoc_mask],
            s=24, c=hit_best_weight[assoc_mask], cmap="viridis", vmin=0.0, vmax=1.0,
            label="Associated hits"
        )
        fig.colorbar(sc_xy, ax=axes[0], fraction=0.046, pad=0.04, label="Assoc weight")
    axes[0].set_title(f"Event {event_idx} | PFO {pfo_idx} | XY")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].grid(alpha=0.2)
    axes[0].legend(loc="best")
    
    # YZ
    axes[1].scatter(y, z, s=8, c="lightgray", alpha=0.35, label="All hits")
    if np.any(assoc_mask):
        sc_yz = axes[1].scatter(
            y[assoc_mask], z[assoc_mask],
            s=24, c=hit_best_weight[assoc_mask], cmap="viridis", vmin=0.0, vmax=1.0,
            label="Associated hits"
        )
        fig.colorbar(sc_yz, ax=axes[1], fraction=0.046, pad=0.04, label="Assoc weight")
    axes[1].set_title(f"Event {event_idx} | PFO {pfo_idx} | YZ")
    axes[1].set_xlabel("y")
    axes[1].set_ylabel("z")
    axes[1].grid(alpha=0.2)
    axes[1].legend(loc="best")
    
    summary = (
        f"pid={pfo_row['pid']} class={pfo_row['pid_class']} q={pfo_row['charge']:+.1f}\n"
        f"p=({pfo_row['px']:.2f}, {pfo_row['py']:.2f}, {pfo_row['pz']:.2f})\n"
        f"|p|={pfo_row['p_mod']:.2f}  log|p|={pfo_row['log_p_mod']:.2f}\n"
        f"assoc_hits={pfo_row['n_associated_hits']}  sum_w={pfo_row['sum_assoc_weight']:.2f}"
    )
    fig.text(
        0.5, 0.01, summary,
        ha="center", va="bottom", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "gray"}
    )
    
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def test_event_diagnostics(
    npz_path: str,
    event_indices: List[int],
    output_dir: str = "test",
    mode: str = "memory",
) -> None:
    """
    Test exhaustivo de inspección por evento.
    
    Para cada evento:
    - guarda CSV de PFOs (`pfos.csv`)
    - guarda CSV de asociaciones (`hit_associations.csv`)
    - genera un plot XY/YZ por PFO (`pfo_XXX.png`)
    """
    print("\n" + "=" * 60)
    print("TEST: Event Diagnostics (CSV + PFO plots)")
    print("=" * 60)
    
    dataset = PFAutoRegressorDataset([npz_path], mode=mode)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput root: {out_root.resolve()}")
    print(f"Selected events: {event_indices}")
    
    pfo_fields = [
        "event_idx", "pfo_idx", "particle_idx", "pid", "pid_class",
        "charge", "gen_status", "px", "py", "pz",
        "p_mod", "log_p_mod", "n_associated_hits", "sum_assoc_weight", "max_assoc_weight",
    ]
    assoc_fields = [
        "event_idx", "hit_idx", "x", "y", "z",
        "energy", "p", "detector_type", "pfo_idx", "particle_idx", "assoc_weight",
    ]
    
    for event_idx in event_indices:
        event_dir = out_root / f"event_{event_idx:06d}"
        event_dir.mkdir(parents=True, exist_ok=True)
        
        diag = _collect_event_diagnostics(dataset, event_idx)
        pfo_rows = diag["pfo_rows"]
        assoc_rows = diag["assoc_rows"]
        
        _write_csv_rows(event_dir / "pfos.csv", pfo_rows, pfo_fields)
        _write_csv_rows(event_dir / "hit_associations.csv", assoc_rows, assoc_fields)
        
        for pfo_row in pfo_rows:
            pfo_idx = int(pfo_row["pfo_idx"])
            plot_path = event_dir / f"pfo_{pfo_idx:03d}.png"
            _plot_pfo_projections(
                plot_path,
                event_idx=event_idx,
                pfo_row=pfo_row,
                hits=diag["hits"],
                hit_best_pfo=diag["hit_best_pfo"],
                hit_best_weight=diag["hit_best_weight"],
            )
        
        print(
            f"  Event {event_idx}: "
            f"{len(pfo_rows)} PFOs, {len(assoc_rows)} hits -> {event_dir}"
        )
    
    dataset.close()
    print("\n✓ Event diagnostics test passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tests y diagnóstico de PFAutoRegressorDataset"
    )
    parser.add_argument("npz_file", type=str, help="Archivo .npz con eventos PF")
    parser.add_argument(
        "--events",
        type=str,
        default="0",
        help="Eventos a procesar (ej: '0,2,5-8')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test",
        help="Carpeta de salida para resultados por evento",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="memory",
        choices=["memory", "lazy"],
        help="Modo del dataset para diagnóstico",
    )
    parser.add_argument(
        "--skip-basic-tests",
        action="store_true",
        help="Omite los tests básicos y ejecuta solo el diagnóstico exhaustivo",
    )
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    dataset_for_index = PFAutoRegressorDataset([args.npz_file], mode=args.mode)
    total_events = dataset_for_index.len()
    selected_events = _parse_event_selection(args.events, total_events)
    dataset_for_index.close()
    
    if not args.skip_basic_tests:
        test_dataset_loading(args.npz_file)
        test_data_structure(args.npz_file)
        test_dataloader(args.npz_file)
        test_pfo_validity(args.npz_file)
        test_preprocessor(args.npz_file)
    
    test_event_diagnostics(
        npz_path=args.npz_file,
        event_indices=selected_events,
        output_dir=args.output_dir,
        mode=args.mode,
    )
    
    print("\n" + "=" * 60)
    print("ALL REQUESTED TESTS PASSED!")
    print("=" * 60)
