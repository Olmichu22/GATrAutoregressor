"""
Lightning DataModule para el dataset PFAutoRegressor.

Integra el dataset de Particle Flow con PyTorch Lightning para
facilitar el entrenamiento del modelo GATrAutoRegressor.
"""

import lightning as L
from typing import List, Optional
from src.datasets.pf_autoregressor_dataset import (
    PFAutoRegressorDataset,
    PFPreprocessor,
    PFPreprocessorConfig,
    make_pf_splits,
)
from torch_geometric.loader import DataLoader


class PFAutoRegressorDataModule(L.LightningDataModule):
    """
    DataModule de Lightning para el dataset PFAutoRegressor.
    
    Args:
        dataset_paths: Lista de rutas a archivos .npz generados por convert_to_nn_format.py
        batch_size: Tamaño del batch
        num_workers: Número de workers para carga de datos
        val_ratio: Fracción de datos para validación
        mode: 'memory' para cargar todo en RAM, 'lazy' para leer bajo demanda
        preprocessor_config: Configuración del preprocesador (opcional)
        pin_memory: Si usar pin_memory en DataLoader
    
    Uso:
        # Con configuración por defecto (log_energy activado)
        datamodule = PFAutoRegressorDataModule(
            dataset_paths=['data1.npz', 'data2.npz'],
            batch_size=32,
            mode='memory',
        )
        
        # Con preprocesador personalizado
        config = PFPreprocessorConfig(
            normalize_coords=True,
            log_energy=True,
            log_momentum=True,
            normalize_energy=True,
        )
        datamodule = PFAutoRegressorDataModule(
            dataset_paths=['data.npz'],
            preprocessor_config=config,
        )
        
        trainer = L.Trainer(...)
        trainer.fit(model, datamodule)
    """
    
    def __init__(
        self,
        dataset_paths: List[str],
        batch_size: int = 32,
        num_workers: int = 4,
        val_ratio: float = 0.1,
        mode: str = 'memory',
        preprocessor_config: Optional[PFPreprocessorConfig] = None,
        pin_memory: bool = True,
    ):
        super().__init__()
        self.dataset_paths = dataset_paths
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.mode = mode
        self.preprocessor_config = preprocessor_config or PFPreprocessorConfig()
        self.pin_memory = pin_memory
        
        # Se inicializan en setup()
        self.train_dataset = None
        self.val_dataset = None
        self.full_dataset = None
        self.preprocessor = None
    
    def setup(self, stage: Optional[str] = None):
        """
        Configura los datasets para train/val.
        
        Args:
            stage: 'fit', 'validate', 'test', o 'predict'
        """
        if stage == "fit" or stage is None:
            # Crear preprocesador con la configuración
            self.preprocessor = PFPreprocessor(self.preprocessor_config)
            
            self.train_dataset, self.val_dataset = make_pf_splits(
                paths=self.dataset_paths,
                val_ratio=self.val_ratio,
                mode=self.mode,
                preprocessor=self.preprocessor,
            )
            
            # Guardar referencia al dataset completo para acceder a stats
            if hasattr(self.train_dataset, 'dataset'):
                self.full_dataset = self.train_dataset.dataset
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def get_preprocessor(self) -> Optional[PFPreprocessor]:
        """
        Obtiene el preprocesador del dataset.
        
        Returns:
            PFPreprocessor con estadísticas ajustadas
        """
        return self.preprocessor
    
    def get_preprocessor_state(self) -> Optional[dict]:
        """
        Obtiene el estado del preprocesador para serialización.
        
        Returns:
            dict con configuración y estadísticas
        """
        if self.preprocessor is not None:
            return self.preprocessor.get_state()
        return None
    
    def teardown(self, stage: Optional[str] = None):
        """Limpia recursos al finalizar."""
        if self.full_dataset is not None and hasattr(self.full_dataset, 'close'):
            self.full_dataset.close()
