"""GPU Utilities für SetFit Training mit ModernBERT.

Bietet geräteunabhängiges Model-Loading und Training-Konfiguration,
die sowohl auf GPU (mit Triton Beschleunigung) als auch auf CPU funktioniert.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from setfit import SetFitModel, TrainingArguments

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Konstanten
MODEL_NAME = "nomic-ai/modernbert-embed-base"
DEFAULT_BATCH_SIZE_GPU = 4 #(A40, 46G)
DEFAULT_BATCH_SIZE_CPU = 2
MAX_TOKENS = 8100  # ModernBERT Limit ist 8192, Puffer für special tokens lassen
OVERLAP_RATIO = 0.2  # 20% Überlappung für Chunking


@dataclass
class DeviceConfig:
    """Konfiguration für gerätespezifische Einstellungen.
    
    Attributes:
        device: Der torch device String ('cuda' oder 'cpu').
        use_amp: Ob automatic mixed precision verwendet werden soll.
        batch_size: Empfohlene Batch-Größe für das Gerät.
    """
    device: str
    use_amp: bool
    batch_size: int


def get_device_config() -> DeviceConfig:
    """Erkennt verfügbare Hardware und gibt optimale Konfiguration zurück.
    
    Erkennt automatisch CUDA Verfügbarkeit und konfiguriert Einstellungen
    entsprechend für GPU oder CPU Ausführung.
    
    Returns:
        DeviceConfig mit gerätespezifischen Einstellungen für GPU oder CPU.
    """
    if torch.cuda.is_available():
        device = "cuda"
        # CUDNN für konsistente Eingabegrößen optimieren (typisch für NLP)
        torch.backends.cudnn.benchmark = True
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU erkannt: {gpu_name}")
        logger.info(f"GPU Speicher: {gpu_memory:.1f} GB")
        
        return DeviceConfig(
            device=device,
            use_amp=True,  # Mixed precision für schnelleres Training
            batch_size=DEFAULT_BATCH_SIZE_GPU,
        )
    else:
        logger.info("Keine GPU erkannt, verwende CPU Modus")
        logger.info("Hinweis: Training wird auf CPU langsamer sein")
        
        return DeviceConfig(
            device="cpu",
            use_amp=False,  # AMP bringt auf CPU keinen Vorteil
            batch_size=DEFAULT_BATCH_SIZE_CPU,
        )


def load_model(config: DeviceConfig, model_name: str = MODEL_NAME) -> SetFitModel:
    """Lädt SetFit Modell mit geräteangepassten Einstellungen.
    
    Handhabt die Komplexität des Ladens von ModernBERT-basierten Modellen mit
    korrekten Einstellungen für GPU und CPU.
    
    Args:
        config: Device-Konfiguration von get_device_config().
        model_name: HuggingFace Model-Identifier.
        
    Returns:
        SetFitModel konfiguriert für das Zielgerät.
    """
    logger.info(f"Lade Modell: {model_name}")
    logger.info(f"Gerät: {config.device}, AMP: {config.use_amp}")
    
    model = SetFitModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        device=config.device,
    )
    
    logger.info("Modell erfolgreich geladen")
    return model


def get_training_args(config: DeviceConfig, **kwargs) -> TrainingArguments:
    """Erstellt TrainingArguments mit geräteoptimierten Standardwerten.
    
    Bietet sinnvolle Standardwerte für SetFit Training, die für die
    erkannte Hardware (GPU oder CPU) optimiert sind.
    
    Args:
        config: Device-Konfiguration von get_device_config().
        **kwargs: Überschreibt beliebige Standard-Trainingsargumente.
        
    Returns:
        TrainingArguments konfiguriert für optimale Leistung.
    """
    defaults = {
        "num_epochs": 1,
        "batch_size": config.batch_size,
        "use_amp": config.use_amp,
        "sampling_strategy": "undersampling",
        "eval_strategy": "no",
        "save_strategy": "no",
    }
    # kwargs können defaults überschreiben
    defaults.update(kwargs)
    
    logger.info(f"Training-Argumente: epochs={defaults['num_epochs']}, "
                f"batch_size={defaults['batch_size']}, use_amp={defaults['use_amp']}")
    
    return TrainingArguments(**defaults)


def check_triton_available() -> bool:
    """Prüft ob Triton installiert und verfügbar ist.
    
    Triton wird für torch.compile mit dem inductor Backend benötigt,
    welches ModernBERT für optimierte Embeddings auf GPU verwendet.
    
    Returns:
        True wenn Triton verfügbar ist, sonst False.
    """
    try:
        import triton
        logger.info(f"Triton Version: {triton.__version__}")
        return True
    except ImportError:
        logger.warning("Triton nicht installiert - GPU Beschleunigung möglicherweise eingeschränkt")
        return False
