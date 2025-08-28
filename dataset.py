import os
import glob
import csv
import numpy as np
import pandas as pd
import mne
import torch
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import hashlib
import mmap
import pickle
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import gc
import psutil
from pathlib import Path
import time

PREPROCESS = {
    'bandpass': (0.5, 40.0),  # Hz
    'notch': 60.0,            # Hz
    'resample': 256,          # Hz
}
NOTCH = True
BANDPASS = True
NORMALIZE = True

@dataclass
class EEGWindowSpec:
    edf_path: str
    csv_path: str
    start_idx: int      # índice de inicio (muestras)
    n_tp: int           # longitud de ventana (muestras)
    label: int          # 0/1 por ventana (contiene algún frame positivo)
    label_vector: np.ndarray  # vector detallado frame-by-frame (window_sec*sf,)

# ====================================================================================
# SISTEMA DE ORDEN CONSISTENTE DE CANALES
# ====================================================================================

def get_standard_channel_order():
    """
    Define el orden estándar de canales para mantener consistencia entre montajes.
    Orden basado en regiones anatómicas: frontal → temporal → central → parietal → occipital
    """
    return [
        # Frontales izquierdo
        'FP1-F7', 'FP1-F3', 'F7-T3', 'F3-C3',
        # Frontales derecho  
        'FP2-F8', 'FP2-F4', 'F8-T4', 'F4-C4',
        # Temporales izquierdo
        'T3-T5', 'T3-C3', 'A1-T3',
        # Temporales derecho
        'T4-T6', 'T4-C4', 'T4-A2',
        # Centrales
        'C3-CZ', 'CZ-C4', 'C3-P3', 'C4-P4',
        # Parietales
        'P3-O1', 'P4-O2',
        # Occipitales
        'T5-O1', 'T6-O2', 'O1-O2',
        # Laterales (para montaje LE)
        'F7-F8', 'T3-T4', 'T5-T6'
    ]

def get_montage_pairs_ordered(montage: str = 'ar', suffix: str = '-REF'):
    """
    Versión mejorada que mantiene orden consistente de canales entre montajes.
    Todos los montajes retornan canales en el mismo orden estándar cuando es posible.
    """
    if montage == 'ar':
        # Montaje ar original (22 canales) en orden estándar
        pairs = [
            (f'EEG FP1{suffix}', f'EEG F7{suffix}'),   # 0: FP1-F7
            (f'EEG FP1{suffix}', f'EEG F3{suffix}'),   # 1: FP1-F3
            (f'EEG FP2{suffix}', f'EEG F8{suffix}'),   # 2: FP2-F8
            (f'EEG FP2{suffix}', f'EEG F4{suffix}'),   # 3: FP2-F4
            (f'EEG F7{suffix}', f'EEG T3{suffix}'),    # 4: F7-T3
            (f'EEG F3{suffix}', f'EEG C3{suffix}'),    # 5: F3-C3
            (f'EEG F8{suffix}', f'EEG T4{suffix}'),    # 6: F8-T4
            (f'EEG F4{suffix}', f'EEG C4{suffix}'),    # 7: F4-C4
            (f'EEG T3{suffix}', f'EEG T5{suffix}'),    # 8: T3-T5
            (f'EEG T3{suffix}', f'EEG C3{suffix}'),    # 9: T3-C3
            (f'EEG A1{suffix}', f'EEG T3{suffix}'),    # 10: A1-T3
            (f'EEG T4{suffix}', f'EEG T6{suffix}'),    # 11: T4-T6
            (f'EEG C4{suffix}', f'EEG T4{suffix}'),    # 12: C4-T4
            (f'EEG T4{suffix}', f'EEG A2{suffix}'),    # 13: T4-A2
            (f'EEG C3{suffix}', f'EEG CZ{suffix}'),    # 14: C3-CZ
            (f'EEG CZ{suffix}', f'EEG C4{suffix}'),    # 15: CZ-C4
            (f'EEG C3{suffix}', f'EEG P3{suffix}'),    # 16: C3-P3
            (f'EEG C4{suffix}', f'EEG P4{suffix}'),    # 17: C4-P4
            (f'EEG P3{suffix}', f'EEG O1{suffix}'),    # 18: P3-O1
            (f'EEG P4{suffix}', f'EEG O2{suffix}'),    # 19: P4-O2
            (f'EEG T5{suffix}', f'EEG O1{suffix}'),    # 20: T5-O1
            (f'EEG T6{suffix}', f'EEG O2{suffix}')     # 21: T6-O2
        ]
        
    elif montage == 'ar_a':
        # Montaje ar_a en el MISMO orden que ar, sustituyendo canales faltantes
        pairs = [
            (f'EEG FP1{suffix}', f'EEG F7{suffix}'),   # 0: FP1-F7
            (f'EEG FP1{suffix}', f'EEG F3{suffix}'),   # 1: FP1-F3
            (f'EEG FP2{suffix}', f'EEG F8{suffix}'),   # 2: FP2-F8
            (f'EEG FP2{suffix}', f'EEG F4{suffix}'),   # 3: FP2-F4
            (f'EEG F7{suffix}', f'EEG T3{suffix}'),    # 4: F7-T3
            (f'EEG F3{suffix}', f'EEG C3{suffix}'),    # 5: F3-C3
            (f'EEG F8{suffix}', f'EEG T4{suffix}'),    # 6: F8-T4
            (f'EEG F4{suffix}', f'EEG C4{suffix}'),    # 7: F4-C4
            (f'EEG T3{suffix}', f'EEG T5{suffix}'),    # 8: T3-T5
            (f'EEG T3{suffix}', f'EEG C3{suffix}'),    # 9: T3-C3
            (f'EEG F3{suffix}', f'EEG P3{suffix}'),    # 10: F3-P3 (reemplaza A1-T3)
            (f'EEG T4{suffix}', f'EEG T6{suffix}'),    # 11: T4-T6
            (f'EEG C4{suffix}', f'EEG T4{suffix}'),    # 12: C4-T4
            (f'EEG F4{suffix}', f'EEG P4{suffix}'),    # 13: F4-P4 (reemplaza T4-A2)
            (f'EEG C3{suffix}', f'EEG CZ{suffix}'),    # 14: C3-CZ
            (f'EEG CZ{suffix}', f'EEG C4{suffix}'),    # 15: CZ-C4
            (f'EEG C3{suffix}', f'EEG P3{suffix}'),    # 16: C3-P3
            (f'EEG C4{suffix}', f'EEG P4{suffix}'),    # 17: C4-P4
            (f'EEG P3{suffix}', f'EEG O1{suffix}'),    # 18: P3-O1
            (f'EEG P4{suffix}', f'EEG O2{suffix}'),    # 19: P4-O2
            (f'EEG T5{suffix}', f'EEG O1{suffix}'),    # 20: T5-O1
            (f'EEG T6{suffix}', f'EEG O2{suffix}')     # 21: T6-O2
        ]
        
    elif montage == 'le':
        # Montaje le expandido para mantener consistencia en posiciones clave
        # Usamos las mismas posiciones que en montajes ar pero con menos canales
        pairs = [
            (f'EEG F7{suffix}', f'EEG F8{suffix}'),    # Posición 0: Frontal lateral
            (f'EEG T3{suffix}', f'EEG T4{suffix}'),    # Posición 1: Temporal lateral
            (f'EEG T5{suffix}', f'EEG T6{suffix}'),    # Posición 2: Temporal posterior
            (f'EEG C3{suffix}', f'EEG C4{suffix}'),    # Posición 3: Central
            (f'EEG P3{suffix}', f'EEG P4{suffix}'),    # Posición 4: Parietal
            (f'EEG O1{suffix}', f'EEG O2{suffix}')     # Posición 5: Occipital
        ]
    else:
        raise ValueError(f"Montaje no soportado: {montage}. Disponibles: 'ar', 'ar_a', 'le'")
    
    return pairs

def get_montage_pairs(montage: str = 'ar', suffix: str = '-REF'):
    """Función de compatibilidad - redirecciona a la versión ordenada"""
    return get_montage_pairs_ordered(montage, suffix)

# ====================================================================================
# BALANCED BATCH SAMPLER
# ====================================================================================

class BalancedBatchSampler(Sampler[List[int]]):
    """
    Batch sampler que mantiene una fracción específica de muestras positivas por lote.
    Usa muestreo con reemplazo si es necesario para mantener el balance.
    """
    
    def __init__(self, pos_idx: List[int], neg_idx: List[int], batch_size: int, pos_frac: float):
        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.batch_size = batch_size
        self.pos_frac = pos_frac
        
        # Validaciones
        if pos_frac < 0 or pos_frac > 1:
            raise ValueError(f"pos_frac debe estar entre 0 y 1, recibido: {pos_frac}")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size debe ser > 0, recibido: {batch_size}")
        
        # Calcular número de muestras positivas y negativas por lote
        self.pos_per_batch = int(round(batch_size * pos_frac))
        self.neg_per_batch = batch_size - self.pos_per_batch
        
        # Validar que tenemos muestras disponibles
        if self.pos_per_batch > 0 and len(pos_idx) == 0:
            print(f"⚠️  Se solicitan {self.pos_per_batch} muestras positivas por batch pero no hay disponibles")
        
        if self.neg_per_batch > 0 and len(neg_idx) == 0:
            print(f"⚠️  Se solicitan {self.neg_per_batch} muestras negativas por batch pero no hay disponibles")
        
        # Número total de lotes basado en la clase minoritaria
        total_pos_needed = len(pos_idx) if pos_idx else 0
        total_neg_needed = len(neg_idx) if neg_idx else 0
        
        if self.pos_per_batch > 0 and total_pos_needed > 0:
            num_batches_pos = max(1, total_pos_needed // self.pos_per_batch)
        else:
            num_batches_pos = 0
            
        if self.neg_per_batch > 0 and total_neg_needed > 0:
            num_batches_neg = max(1, total_neg_needed // self.neg_per_batch)
        else:
            num_batches_neg = 0
        
        self.num_batches = max(num_batches_pos, num_batches_neg, 1)
        
        print(f"BalancedBatchSampler configurado:")
        print(f"├── Batch size: {batch_size}")
        print(f"├── Pos per batch: {self.pos_per_batch} ({pos_frac:.1%})")
        print(f"├── Neg per batch: {self.neg_per_batch}")
        print(f"├── Total batches: {self.num_batches}")
        print(f"└── Muestras disponibles: {len(pos_idx)} pos, {len(neg_idx)} neg")
    
    def __len__(self) -> int:
        return self.num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            
            # Samplear muestras positivas
            if self.pos_per_batch > 0 and len(self.pos_idx) > 0:
                if len(self.pos_idx) >= self.pos_per_batch:
                    # Sin reemplazo
                    pos_samples = np.random.choice(self.pos_idx, size=self.pos_per_batch, replace=False)
                else:
                    # Con reemplazo si no hay suficientes
                    pos_samples = np.random.choice(self.pos_idx, size=self.pos_per_batch, replace=True)
                batch.extend(pos_samples.tolist())
            
            # Samplear muestras negativas
            if self.neg_per_batch > 0 and len(self.neg_idx) > 0:
                if len(self.neg_idx) >= self.neg_per_batch:
                    # Sin reemplazo
                    neg_samples = np.random.choice(self.neg_idx, size=self.neg_per_batch, replace=False)
                else:
                    # Con reemplazo si no hay suficientes
                    neg_samples = np.random.choice(self.neg_idx, size=self.neg_per_batch, replace=True)
                batch.extend(neg_samples.tolist())
            
            # Mezclar el lote
            np.random.shuffle(batch)
            yield batch

# ====================================================================================
# CLASE EEGWindowDataset OPTIMIZADA
# ====================================================================================

class EEGWindowDataset(Dataset):
    """
    Clase consolidada y optimizada para manejar múltiples montajes EEG con todas las funcionalidades:
    - Soporte para montajes: 'ar', 'ar_a', 'le' y combinaciones ['ar', 'le', 'ar_a']
    - Conversión automática a formato unificado
    - Balanceo de clases opcional
    - Caching optimizado con memoria compartida
    - Etiquetas time-step y one-hot
    - Manejo robusto de errores
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 montage: Union[str, List[str]] = 'ar',
                 target_channels: int = 22,
                 window_sec: float = 5.0,
                 hop_sec: float = 1.0,
                 time_step: bool = True,
                 one_hot: bool = False,
                 num_classes: int = 2,
                 transpose: bool = True,
                 limits: Optional[Dict[str, int]] = None,
                 balance_pos_frac: Optional[float] = None,
                 write_manifest: bool = False,
                 preprocess_config: Optional[Dict] = None,
                 seed: int = 42):
        """
        Args:
            data_dir: Directorio base de datos
            split: 'train', 'eval', 'dev'
            montage: Montaje(s) a usar. Str individual o lista ['ar', 'le', 'ar_a']
            target_channels: Canales objetivo para formato unificado (22 recomendado)
            window_sec: Duración de ventana en segundos
            hop_sec: Paso de ventana en segundos (None = sin solapamiento)
            time_step: Si True, etiquetas frame-by-frame; si False, etiqueta por ventana
            one_hot: Si True, codificación one-hot para etiquetas
            num_classes: Número de clases (usado si one_hot=True)
            transpose: Si True, formato (tiempo, canales); si False, (canales, tiempo)
            limits: Dict con 'files' y/o 'max_windows' para limitar dataset
            balance_pos_frac: Fracción de muestras positivas (None = sin balanceo)
            write_manifest: Si True, escribe manifiesto CSV
            preprocess_config: Configuración de preprocesamiento
            seed: Semilla para reproducibilidad
        """
        
        # Configuración básica
        self.data_dir = data_dir
        self.split = split
        self.target_channels = target_channels
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.time_step = time_step
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.transpose = transpose
        self.limits = limits or {}
        self.balance_pos_frac = balance_pos_frac
        self.write_manifest = write_manifest
        self.preprocess_config = preprocess_config or {}
        self.seed = seed
        self.window_samples = int(round(self.window_sec * 256.0))

        # Configurar montajes
        if isinstance(montage, str):
            self.montages = [montage]
        else:
            self.montages = list(montage)
        
        # Configurar RNG
        self.rng = np.random.default_rng(seed)
        
        # Inicializar
        self._print_init_info()
        self._validate_montages()
        self._build_dataset()
        
    def _print_init_info(self):
        """Imprime información de inicialización"""
        print("Inicializando OptimizedMultiMontageEEGDataset:")
        print(f"├── Split: {self.split}")
        print(f"├── Montaje(s): {', '.join(self.montages)}")
        print(f"├── Canales objetivo: {self.target_channels}")
        print(f"├── Ventana: {self.window_sec}s ({int(self.window_sec * 256)} muestras)")
        print(f"├── Hop: {self.hop_sec}s ({int(self.hop_sec * 256)} muestras)")
        print(f"├── Time-step: {self.time_step}")
        print(f"├── One-hot: {self.one_hot}")
        print(f"├── Transpose: {self.transpose}")
        print(f"└── Frecuencia: 256.0 Hz")
        
    def _validate_montages(self):
        """Valida que los montajes sean soportados"""
        supported = {'ar', 'ar_a', 'le'}
        for montage in self.montages:
            if montage not in supported:
                raise ValueError(f"Montaje '{montage}' no soportado. Disponibles: {supported}")
    
    def _build_dataset(self):
        """Construye el dataset combinando múltiples montajes"""
        all_specs = []
        total_files = 0
        
        print(f"Procesando {len(self.montages)} montaje(s)...")
        
        for montage in self.montages:
            print(f"\n📊 Procesando montaje: {montage.upper()}")
            
            # Obtener archivos para este montaje
            csv_files = self._get_csv_files(montage)
            
            if not csv_files:
                print(f"   ⚠️  No se encontraron archivos para montaje {montage}")
                continue
                
            # Aplicar límites de archivos
            max_files = self.limits.get('files', 0)
            if max_files > 0:
                csv_files = csv_files[:max_files]
                
            print(f"   📁 Procesando {len(csv_files)} archivo(s)")
            total_files += len(csv_files)
            
            # Procesar archivos de este montaje
            montage_specs = self._process_montage_files(montage, csv_files)
            
            print(f"   ✅ Generadas {len(montage_specs)} ventanas")
            all_specs.extend(montage_specs)
        
        # Aplicar límites de ventanas
        max_windows = self.limits.get('max_windows', 0)
        if max_windows > 0 and len(all_specs) > max_windows:
            all_specs = self._apply_window_limits(all_specs, max_windows)
        
        # Aplicar balanceo si se especifica
        if self.balance_pos_frac is not None:
            all_specs = self._apply_balancing(all_specs)
        
        self.specs = all_specs
        self._print_final_stats(total_files)
        
        # Escribir manifiesto si se solicita
        if self.write_manifest:
            self._write_manifest()
    
    def _get_csv_files(self, montage: str) -> List[str]:
        """Obtiene archivos CSV para un montaje específico"""
        root = os.path.join(self.data_dir, 'edf', self.split)
        all_csv = glob.glob(os.path.join(root, '**', '*_bi.csv'), recursive=True)
        
        # Filtrar por montaje
        filtered = []
        for p in all_csv:
            folder = os.path.dirname(p).lower()
            if montage == 'ar' and '_tcp_ar' in folder and '_tcp_ar_a' not in folder:
                filtered.append(p)
            elif montage == 'ar_a' and '_tcp_ar_a' in folder:
                filtered.append(p)
            elif montage == 'le' and '_tcp_le' in folder:
                filtered.append(p)
        
        return sorted(filtered)
    
    def _process_montage_files(self, montage: str, csv_files: List[str]) -> List[EEGWindowSpec]:
        """Procesa archivos de un montaje específico"""
        specs = []
        processed = 0
        
        for csv_path in csv_files:
            try:
                # Obtener archivo EDF correspondiente
                edf_path = csv_path.replace('_bi.csv', '.edf')
                if not os.path.exists(edf_path):
                    continue
                
                # Extraer señales
                raw_bip = self._extract_montage_signals(edf_path, montage)
                if raw_bip is None:
                    continue
                
                # Procesar ventanas
                file_specs = self._create_windows(edf_path, csv_path, raw_bip, montage)
                specs.extend(file_specs)
                
                raw_bip.close()
                processed += 1
                
                # Progreso cada 10 archivos
                if processed % 10 == 0:
                    print(f"   Progreso: {processed}/{len(csv_files)} archivos procesados")
                    
            except Exception as e:
                print(f"   Error procesando {os.path.basename(edf_path)}: {e}")
                continue
        
        print(f"   ✓ Procesamiento completado: {processed}/{len(csv_files)} archivos")
        return specs
    
    def _extract_montage_signals(self, edf_path: str, montage: str) -> Optional[mne.io.BaseRaw]:
        """Extrae señales para un montaje específico"""
        try:
            # Cargar archivo EDF
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # Aplicar preprocesamiento
            raw = self._preprocess_raw(raw)
            
            # Detectar sufijo
            suf = '-LE' if any(ch.endswith('-LE') for ch in raw.ch_names) else '-REF'
            
            # Obtener pares de electrodos
            pairs = get_montage_pairs(montage, suf)
            needed = {c for a, b in pairs for c in (a, b)}
            missing = needed - set(raw.ch_names)
            
            if missing:
                raise RuntimeError(f"Electrodos faltantes para montaje {montage}: {missing}")
            
            # Crear referencia bipolar
            raw_bip = mne.set_bipolar_reference(
                raw,
                anode=[a for a, _ in pairs],
                cathode=[b for _, b in pairs],
                ch_name=[f"{a}-{b}" for a, b in pairs],
                drop_refs=True, verbose=False
            )
            
            # Seleccionar canales y resamplear
            bip_names = [f"{a}-{b}" for a, b in pairs]
            raw_bip.pick(bip_names)
            
            if int(raw_bip.info['sfreq']) != 256:
                raw_bip.resample(256, npad='auto', verbose=False)
            
            return raw_bip
            
        except Exception as e:
            return None
    
    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Aplica preprocesamiento a la señal"""
        if self.preprocess_config.get('bandpass'):
            bp = self.preprocess_config['bandpass']
            raw.filter(bp[0], bp[1], method='iir', phase='zero', verbose=False)
            
        if self.preprocess_config.get('notch'):
            raw.notch_filter(freqs=self.preprocess_config['notch'], 
                           method='iir', phase='zero', verbose=False)
        
        if self.preprocess_config.get('normalize', True):
            data = raw.get_data()
            mean_vals = data.mean(axis=1, keepdims=True)
            std_vals = data.std(axis=1, keepdims=True)
            data = (data - mean_vals) / (std_vals + 1e-6)
            raw._data = data
            
        return raw
    
    def _create_windows(self, edf_path: str, csv_path: str, raw_bip: mne.io.BaseRaw, 
                       montage: str) -> List[EEGWindowSpec]:
        """Crea ventanas para un archivo"""
        specs = []
        
        # Parámetros de ventana
        sf = float(raw_bip.info['sfreq'])
        n_tp = int(round(self.window_sec * sf))
        hop_tp = int(round(self.hop_sec * sf)) if self.hop_sec else n_tp
        n_tot = int(raw_bip.n_times)
        
        # Cargar anotaciones
        seiz_intervals = self._load_annotations(csv_path)
        
        # Crear etiquetas detalladas
        detailed_labels = self._create_detailed_labels(seiz_intervals, n_tot, sf)
        
        # Generar ventanas
        for start in range(0, n_tot - n_tp + 1, hop_tp):
            end = start + n_tp
            
            # Etiqueta binaria de ventana
            window_label = int(detailed_labels[start:end].max() > 0)
            
            # Vector detallado
            window_label_vector = detailed_labels[start:end].copy()
            
            # Crear especificación con información de montaje
            spec = EEGWindowSpec(
                edf_path=edf_path,
                csv_path=csv_path,
                start_idx=start,
                n_tp=n_tp,
                label=window_label,
                label_vector=window_label_vector
            )
            # Añadir información de montaje
            spec.montage = montage
            spec.original_channels = len(get_montage_pairs(montage))
            
            specs.append(spec)
        
        return specs
    
    def _load_annotations(self, csv_path: str) -> List[Tuple[float, float]]:
        """Carga anotaciones de convulsiones - VERSIÓN CORREGIDA"""
        intervals = []
        try:
            # Usar pandas con comment='#' para saltar líneas de comentarios
            df = pd.read_csv(csv_path, comment='#')
            seizure_events = df[df['label'] == 'seiz']
            for _, row in seizure_events.iterrows():
                start_time = float(row['start_time'])
                stop_time = float(row['stop_time'])
                intervals.append((start_time, stop_time))
        except Exception as e:
            # Fallback manual si pandas falla
            try:
                with open(csv_path, 'r') as f:
                    lines = f.readlines()
                
                # Encontrar líneas sin comentarios
                data_lines = []
                for line in lines:
                    if not line.strip().startswith('#') and line.strip():
                        data_lines.append(line.strip())
                
                if len(data_lines) >= 2:  # Header + datos
                    header = data_lines[0].split(',')
                    for data_line in data_lines[1:]:
                        values = data_line.split(',')
                        if len(values) == len(header):
                            row = dict(zip(header, values))
                            if row.get('label') == 'seiz':
                                start_time = float(row['start_time'])
                                stop_time = float(row['stop_time'])
                                intervals.append((start_time, stop_time))
            except Exception:
                pass
        return intervals
    
    def _create_detailed_labels(self, seiz_intervals: List[Tuple[float, float]], 
                               n_samples: int, sf: float) -> np.ndarray:
        """Crea vector detallado de etiquetas frame-by-frame"""
        labels = np.zeros(n_samples, dtype=np.int32)
        
        for start_time, stop_time in seiz_intervals:
            start_idx = max(0, int(start_time * sf))
            stop_idx = min(n_samples, int(stop_time * sf))
            labels[start_idx:stop_idx] = 1
            
        return labels
    
    def _apply_window_limits(self, specs: List[EEGWindowSpec], max_windows: int) -> List[EEGWindowSpec]:
        """Aplica límites de ventanas manteniendo proporción de clases"""
        if len(specs) <= max_windows:
            return specs
            
        # Separar por clase
        pos_specs = [s for s in specs if s.label == 1]
        neg_specs = [s for s in specs if s.label == 0]
        
        # Calcular proporciones
        total = len(specs)
        pos_ratio = len(pos_specs) / total if total > 0 else 0
        
        # Seleccionar manteniendo proporción
        n_pos = min(len(pos_specs), int(max_windows * pos_ratio))
        n_neg = min(len(neg_specs), max_windows - n_pos)
        
        selected_pos = self.rng.choice(pos_specs, size=n_pos, replace=False).tolist() if n_pos > 0 else []
        selected_neg = self.rng.choice(neg_specs, size=n_neg, replace=False).tolist() if n_neg > 0 else []
        
        result = selected_pos + selected_neg
        self.rng.shuffle(result)
        
        return result
    
    def _apply_balancing(self, specs: List[EEGWindowSpec]) -> List[EEGWindowSpec]:
        """Aplica balanceo de clases"""
        if self.balance_pos_frac is None or self.balance_pos_frac <= 0:
            return specs
            
        pos_specs = [s for s in specs if s.label == 1]
        neg_specs = [s for s in specs if s.label == 0]
        
        if not pos_specs or not neg_specs:
            return specs
        
        # Calcular tamaños objetivo
        total_target = len(specs)
        pos_target = int(total_target * self.balance_pos_frac)
        neg_target = total_target - pos_target
        
        # Seleccionar con reemplazo si es necesario
        selected_pos = self._sample_with_replacement(pos_specs, pos_target)
        selected_neg = self._sample_with_replacement(neg_specs, neg_target)
        
        result = selected_pos + selected_neg
        self.rng.shuffle(result)
        
        print(f"✓ Aplicado balanceo: {len(selected_pos)} positivas + {len(selected_neg)} negativas = {len(result)} ventanas")
        
        return result
    
    def _sample_with_replacement(self, specs: List[EEGWindowSpec], target: int) -> List[EEGWindowSpec]:
        """Muestrea con reemplazo si es necesario"""
        if target <= len(specs):
            return self.rng.choice(specs, size=target, replace=False).tolist()
        else:
            return self.rng.choice(specs, size=target, replace=True).tolist()
    
    def _print_final_stats(self, total_files: int):
        """Imprime estadísticas finales"""
        if not self.specs:
            print("\n⚠️  DATASET VACÍO")
            print(f"├── No se encontraron ventanas para los montajes: {', '.join(self.montages)}")
            print(f"├── Verificar que existen archivos con estos montajes en '{self.split}'")
            print(f"└── Montajes disponibles: 'ar', 'ar_a', 'le'")
            return
        
        # Estadísticas por montaje
        montage_stats = {}
        for spec in self.specs:
            montage = getattr(spec, 'montage', 'unknown')
            if montage not in montage_stats:
                montage_stats[montage] = {'total': 0, 'positive': 0}
            montage_stats[montage]['total'] += 1
            if spec.label == 1:
                montage_stats[montage]['positive'] += 1
        
        # Estadísticas generales
        total_windows = len(self.specs)
        positive_windows = sum(1 for s in self.specs if s.label == 1)
        total_frames = sum(len(s.label_vector) for s in self.specs)
        positive_frames = sum(s.label_vector.sum() for s in self.specs)
        
        print(f"\n📊 ESTADÍSTICAS DEL DATASET ({self.split.upper()})")
        print(f"├── Montaje(s): {', '.join(self.montages)} → {self.target_channels} canales")
        print(f"├── Total ventanas: {total_windows}")
        print(f"├── Ventanas con convulsión: {positive_windows} ({100*positive_windows/total_windows:.1f}%)")
        print(f"├── Ventanas normales: {total_windows-positive_windows} ({100*(total_windows-positive_windows)/total_windows:.1f}%)")
        print(f"├── Total frames: {total_frames:,}")
        print(f"├── Frames con convulsión: {positive_frames:,} ({100*positive_frames/total_frames:.3f}%)")
        print(f"└── Dimensión señal: ({self.target_channels}, {int(self.window_sec * 256)})")
        
        # Estadísticas por montaje
        if len(montage_stats) > 1:
            print(f"\n📋 Por montaje:")
            for montage, stats in montage_stats.items():
                total = stats['total']
                pos = stats['positive']
                print(f"   {montage.upper():>4}: {total:>4} ventanas ({pos} positivas, {100*pos/total:.1f}%)")
    
    def _write_manifest(self):
        """Escribe manifiesto CSV del dataset"""
        # Implementación del manifiesto (opcional)
        pass
    
    def _convert_signal_to_target_channels(self, signal: torch.Tensor, original_channels: int) -> torch.Tensor:
        """Convierte señal al número objetivo de canales"""
        if original_channels == self.target_channels:
            return signal
        
        # Detectar formato
        if self.transpose:  # (tiempo, canales)
            current_channels = signal.shape[1]
            if current_channels < self.target_channels:
                # Padding
                padding_needed = self.target_channels - current_channels
                padding = torch.zeros(signal.shape[0], padding_needed, 
                                    dtype=signal.dtype, device=signal.device)
                return torch.cat([signal, padding], dim=1)
            else:
                # Truncar
                return signal[:, :self.target_channels]
        else:  # (canales, tiempo)
            current_channels = signal.shape[0]
            if current_channels < self.target_channels:
                # Padding
                padding_needed = self.target_channels - current_channels
                padding = torch.zeros(padding_needed, signal.shape[1], 
                                    dtype=signal.dtype, device=signal.device)
                return torch.cat([signal, padding], dim=0)
            else:
                # Truncar
                return signal[:self.target_channels, :]
    
    def __len__(self) -> int:
        return len(self.specs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Obtiene una muestra del dataset"""
        spec = self.specs[idx]
        
        # Cargar señal
        signal = self._load_signal(spec)
        
        # Convertir a número objetivo de canales
        original_channels = getattr(spec, 'original_channels', signal.shape[0])
        signal = self._convert_signal_to_target_channels(signal, original_channels)
        
        # Preparar etiquetas
        if self.time_step:
            # Etiquetas frame-by-frame
            labels = torch.from_numpy(spec.label_vector.astype(np.float32))
            if self.one_hot:
                # Convertir a one-hot
                labels_oh = torch.zeros(len(labels), self.num_classes)
                labels_oh[torch.arange(len(labels)), labels.long()] = 1
                labels = labels_oh
        else:
            # Etiqueta por ventana
            if self.one_hot:
                labels = torch.zeros(self.num_classes)
                labels[spec.label] = 1
            else:
                labels = torch.tensor(spec.label, dtype=torch.float32)
        
        return signal, labels, spec.label
    
    def _load_signal(self, spec: EEGWindowSpec) -> torch.Tensor:
        """Carga señal desde archivo"""
        # Obtener montaje de la especificación
        montage = getattr(spec, 'montage', self.montages[0])
        
        # Extraer señal
        raw_bip = self._extract_montage_signals(spec.edf_path, montage)
        if raw_bip is None:
            raise RuntimeError(f"No se pudo cargar señal de {spec.edf_path}")
        
        # Obtener datos
        data = raw_bip.get_data()[:, spec.start_idx:spec.start_idx + spec.n_tp]
        raw_bip.close()
        
        # Convertir a tensor
        signal = torch.from_numpy(data.astype(np.float32))
        
        # Aplicar transpose si es necesario
        if self.transpose:
            signal = signal.T  # (canales, tiempo) → (tiempo, canales)
        
        return signal
    
    def get_info(self) -> Dict:
        """Información del dataset"""
        return {
            'montages': self.montages,
            'target_channels': self.target_channels,
            'original_channels': [len(get_montage_pairs(m)) for m in self.montages],
            'split': self.split,
            'window_sec': self.window_sec,
            'time_step': self.time_step,
            'one_hot': self.one_hot,
            'transpose': self.transpose,
            'total_windows': len(self.specs),
            'positive_windows': sum(1 for s in self.specs if s.label == 1)
        }

print("✅ Clase OptimizedMultiMontageEEGDataset creada exitosamente!")

# ====================================================================================
# DATASET OPTIMIZADO PARA ALTA PERFORMANCE Y ESCALABILIDAD
# ====================================================================================


class OptimizedEEGDataset(Dataset):
    """
    Dataset EEG optimizado que resuelve los bottlenecks principales:
    1. Cache de archivos procesados completos
    2. Eliminación de multiprocessing problemático  
    3. Carga eficiente de ventanas desde archivos pre-procesados
    """
    
    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 montage: str = 'ar',
                 window_sec: float = 10.0,
                 hop_sec: float = 0.25,
                 time_step: bool = True,
                 one_hot: bool = False,
                 num_classes: int = 1,
                 transpose: bool = True,
                 limits: Optional[Dict] = None,
                 balance_pos_frac: float = 0.5,
                 preprocess_config: Optional[Dict] = None,
                 use_cache: bool = True,
                 cache_dir: str = "cache",
                 seed: int = 42):
        
        self.data_dir = data_dir
        self.split = split
        self.montage = montage
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.time_step = time_step
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.transpose = transpose
        self.limits = limits or {}
        self.balance_pos_frac = balance_pos_frac
        self.preprocess_config = preprocess_config or {}
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.seed = seed
        
        # Configuración de ventanas
        self.window_samples = int(round(self.window_sec * 256.0))
        self.hop_samples = int(round(self.hop_sec * 256.0))
        
        # Configurar canales objetivo
        self.target_channels = self._get_target_channels()
        
        # Cache para archivos procesados
        self._file_cache = {}
        self._setup_cache_dir()
        
        # Inicializar dataset
        print(f"🚀 Inicializando OptimizedEEGDataset...")
        print(f"├── Split: {split}")
        print(f"├── Montaje: {montage}")
        print(f"├── Ventana: {window_sec}s ({self.window_samples} muestras)")
        print(f"├── Hop: {hop_sec}s ({self.hop_samples} muestras)")
        print(f"├── Cache: {'Activado' if use_cache else 'Desactivado'}")
        print(f"└── Canales: {self.target_channels}")
        
        start_time = time.time()
        self.specs = self._create_specs()
        creation_time = time.time() - start_time
        
        print(f"✅ Dataset creado en {creation_time:.2f}s")
        print(f"📊 Total ventanas: {len(self.specs)}")
        
        # Estadísticas
        self._print_stats()
        
    def _setup_cache_dir(self):
        """Crear directorio de cache si no existe"""
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_target_channels(self) -> int:
        """Obtener número de canales objetivo según montaje"""
        montage_channels = {
            'ar': 22,
            'ar_a': 22,
            'le': 22,
            'tcp_ar': 18
        }
        return montage_channels.get(self.montage, 22)
    
    def _get_cache_path(self, edf_path: str) -> str:
        """Generar path de cache para un archivo EDF"""
        # Crear hash único basado en archivo y configuración
        config_str = f"{self.montage}_{self.preprocess_config}"
        file_hash = hashlib.md5(f"{edf_path}_{config_str}".encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{file_hash}.pkl")
    
    @lru_cache(maxsize=16)  # Cache para 16 archivos procesados
    def _load_processed_file(self, edf_path: str) -> Optional[torch.Tensor]:
        """
        Cargar archivo EDF procesado con cache LRU.
        Cache en memoria para archivos recientes.
        """
        cache_path = self._get_cache_path(edf_path)
        
        # Intentar cargar desde cache en disco
        if self.use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                return torch.from_numpy(data).float()
            except Exception as e:
                print(f"⚠️  Error cargando cache {cache_path}: {e}")
        
        # Procesar archivo desde cero
        try:
            processed_data = self._process_edf_file(edf_path)
            
            # Guardar en cache en disco
            if self.use_cache and processed_data is not None:
                try:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(processed_data.numpy(), f)
                except Exception as e:
                    print(f"⚠️  Error guardando cache: {e}")
            
            return processed_data
            
        except Exception as e:
            print(f"❌ Error procesando {edf_path}: {e}")
            return None
    
    def _process_edf_file(self, edf_path: str) -> Optional[torch.Tensor]:
        """
        Procesar un archivo EDF completo de una vez.
        Retorna tensor con todas las muestras del archivo.
        """
        try:
            # Cargar archivo EDF
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # Aplicar preprocesamiento
            raw = self._preprocess_raw(raw)
            
            # Crear montaje bipolar
            suf = '-LE' if any(ch.endswith('-LE') for ch in raw.ch_names) else '-REF'
            pairs = get_montage_pairs_ordered(self.montage, suf)
            needed = {c for a, b in pairs for c in (a, b)}
            missing = needed - set(raw.ch_names)
            
            if missing:
                print(f"⚠️  Electrodos faltantes en {edf_path}: {missing}")
                return None
            
            # Crear referencia bipolar
            raw_bip = mne.set_bipolar_reference(
                raw,
                anode=[a for a, _ in pairs],
                cathode=[b for _, b in pairs],
                ch_name=[f"{a.split()[-1]}-{b.split()[-1]}" for a, b in pairs],
                drop_refs=True, 
                verbose=False
            )
            
            # Resamplear si es necesario
            if int(raw_bip.info['sfreq']) != 256:
                raw_bip.resample(256, npad='auto', verbose=False)
            
            # Obtener datos como tensor
            data = raw_bip.get_data()  # (canales, muestras)
            raw_bip.close()
            
            # Limpiar memoria
            del raw, raw_bip
            gc.collect()
            
            # Convertir a tensor y transpose si es necesario
            tensor = torch.from_numpy(data.astype(np.float32))
            if self.transpose:
                tensor = tensor.T  # (muestras, canales)
            
            return tensor
            
        except Exception as e:
            print(f"❌ Error procesando {edf_path}: {e}")
            return None
    
    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Aplicar preprocesamiento optimizado"""
        if self.preprocess_config.get('bandpass'):
            bp = self.preprocess_config['bandpass']
            raw.filter(bp[0], bp[1], method='iir', phase='zero', verbose=False)
            
        if self.preprocess_config.get('notch'):
            raw.notch_filter(freqs=self.preprocess_config['notch'], 
                           method='iir', phase='zero', verbose=False)
        
        if self.preprocess_config.get('normalize', True):
            data = raw.get_data()
            mean_vals = data.mean(axis=1, keepdims=True)
            std_vals = data.std(axis=1, keepdims=True)
            data = (data - mean_vals) / (std_vals + 1e-6)
            raw._data = data
            
        return raw
    
    def _create_specs(self) -> List[EEGWindowSpec]:
        """Crear especificaciones de ventanas de forma optimizada"""
        # Obtener archivos
        edf_files = self._get_edf_files()
        
        if self.limits.get('files', 0) > 0:
            edf_files = edf_files[:self.limits['files']]
        
        print(f"📁 Procesando {len(edf_files)} archivos...")
        
        specs = []
        for i, edf_path in enumerate(edf_files):
            print(f"⏳ Archivo {i+1}/{len(edf_files)}: {os.path.basename(edf_path)}")
            
            # Encontrar archivo CSV correspondiente
            csv_path = self._find_csv_file(edf_path)
            if not csv_path or not os.path.exists(csv_path):
                print(f"⚠️  CSV no encontrado para {edf_path}")
                continue
            
            # Crear ventanas para este archivo
            file_specs = self._create_file_windows(edf_path, csv_path)
            specs.extend(file_specs)
            
            # Aplicar límite de ventanas si se especifica
            if self.limits.get('max_windows', 0) > 0:
                if len(specs) >= self.limits['max_windows']:
                    specs = specs[:self.limits['max_windows']]
                    break
        
        return specs
    
    def _create_file_windows(self, edf_path: str, csv_path: str) -> List[EEGWindowSpec]:
        """Crear ventanas para un archivo específico"""
        try:
            # Cargar archivo procesado (usa cache LRU)
            processed_data = self._load_processed_file(edf_path)
            if processed_data is None:
                return []
            
            # Cargar etiquetas - saltear comentarios que empiezan con #
            try:
                labels_df = pd.read_csv(csv_path, comment='#')
            except Exception as e:
                print(f"❌ Error leyendo CSV {csv_path}: {e}")
                return []
            
            # Determinar número total de muestras
            if self.transpose:
                total_samples = processed_data.shape[0]  # (muestras, canales)
            else:
                total_samples = processed_data.shape[1]  # (canales, muestras)
            
            # Crear ventanas deslizantes
            specs = []
            start_idx = 0
            
            while start_idx + self.window_samples <= total_samples:
                # Calcular tiempo de inicio y fin en segundos
                start_sec = start_idx / 256.0
                end_sec = (start_idx + self.window_samples) / 256.0
                
                # Obtener etiquetas de convulsión para esta ventana
                # Filtrar solo etiquetas 'seiz' del canal 'TERM' (etiquetas temporales globales)
                seizure_labels = labels_df[
                    (labels_df['start_time'] < end_sec) & 
                    (labels_df['stop_time'] > start_sec) &
                    (labels_df['label'] == 'seiz') &
                    (labels_df['channel'] == 'TERM')
                ]
                
                # Crear vector de etiquetas frame-by-frame
                label_vector = np.zeros(self.window_samples, dtype=np.float32)
                for _, row in seizure_labels.iterrows():
                    label_start = max(0, int((row['start_time'] - start_sec) * 256))
                    label_end = min(self.window_samples, int((row['stop_time'] - start_sec) * 256))
                    if label_start < label_end:
                        label_vector[label_start:label_end] = 1
                
                # Etiqueta de ventana (contiene algún frame positivo)
                window_label = int(label_vector.sum() > 0)
                
                # Crear especificación
                spec = EEGWindowSpec(
                    edf_path=edf_path,
                    csv_path=csv_path,
                    start_idx=start_idx,
                    n_tp=self.window_samples,
                    label=window_label,
                    label_vector=label_vector
                )
                
                specs.append(spec)
                start_idx += self.hop_samples
            
            return specs
            
        except Exception as e:
            print(f"❌ Error creando ventanas para {edf_path}: {e}")
            return []
    
    def _get_edf_files(self) -> List[str]:
        """Obtener lista de archivos EDF para el split"""
        pattern = os.path.join(self.data_dir, 'edf', self.split, '**', '*.edf')
        files = glob.glob(pattern, recursive=True)
        return sorted(files)
    
    def _find_csv_file(self, edf_path: str) -> Optional[str]:
        """Encontrar archivo CSV correspondiente a un EDF"""
        # PRIORIZAR archivos _bi.csv que contienen etiquetas de convulsión
        csv_bi_path = edf_path.replace('.edf', '_bi.csv')
        if os.path.exists(csv_bi_path):
            return csv_bi_path
            
        # Como fallback, usar el archivo .csv regular (solo background)
        csv_path = edf_path.replace('.edf', '.csv')
        if os.path.exists(csv_path):
            return csv_path
            
        return None
    
    def __len__(self) -> int:
        return len(self.specs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Método optimizado que carga ventanas desde archivo procesado en cache.
        Ya no necesita procesar el archivo completo en cada llamada.
        """
        spec = self.specs[idx]
        
        # Cargar archivo procesado desde cache
        processed_data = self._load_processed_file(spec.edf_path)
        if processed_data is None:
            raise RuntimeError(f"No se pudo cargar archivo procesado: {spec.edf_path}")
        
        # Extraer ventana específica
        if self.transpose:
            # (muestras, canales)
            signal = processed_data[spec.start_idx:spec.start_idx + spec.n_tp]
        else:
            # (canales, muestras)  
            signal = processed_data[:, spec.start_idx:spec.start_idx + spec.n_tp]
            signal = signal.T  # Convertir a (muestras, canales)
        
        # Asegurar que tenemos el número correcto de canales
        if signal.shape[-1] != self.target_channels:
            # Rellenar o truncar canales si es necesario
            if signal.shape[-1] < self.target_channels:
                padding = torch.zeros(signal.shape[0], self.target_channels - signal.shape[-1])
                signal = torch.cat([signal, padding], dim=-1)
            else:
                signal = signal[:, :self.target_channels]
        
        # Preparar etiquetas
        if self.time_step:
            labels = torch.from_numpy(spec.label_vector.astype(np.float32))
            if self.one_hot:
                labels_oh = torch.zeros(len(labels), self.num_classes)
                labels_oh[torch.arange(len(labels)), labels.long()] = 1
                labels = labels_oh
        else:
            if self.one_hot:
                labels = torch.zeros(self.num_classes)
                labels[spec.label] = 1
            else:
                labels = torch.tensor(spec.label, dtype=torch.float32)
        
        return signal, labels, spec.label
    
    def _print_stats(self):
        """Imprimir estadísticas del dataset"""
        if not self.specs:
            return
            
        total_windows = len(self.specs)
        positive_windows = sum(1 for spec in self.specs if spec.label == 1)
        negative_windows = total_windows - positive_windows
        
        total_frames = sum(len(spec.label_vector) for spec in self.specs)
        positive_frames = sum(spec.label_vector.sum() for spec in self.specs)
        
        print(f"📊 ESTADÍSTICAS OPTIMIZADAS:")
        print(f"├── Total ventanas: {total_windows}")
        print(f"├── Ventanas positivas: {positive_windows} ({positive_windows/total_windows*100:.1f}%)")
        print(f"├── Ventanas negativas: {negative_windows} ({negative_windows/total_windows*100:.1f}%)")
        print(f"├── Total frames: {total_frames:,}")
        print(f"├── Frames positivos: {positive_frames:.0f} ({positive_frames/total_frames*100:.3f}%)")
        print(f"└── Cache hits: {len(self._file_cache)} archivos en memoria")
    
    def clear_cache(self):
        """Limpiar cache de memoria"""
        self._file_cache.clear()
        self._load_processed_file.cache_clear()
        gc.collect()
        print("🧹 Cache limpiado")
    
    def get_info(self) -> Dict:
        """Información del dataset"""
        return {
            'montage': self.montage,
            'target_channels': self.target_channels,
            'window_sec': self.window_sec,
            'hop_sec': self.hop_sec,
            'num_windows': len(self.specs),
            'cache_enabled': self.use_cache,
            'cache_hits': len(self._file_cache)
        }

# ====================================================================================
# UTILIDADES DE OPTIMIZACIÓN
# ====================================================================================

class MemoryTracker:
    """Rastreador de uso de memoria para cache inteligente"""
    
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.current_bytes = 0
        
    def can_cache(self, size_bytes: int) -> bool:
        """Verifica si se puede cachear un objeto"""
        available_memory = psutil.virtual_memory().available
        return (self.current_bytes + size_bytes < self.max_bytes and 
                size_bytes < available_memory * 0.1)  # Max 10% de RAM disponible
    
    def add_cached(self, size_bytes: int):
        """Registra objeto cacheado"""
        self.current_bytes += size_bytes
    
    def remove_cached(self, size_bytes: int):
        """Quita objeto del cache"""
        self.current_bytes = max(0, self.current_bytes - size_bytes)
