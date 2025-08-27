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
    Dataset EEG optimizado para alta performance con:
    - Cache inteligente en disco y memoria
    - Lazy loading de señales
    - Pre-computación de índices
    - Multiprocesamiento para carga
    - Memory mapping de archivos
    - Limpieza automática de memoria
    - Batching optimizado
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
                 preprocess_config: Optional[Dict] = None,
                 seed: int = 42,
                 # Nuevos parámetros de optimización
                 cache_dir: Optional[str] = None,
                 cache_signals: bool = True,
                 lazy_loading: bool = True,
                 max_memory_gb: float = 8.0,
                 num_workers: int = 4,
                 prefetch_factor: int = 2,
                 pin_memory: bool = True,
                 mmap_mode: bool = True):
        """
        Args optimizados para performance:
            cache_dir: Directorio para cache persistente (None = auto)
            cache_signals: Si cachear señales preprocessadas
            lazy_loading: Si cargar señales bajo demanda
            max_memory_gb: Límite de memoria RAM en GB
            num_workers: Trabajadores para carga paralela
            prefetch_factor: Factor de pre-carga de batches
            pin_memory: Pin memory para GPU transfer
            mmap_mode: Usar memory mapping para archivos grandes
        """
        
        # Configuración básica
        self.data_dir = Path(data_dir)
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
        self.preprocess_config = preprocess_config or {}
        self.seed = seed
        
        # Configuración de optimización
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir / '.cache'
        self.cache_signals = cache_signals
        self.lazy_loading = lazy_loading
        self.max_memory_bytes = max_memory_gb * 1024**3
        self.num_workers = min(num_workers, os.cpu_count() or 4)
        self.prefetch_factor = prefetch_factor
        self.pin_memory = pin_memory
        self.mmap_mode = mmap_mode
        
        # Configurar montajes
        if isinstance(montage, str):
            self.montages = [montage]
        else:
            self.montages = list(montage)
        
        # Estado interno optimizado
        self.rng = np.random.default_rng(seed)
        self._signal_cache = {}
        self._memory_tracker = MemoryTracker(self.max_memory_bytes)
        self._file_handles = {}
        
        # Crear directorio de cache
        self.cache_dir.mkdir(exist_ok=True)
        
        # Inicializar
        self._print_init_info()
        self._validate_montages()
        self._build_optimized_dataset()
        
    def _print_init_info(self):
        """Información de inicialización optimizada"""
        print("🚀 Inicializando OptimizedEEGDataset:")
        print(f"├── Split: {self.split}")
        print(f"├── Montaje(s): {', '.join(self.montages)}")
        print(f"├── Canales objetivo: {self.target_channels}")
        print(f"├── Ventana: {self.window_sec}s / Hop: {self.hop_sec}s")
        print(f"├── Cache: {self.cache_signals} | Lazy: {self.lazy_loading}")
        print(f"├── Memoria máx: {self.max_memory_bytes/1024**3:.1f}GB")
        print(f"├── Workers: {self.num_workers} | MMap: {self.mmap_mode}")
        print(f"└── Cache dir: {self.cache_dir}")
    
    def _validate_montages(self):
        """Validación optimizada de montajes"""
        supported = {'ar', 'ar_a', 'le'}
        invalid = set(self.montages) - supported
        if invalid:
            raise ValueError(f"Montajes no soportados: {invalid}. Disponibles: {supported}")
    
    def _build_optimized_dataset(self):
        """Construcción optimizada del dataset"""
        # 1. Verificar cache existente
        cache_key = self._generate_cache_key()
        cache_file = self.cache_dir / f"dataset_{cache_key}.pkl"
        
        if cache_file.exists():
            print("📂 Cargando dataset desde cache...")
            try:
                self.specs = self._load_from_cache(cache_file)
                self._print_final_stats_optimized(len(self.specs))
                return
            except Exception as e:
                print(f"⚠️  Error cargando cache: {e}. Regenerando...")
        
        # 2. Construcción optimizada
        print("🔧 Construyendo dataset optimizado...")
        all_specs = self._build_specs_parallel()
        
        # 3. Aplicar límites y balanceo
        all_specs = self._apply_optimizations(all_specs)
        
        # 4. Guardar en cache
        if self.cache_signals:
            self._save_to_cache(all_specs, cache_file)
        
        self.specs = all_specs
        self._print_final_stats_optimized(len(all_specs))
    
    def _generate_cache_key(self) -> str:
        """Genera clave única para cache - VERSIÓN COMPLETA"""
        key_data = {
            'montages': sorted(self.montages),
            'split': self.split,
            'window_sec': self.window_sec,
            'hop_sec': self.hop_sec,
            'target_channels': self.target_channels,
            'time_step': self.time_step,
            'one_hot': self.one_hot,
            'num_classes': self.num_classes,
            'transpose': self.transpose,
            'limits': self.limits,
            'balance_pos_frac': self.balance_pos_frac,
            'preprocess': self.preprocess_config,
            'seed': self.seed
        }
        key_str = str(sorted(key_data.items()))
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def _build_specs_parallel(self) -> List[EEGWindowSpec]:
        """Construcción paralela de especificaciones - CORREGIDA"""
        all_specs = []
        
        for montage in self.montages:
            print(f"\n📊 Procesando montaje {montage.upper()}...")
            
            # Obtener archivos
            csv_files = self._get_csv_files(montage)
            if not csv_files:
                continue
            
            # Aplicar límites
            max_files = self.limits.get('files', 0)
            if max_files > 0:
                csv_files = csv_files[:max_files]
            
            print(f"   📁 Procesando {len(csv_files)} archivo(s)")
            
            # Procesar archivos secuencialmente para mantener consistencia
            # (el multiprocesamiento causaba inconsistencias en parámetros)
            montage_specs = []
            processed = 0
            
            for csv_path in csv_files:
                try:
                    file_specs = self._process_file_optimized(csv_path, montage)
                    montage_specs.extend(file_specs)
                    processed += 1
                    
                    # Progreso cada 10 archivos
                    if processed % 10 == 0:
                        print(f"   Progreso: {processed}/{len(csv_files)} archivos procesados")
                        
                except Exception as e:
                    print(f"   Error procesando {os.path.basename(csv_path)}: {e}")
                    continue
            
            print(f"   ✓ Procesamiento completado: {processed}/{len(csv_files)} archivos")
            print(f"   ✅ {montage.upper()}: {len(montage_specs)} ventanas")
            all_specs.extend(montage_specs)
        
        return all_specs
    
    def _extract_montage_signals_for_processing(self, edf_path: str, montage: str) -> Optional[mne.io.BaseRaw]:
        """Extrae señales para procesamiento (igual que EEGWindowDataset)"""
        try:
            # Cargar archivo EDF
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # Aplicar preprocesamiento
            raw = self._preprocess_raw_fast(raw)
            
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
    
    def _process_file_optimized(self, csv_path: str, montage: str) -> List[EEGWindowSpec]:
        """Procesa un archivo de forma optimizada"""
        try:
            edf_path = csv_path.replace('_bi.csv', '.edf')
            if not os.path.exists(edf_path):
                return []
            
            # Cargar y verificar señal igual que en EEGWindowDataset
            raw_bip = self._extract_montage_signals_for_processing(edf_path, montage)
            if raw_bip is None:
                return []
            
            # Usar parámetros de instancia, NO hardcodeados
            sf = float(raw_bip.info['sfreq'])
            n_tp = int(round(self.window_sec * sf))
            hop_tp = int(round(self.hop_sec * sf)) if self.hop_sec else n_tp
            n_tot = int(raw_bip.n_times)
            
            # Cargar anotaciones igual que en EEGWindowDataset
            seiz_intervals = self._load_annotations(csv_path)
            detailed_labels = self._create_detailed_labels(seiz_intervals, n_tot, sf)
            
            # Generar ventanas igual que en EEGWindowDataset
            specs = []
            for start in range(0, n_tot - n_tp + 1, hop_tp):
                end = start + n_tp
                
                # Etiqueta binaria de ventana
                window_label = int(detailed_labels[start:end].max() > 0)
                
                # Vector detallado
                window_label_vector = detailed_labels[start:end].copy()
                
                # Crear especificación
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
            
            raw_bip.close()
            return specs
            
        except Exception as e:
            print(f"Error procesando {os.path.basename(csv_path)}: {e}")
            return []
    
    def _load_annotations(self, csv_path: str) -> List[Tuple[float, float]]:
        """Carga anotaciones de convulsiones - VERSIÓN CONSISTENTE"""
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
        """Crea vector detallado de etiquetas frame-by-frame - VERSIÓN CONSISTENTE"""
        labels = np.zeros(n_samples, dtype=np.int32)
        
        for start_time, stop_time in seiz_intervals:
            start_idx = max(0, int(start_time * sf))
            stop_idx = min(n_samples, int(stop_time * sf))
            labels[start_idx:stop_idx] = 1
            
        return labels
    
    def _apply_optimizations(self, specs: List[EEGWindowSpec]) -> List[EEGWindowSpec]:
        """Aplica optimizaciones de balanceo y límites"""
        # Límites de ventanas
        max_windows = self.limits.get('max_windows', 0)
        if max_windows > 0 and len(specs) > max_windows:
            specs = self._apply_window_limits_fast(specs, max_windows)
        
        # Balanceo
        if self.balance_pos_frac is not None:
            specs = self._apply_balancing_fast(specs)
        
        return specs
    
    def _apply_window_limits_fast(self, specs: List[EEGWindowSpec], max_windows: int) -> List[EEGWindowSpec]:
        """Aplicación rápida de límites con numpy"""
        if len(specs) <= max_windows:
            return specs
        
        # Separar usando list comprehension optimizada
        labels = np.array([s.label for s in specs])
        pos_mask = labels == 1
        
        pos_indices = np.where(pos_mask)[0]
        neg_indices = np.where(~pos_mask)[0]
        
        # Mantener proporción
        pos_ratio = len(pos_indices) / len(specs)
        n_pos = min(len(pos_indices), int(max_windows * pos_ratio))
        n_neg = max_windows - n_pos
        
        # Sampling optimizado
        selected_pos = self.rng.choice(pos_indices, size=n_pos, replace=False) if n_pos > 0 else []
        selected_neg = self.rng.choice(neg_indices, size=min(n_neg, len(neg_indices)), replace=False) if n_neg > 0 else []
        
        selected_indices = np.concatenate([selected_pos, selected_neg])
        self.rng.shuffle(selected_indices)
        
        return [specs[i] for i in selected_indices]
    
    def _apply_balancing_fast(self, specs: List[EEGWindowSpec]) -> List[EEGWindowSpec]:
        """Balanceo rápido optimizado"""
        labels = np.array([s.label for s in specs])
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]
        
        if len(pos_indices) == 0 or len(neg_indices) == 0:
            return specs
        
        total_target = len(specs)
        pos_target = int(total_target * self.balance_pos_frac)
        neg_target = total_target - pos_target
        
        # Sampling con reemplazo optimizado
        selected_pos = self.rng.choice(pos_indices, size=pos_target, replace=len(pos_indices) < pos_target)
        selected_neg = self.rng.choice(neg_indices, size=neg_target, replace=len(neg_indices) < neg_target)
        
        selected_indices = np.concatenate([selected_pos, selected_neg])
        self.rng.shuffle(selected_indices)
        
        result = [specs[i] for i in selected_indices]
        print(f"✓ Balanceo aplicado: {pos_target} pos + {neg_target} neg = {len(result)} ventanas")
        
        return result
    
    def _save_to_cache(self, specs: List[EEGWindowSpec], cache_file: Path):
        """Guarda dataset en cache optimizado"""
        try:
            print(f"💾 Guardando cache: {len(specs)} specs...")
            with open(cache_file, 'wb') as f:
                pickle.dump(specs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"✅ Cache guardado: {cache_file.name}")
        except Exception as e:
            print(f"⚠️  Error guardando cache: {e}")
    
    def _load_from_cache(self, cache_file: Path) -> List[EEGWindowSpec]:
        """Carga dataset desde cache"""
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    def _get_csv_files(self, montage: str) -> List[str]:
        """Obtención optimizada de archivos CSV"""
        root = self.data_dir / 'edf' / self.split
        
        # Usar pathlib para mejor performance
        if montage == 'ar':
            pattern = '**/s*/*_tcp_ar/*_bi.csv'
        elif montage == 'ar_a':
            pattern = '**/s*/*_tcp_ar_a/*_bi.csv'
        elif montage == 'le':
            pattern = '**/s*/*_tcp_le/*_bi.csv'
        else:
            return []
        
        files = list(root.glob(pattern))
        return [str(f) for f in sorted(files)]
    
    def __len__(self) -> int:
        return len(self.specs)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Obtención optimizada de muestras con lazy loading"""
        spec = self.specs[idx]
        
        # Lazy loading de señal
        if self.lazy_loading:
            signal = self._load_signal_lazy(spec)
        else:
            signal = self._load_signal_cached(spec)
        
        # Preparar etiquetas (optimizado)
        labels = self._prepare_labels_fast(spec)
        
        return signal, labels, spec.label
    
    def _load_signal_lazy(self, spec: EEGWindowSpec) -> torch.Tensor:
        """Carga lazy optimizada de señales"""
        montage = getattr(spec, 'montage', self.montages[0])
        
        # Verificar cache en memoria
        cache_key = f"{spec.edf_path}_{spec.start_idx}_{montage}"
        if cache_key in self._signal_cache:
            return self._signal_cache[cache_key]
        
        # Cargar y procesar
        try:
            raw_bip = self._extract_montage_signals_optimized(spec.edf_path, montage)
            if raw_bip is None:
                raise RuntimeError(f"Error cargando {spec.edf_path}")
            
            # Extraer ventana específica
            data = raw_bip.get_data()[:, spec.start_idx:spec.start_idx + spec.n_tp]
            raw_bip.close()
            
            # Convertir a tensor optimizado
            signal = torch.from_numpy(data.astype(np.float32))
            
            # Aplicar transformaciones
            if self.transpose:
                signal = signal.T
            
            # Ajustar canales
            original_channels = getattr(spec, 'original_channels', signal.shape[-1 if self.transpose else 0])
            signal = self._convert_channels_fast(signal, original_channels)
            
            # Cache inteligente (verificar memoria)
            if self._memory_tracker.can_cache(signal.nbytes):
                self._signal_cache[cache_key] = signal
                self._memory_tracker.add_cached(signal.nbytes)
            
            return signal
            
        except Exception as e:
            raise RuntimeError(f"Error cargando señal: {e}")
    
    def _load_signal_cached(self, spec: EEGWindowSpec) -> torch.Tensor:
        """Carga con cache persistente en disco"""
        # Implementación de cache en disco para señales preprocessadas
        pass
    
    @lru_cache(maxsize=32)
    def _extract_montage_signals_optimized(self, edf_path: str, montage: str) -> Optional[mne.io.BaseRaw]:
        """Extracción optimizada con cache LRU"""
        try:
            # Memory mapping si está habilitado
            if self.mmap_mode and edf_path not in self._file_handles:
                self._file_handles[edf_path] = open(edf_path, 'rb')
            
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # Preprocesamiento optimizado
            raw = self._preprocess_raw_fast(raw)
            
            # Montaje bipolar optimizado
            suf = '-LE' if any(ch.endswith('-LE') for ch in raw.ch_names) else '-REF'
            pairs = get_montage_pairs_ordered(montage, suf)
            
            raw_bip = mne.set_bipolar_reference(
                raw,
                anode=[a for a, _ in pairs],
                cathode=[b for _, b in pairs],
                ch_name=[f"{a.split()[-1]}-{b.split()[-1]}" for a, b in pairs],
                drop_refs=True, verbose=False
            )
            
            if int(raw_bip.info['sfreq']) != 256:
                raw_bip.resample(256, npad='auto', verbose=False)
            
            return raw_bip
            
        except Exception:
            return None
    
    def _preprocess_raw_fast(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """Preprocesamiento optimizado pero consistente"""
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
    
    def _convert_channels_fast(self, signal: torch.Tensor, original_channels: int) -> torch.Tensor:
        """Conversión rápida de canales"""
        if original_channels == self.target_channels:
            return signal
        
        if self.transpose:  # (tiempo, canales)
            current_channels = signal.shape[1]
            if current_channels < self.target_channels:
                padding = torch.zeros(signal.shape[0], self.target_channels - current_channels, 
                                    dtype=signal.dtype, device=signal.device)
                return torch.cat([signal, padding], dim=1)
            else:
                return signal[:, :self.target_channels]
        else:  # (canales, tiempo)
            current_channels = signal.shape[0]
            if current_channels < self.target_channels:
                padding = torch.zeros(self.target_channels - current_channels, signal.shape[1], 
                                    dtype=signal.dtype, device=signal.device)
                return torch.cat([signal, padding], dim=0)
            else:
                return signal[:self.target_channels, :]
    
    def _prepare_labels_fast(self, spec: EEGWindowSpec) -> torch.Tensor:
        """Preparación rápida de etiquetas"""
        if self.time_step:
            labels = torch.from_numpy(spec.label_vector.astype(np.float32))
            if self.one_hot:
                labels_oh = torch.zeros(len(labels), self.num_classes, dtype=torch.float32)
                labels_oh[torch.arange(len(labels)), labels.long()] = 1
                return labels_oh
            return labels
        else:
            if self.one_hot:
                labels = torch.zeros(self.num_classes, dtype=torch.float32)
                labels[spec.label] = 1
                return labels
            return torch.tensor(spec.label, dtype=torch.float32)
    
    def _print_final_stats_optimized(self, total_windows: int):
        """Estadísticas finales optimizadas"""
        if total_windows == 0:
            print("\n⚠️  DATASET VACÍO")
            return
        
        # Calcular estadísticas de forma vectorizada
        labels = np.array([s.label for s in self.specs])
        positive_windows = np.sum(labels)
        
        print(f"\n📊 DATASET OPTIMIZADO COMPLETADO")
        print(f"├── Total ventanas: {total_windows:,}")
        print(f"├── Ventanas positivas: {positive_windows:,} ({100*positive_windows/total_windows:.1f}%)")
        print(f"├── Memoria cache: {len(self._signal_cache)} señales")
        print(f"├── Workers usados: {self.num_workers}")
        print(f"└── Cache dir: {self.cache_dir}")
    
    def cleanup(self):
        """Limpieza de recursos"""
        self._signal_cache.clear()
        for handle in self._file_handles.values():
            handle.close()
        self._file_handles.clear()
        gc.collect()
    
    def __del__(self):
        """Destructor para limpieza automática"""
        try:
            self.cleanup()
        except:
            pass

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
