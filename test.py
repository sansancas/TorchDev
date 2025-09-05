import torch
from dataset import OptimizedEEGDataset
from models.TCN import create_seizure_tcn
from models.Hybrid import create_hybrid_seizure_model
from models.Transformer import create_deep_transformer, create_transformer_seizure_model
from train import TrainConfig, run_training
import torch.multiprocessing as mp
import datetime

DATA_DIR = 'DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/'
MONTAGE = 'ar'
WINDOW_SEC = 5
TIME_STEP = True
ONEHOT = False
NUM_CLASSES = 2 if ONEHOT else 1
BALANCE_POS_FRAC = 0
TRANSPOSE = True
PREPROCESS = {
    'bandpass': (0.5, 40.0),
    'notch': 60.0,
    'resample': 256,
}
TIME_LIMIT = 48
SEED = 42
MODEL = 'HYB'

# ============================================================================
# CONFIGURACIONES PRESETS PARA DIFERENTES CASOS DE USO
# ============================================================================

CONFIGS = {
    'balanced': {
        'name': 'Balanceado',
        'hop_sec': 3.0,  # Cambiar de 3.0s a 5.0s - reduce ventanas a la mitad
        'batch_size': 8,  # Aumentar batch size para eficiencia
        'epochs': 100,
        'limits_train': {'files': 250, 'max_windows': 0},  # Reducir de 200 a 30 archivos
        'limits_val': {'files': 75, 'max_windows': 0},   # Reducir de 75 a 15 archivos
        'time_limit_sec': TIME_LIMIT*3600,  
        'description': 'Balance entre velocidad y precisión - OPTIMIZADO',
        
    },
}

def run_training_config(config_name: str):
    """Ejecutar entrenamiento con una configuración específica"""
    
    if config_name not in CONFIGS:
        print(f"❌ Configuración '{config_name}' no encontrada")
        print(f"Disponibles: {list(CONFIGS.keys())}")
        return
    
    config = CONFIGS[config_name]
    
    print(f"🚀 ENTRENAMIENTO: {config['name'].upper()}")
    print("=" * 70)
    print(f"📋 DESCRIPCIÓN: {config['description']}")
    print(f"⚙️  PARÁMETROS:")
    print(f"├── Hop: {config['hop_sec']}s")
    print(f"├── Batch size: {config['batch_size']}")
    print(f"├── Épocas: {config['epochs']}")
    print(f"├── Archivos train: {config['limits_train']['files'] or 'TODOS'}")
    print(f"└── Archivos val: {config['limits_val']['files'] or 'TODOS'}")
    
    # Configurar multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass
    
    # Crear datasets
    print(f"\n📊 Creando datasets...")
    
    train_ds = OptimizedEEGDataset(
        data_dir=DATA_DIR,
        split='train',
        montage=MONTAGE,
        window_sec=WINDOW_SEC,
        hop_sec=config['hop_sec'],
        time_step=TIME_STEP,
        one_hot=ONEHOT,
        num_classes=NUM_CLASSES,
        transpose=TRANSPOSE,
        limits=config['limits_train'],
        balance_pos_frac=BALANCE_POS_FRAC,
        preprocess_config=PREPROCESS,
        use_cache=True,
        cache_dir=f"cache_train_{config_name}",
        seed=SEED
    )
    
    val_ds = OptimizedEEGDataset(
        data_dir=DATA_DIR,
        split='dev',
        montage=MONTAGE,
        window_sec=WINDOW_SEC,
        hop_sec=config['hop_sec'],
        time_step=TIME_STEP,
        one_hot=ONEHOT,
        num_classes=NUM_CLASSES,
        transpose=TRANSPOSE,
        limits=config['limits_val'],
        balance_pos_frac=BALANCE_POS_FRAC,
        preprocess_config=PREPROCESS,
        use_cache=True,
        cache_dir=f"cache_val_{config_name}",
        seed=SEED
    )
    
    # Calcular estimación de speedup vs original
    original_estimate = len(train_ds) * (config['hop_sec'] / 0.25)
    speedup = original_estimate / len(train_ds) if len(train_ds) > 0 else 1
    
    print(f"\n📈 ESTADÍSTICAS:")
    print(f"├── Train: {len(train_ds):,} ventanas")
    print(f"├── Val: {len(val_ds):,} ventanas")
    print(f"├── Estimado vs hop=0.25s: {original_estimate:,.0f} ventanas")
    print(f"├── Speedup: {speedup:.1f}x más rápido")
    print(f"├── Solapamiento: {((WINDOW_SEC - config['hop_sec']) / WINDOW_SEC * 100):.1f}%")
    print(f"└── GPU: {torch.cuda.is_available()}")
    
    model_args = {'time_step': True,
                  'one_hot': True,
                'num_filters': 64,
                'kernel_size': 7,
                'num_blocks': 8,
                'time_step': True,
                'one_hot': True,
                'dropout': 0.25,
                'use_se': True,
                'use_multiscale': False,
                'class_weights': [1.0, 5.0]
            }

    # Crear modelo
    if MODEL == 'HYB':
        model_args = {
            'dropout_rate': 0.2,  # Mayor dropout para regularización
            'one_hot': ONEHOT,
            'time_step': TIME_STEP,
            'se_position': 'after_conv',  # SE después de conv para mejor extracción
            'attention_position': 'final',  # Atención final para capturar patrones críticos
            'rnn_type': 'lstm',  # LSTM mejor para secuencias largas EEG
            'se_ratio': 12,  # SE más fuerte para EEG
            'num_heads': 8,  # Más cabezas de atención
            'pool_size_if_ts': 1,  # Sin downsampling temporal
            'pool_size_if_win': 2,
            'class_weights': [1.0, 20.0],  # Peso muy alto para convulsiones
            'use_layer_norm': True,  # LayerNorm mejor para EEG
            'enhanced_rnn': True,  # RNN bidireccional + mayor hidden size
            'kernels': 7
        }
        print('hybrid model')
        model = create_hybrid_seizure_model(input_channels=train_ds.target_channels,
                                            **model_args)
    if MODEL == 'TRANS':
        model_args = {
        'num_classes': 1,
        'embed_dim': 256,  # Dimensión moderada para EEG
        'num_layers': 12,   # Suficientes capas para capturar dependencias temporales
        'num_heads': 8,    # Múltiples cabezas para diferentes patrones
        'mlp_dim': 256,    # FFN de tamaño moderado
        'dropout_rate': 0.25,  # Dropout moderado
        'time_step_classification': TIME_STEP,
        'one_hot': ONEHOT,
        'use_se': True,    # SE para enfoque en características importantes
        'se_ratio': 12,     # SE más fuerte
        'class_weights': [1.0, 20.0],  # Peso muy alto para convulsiones
        'use_cls_token': False,  # Sin CLS para time-step classification
        'enhanced_pos_enc': True,
        'use_layer_scale': True
    }
        print('transformer model')
        model = create_transformer_seizure_model(input_channels=train_ds.target_channels,
                                            **model_args)
    else:
        model_args = {
                'num_filters': 256,
                'kernel_size': 7,
                'num_blocks': 7,
                'time_step': TIME_STEP,
                'one_hot': ONEHOT,
                'dropout': 0.25,
                'use_se': True,
                'use_multiscale': False,
                'class_weights': [1.0, 15.0]
            }

        print('convolution model')
        model = create_seizure_tcn(
            input_channels=train_ds.target_channels,
            time_step=TIME_STEP,
            one_hot=ONEHOT
        )
    name = f'{MODEL}_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}'
    # Configuración de entrenamiento
    train_config = TrainConfig(
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        lr=3e-4,
        weight_decay=1e-2,
        pos_weight_auto=True,
        use_weighted_sampler=False,
        use_cosine_warm_restarts=True,
        T_0=10,
        T_mult=2,
        decision_threshold=0.5,
        monitor_metric='val_frame_balanced_accuracy',
        window_agg='max',
        metrics_backend='torchmetrics',
        compile_model=True,
        amp=True,
        min_delta=1e-4,
        num_workers=0,  # Optimizado
        persistent_workers=True,
        pin_memory=True,
        progress_updates=20,
        show_progress=True,
        run_name=f"eeg_torch_{name}",
        time_limit_sec=config['time_limit_sec'],
    )
    
    # Ejecutar entrenamiento
    print(f"\n🏁 INICIANDO ENTRENAMIENTO...")
    print("=" * 70)
    
    try:
        best, hist = run_training(model, train_ds, val_ds, train_config)
        
        print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
        print(f"├── Configuración: {config['name']}")
        print(f"├── Best metric: {best}")
        print(f"├── Épocas: {len(hist)}")
        print(f"└── Speedup logrado: {speedup:.1f}x")
        
        # Limpiar cache
        train_ds.clear_cache()
        val_ds.clear_cache()
        
        return best, hist
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Entrenamiento interrumpido")
        return None, None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise

def main():
    """Menú principal para seleccionar configuración"""
    
    print("🎯 ENTRENAMIENTO EEG CON CONFIGURACIONES OPTIMIZADAS")
    print("=" * 60)
    print("Selecciona una configuración:")
    print()
    
    for key, config in CONFIGS.items():
        hop = config['hop_sec']
        speedup = 0.25 / hop
        overlap = ((WINDOW_SEC - hop) / WINDOW_SEC * 100)
        
        print(f"{key:>15}: {config['name']}")
        print(f"{'':>15}  ├── Hop: {hop}s (speedup {speedup:.1f}x)")
        print(f"{'':>15}  ├── Solapamiento: {overlap:.1f}%")
        print(f"{'':>15}  └── {config['description']}")
        print()
    
    # Por defecto usar 'balanced'
    selected = 'balanced'
    print(f"🚀 Ejecutando configuración: {selected}")
    print(f"💡 Para cambiar, modifica la variable 'selected' en main()")
    run_training_config(selected)

if __name__ == "__main__":
    main()
