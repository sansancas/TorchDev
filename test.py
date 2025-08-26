import torch
from dataset import EEGWindowDataset, OptimizedEEGDataset
from models.TCN import create_seizure_tcn
from train import TrainConfig, run_training
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import torch.multiprocessing as mp
import os

DATA_DIR = 'DATA_EEG_TUH/tuh_eeg_seizure/v2.0.3/'
MONTAGE = 'ar'
WINDOW_SEC = 10
HOP_SEC = 0.25
TIME_STEP = True
ONEHOT = False
BATCH_SIZE = 32
NUM_CLASSES = 2 if ONEHOT else 1
BALANCE_POS_FRAC = 0
TRANSPOSE = True
PREPROCESS = {
    'bandpass': (0.5, 40.0),  # Hz
    'notch': 60.0,            # Hz
    'resample': 256,          # Hz
}
SEED = 42
LIMITS_TRAIN = {'files': 5, 'max_windows': 0}
LIMITS_VAL = {'files': 5, 'max_windows': 0}

def main():
    train_ds = EEGWindowDataset(
        data_dir=DATA_DIR,
        split='dev',# train
        montage=MONTAGE,
        window_sec=WINDOW_SEC,
        hop_sec=HOP_SEC,
        time_step=TIME_STEP,
        one_hot=ONEHOT,
        num_classes=NUM_CLASSES,
        transpose=TRANSPOSE,
        limits=LIMITS_TRAIN,
        balance_pos_frac=BALANCE_POS_FRAC,  # SIN BALANCEO
        write_manifest=False,
        preprocess_config=PREPROCESS,
        seed=SEED
    )
    val_ds   = EEGWindowDataset(
        data_dir=DATA_DIR,
        split='dev',# train
        montage=MONTAGE,
        window_sec=WINDOW_SEC,
        hop_sec=HOP_SEC,
        time_step=TIME_STEP,
        one_hot=ONEHOT,
        num_classes=NUM_CLASSES,
        transpose=TRANSPOSE,
        limits=LIMITS_TRAIN,
        balance_pos_frac=BALANCE_POS_FRAC,  # SIN BALANCEO
        write_manifest=False,
        preprocess_config=PREPROCESS,
        seed=SEED
    )

    model = create_seizure_tcn(input_channels=train_ds.target_channels, time_step=TIME_STEP, one_hot=ONEHOT)

    try:
        mp.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy('file_system')
    except RuntimeError:
        pass

    cfg = TrainConfig(
        epochs=3,
        batch_size=32,
        lr=3e-4, weight_decay=1e-2,
        pos_weight_auto=True,
        use_weighted_sampler=True,
        use_cosine_warm_restarts=True, T_0=10, T_mult=2,
        decision_threshold=0.5,
        monitor_metric='val_window_balanced_accuracy',
        window_agg='max',
        metrics_backend='torchmetrics',     # <- usa torcheval/torchmetrics si están disponibles
        compile_model=False,
        num_workers=2, persistent_workers=False,
        progress_updates=20, show_progress=True,
    )
    best, hist = run_training(model, train_ds, val_ds, cfg)


if __name__ == "__main__":
    main()