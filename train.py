from __future__ import annotations
import os
import csv
import json
import math
import time
import random
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except Exception:
    _TB = False

# backends opcionales
_TORCHEVAL_OK = False
_TORCHMETRICS_OK = False
try:
    import torcheval.metrics as TE
    _TORCHEVAL_OK = True
except Exception:
    TE = None  # type: ignore
try:
    import importlib
    TM_cls = importlib.import_module('torchmetrics.classification')
    _TORCHMETRICS_OK = True
except Exception:
    TM_cls = None  # type: ignore

# Acelera GEMMs en Ampere+
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')

# ------------------------------- Utilidades ---------------------------------

def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def to_device(x, device):
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device, non_blocking=True)

# ------------------------------- Config -------------------------------------

@dataclass
class TrainConfig:
    run_root: str = "runs"
    run_name: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float = 1.0
    amp: bool = True
    compile_model: bool = True
    seed: int = 1337

    # DataLoader
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True

    # Pérdidas
    loss_name: str = "auto"  # "auto" | "bce" | "focal" | "ce"
    use_model_loss: bool = False  # True → usa model.compute_loss si existe

    # Desbalanceo (usa sólo uno simultáneamente)
    pos_weight_auto: bool = False
    pos_weight_override: Optional[float] = None
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    use_weighted_sampler: bool = False

    # Umbral
    decision_threshold: float = 0.5

    # Scheduler
    use_cosine_warm_restarts: bool = True
    T_0: int = 10
    T_mult: int = 2
    eta_min: float = 1e-6

    # Early stopping / checkpoints
    monitor_metric: str = "val_balanced_accuracy"  # ó 'val_window_balanced_accuracy' si time_step
    mode: str = "max"
    patience: int = 10
    min_delta: float = 1e-4
    save_last: bool = True

    # Logging
    use_tensorboard: bool = True

    # Progreso
    show_progress: bool = True
    progress_updates: int = 10

    # Agregación por ventana (time_step=True)
    window_agg: str = "max"  # "max" | "mean"

    # Backend de métricas
    metrics_backend: str = 'auto'  # 'manual' | 'torcheval' | 'torchmetrics' | 'auto'

    def __post_init__(self):
        if self.window_agg not in ("max", "mean"):
            raise ValueError("window_agg debe ser 'max' o 'mean'")
        if self.metrics_backend not in ('manual','torcheval','torchmetrics','auto'):
            raise ValueError("metrics_backend inválido")

# ------------------------------ Pérdidas -------------------------------------

class FocalLossWithLogits(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1 - p)
        w = self.alpha * (1 - pt).clamp(min=1e-6).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        loss = w * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# ------------------------------ Métricas -------------------------------------

class MetricGroup:
    """ Métricas binarias en streaming (GPU-friendly). """
    def __init__(self, backend: str, device: str, threshold: float = 0.5):
        self.backend = backend
        self.device = device
        self.threshold = float(threshold)
        # contadores manuales
        self.tp = torch.tensor(0.0, device=device)
        self.tn = torch.tensor(0.0, device=device)
        self.fp = torch.tensor(0.0, device=device)
        self.fn = torch.tensor(0.0, device=device)
        self.brier_sum = torch.tensor(0.0, device=device)
        self.n = torch.tensor(0.0, device=device)
        # backends
        self.m = {}
        if backend == 'torcheval' and _TORCHEVAL_OK:
            try:
                self.m['acc']  = TE.BinaryAccuracy(threshold=self.threshold, device=device)
                self.m['f1']   = TE.BinaryF1Score(threshold=self.threshold, device=device)
                self.m['prec'] = TE.BinaryPrecision(threshold=self.threshold, device=device)
                self.m['rec']  = TE.BinaryRecall(threshold=self.threshold, device=device)
                BAUROC = getattr(TE, 'BinaryAUROC', None)
                BAPRC  = getattr(TE, 'BinaryAUPRC', None)
                if BAUROC: self.m['auroc'] = BAUROC(device=device)
                if BAPRC:  self.m['auprc'] = BAPRC(device=device)
            except Exception:
                pass
        elif backend == 'torchmetrics' and _TORCHMETRICS_OK:
            try:
                BA = getattr(TM_cls, 'BinaryAccuracy')
                BF1 = getattr(TM_cls, 'BinaryF1Score')
                BP = getattr(TM_cls, 'BinaryPrecision')
                BR = getattr(TM_cls, 'BinaryRecall')
                self.m['acc']  = BA(threshold=self.threshold).to(device)
                self.m['f1']   = BF1(threshold=self.threshold).to(device)
                self.m['prec'] = BP(threshold=self.threshold).to(device)
                self.m['rec']  = BR(threshold=self.threshold).to(device)
                BAUROC = getattr(TM_cls, 'BinaryAUROC', None)
                BAP = getattr(TM_cls, 'BinaryAveragePrecision', None)
                if BAUROC: self.m['auroc'] = BAUROC().to(device)
                if BAP:    self.m['auprc'] = BAP().to(device)
            except Exception:
                pass

    def update(self, prob: torch.Tensor, target: torch.Tensor):
        prob = prob.detach()
        target = target.detach().to(prob.device)
        # Brier y n
        self.brier_sum += torch.sum((prob - target.float())**2)
        self.n += target.numel()
        # contadores
        pred = (prob >= self.threshold).to(torch.long)
        t = target.to(torch.long)
        self.tp += torch.sum((pred == 1) & (t == 1)).float()
        self.tn += torch.sum((pred == 0) & (t == 0)).float()
        self.fp += torch.sum((pred == 1) & (t == 0)).float()
        self.fn += torch.sum((pred == 0) & (t == 1)).float()
        # backend
        if self.m:
            try:
                if 'acc' in self.m:  self.m['acc'].update(prob, t)
                if 'f1' in self.m:   self.m['f1'].update(prob, t)
                if 'prec' in self.m: self.m['prec'].update(prob, t)
                if 'rec' in self.m:  self.m['rec'].update(prob, t)
                if 'auroc' in self.m: self.m['auroc'].update(prob, t)
                if 'auprc' in self.m: self.m['auprc'].update(prob, t)
            except Exception:
                pass

    def compute(self) -> Dict[str, float]:
        eps = 1e-12
        tp = float(self.tp.item()); tn = float(self.tn.item())
        fp = float(self.fp.item()); fn = float(self.fn.item())
        n  = max(float(self.n.item()), 1.0)
        acc = (tp + tn) / max(tp + tn + fp + fn, eps)
        tpr = tp / max(tp + fn, eps)
        tnr = tn / max(tn + fp, eps)
        prec = tp / max(tp + fp, eps)
        rec  = tpr
        f1   = 2 * prec * rec / max(prec + rec, eps)
        bacc = 0.5 * (tpr + tnr)
        brier = float(self.brier_sum.item()) / n
        out = {
            'accuracy': acc, 'balanced_accuracy': bacc,
            'precision': prec, 'recall': rec, 'f1': f1,
            'sensitivity': tpr, 'specificity': tnr,
            'recall_class_0': tnr, 'recall_class_1': tpr,
            'brier': brier, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        }
        both = (tp + fn) > 0 and (tn + fp) > 0
        if both and self.m:
            try:
                if 'auroc' in self.m: out['auroc'] = float(self.m['auroc'].compute().item())
                else: out['auroc'] = float('nan')
            except Exception: out['auroc'] = float('nan')
            try:
                if 'auprc' in self.m: out['auprc'] = float(self.m['auprc'].compute().item())
                else: out['auprc'] = float('nan')
            except Exception: out['auprc'] = float('nan')
        else:
            out['auroc'] = float('nan'); out['auprc'] = float('nan')
        return out

    def reset(self):
        self.tp.zero_(); self.tn.zero_(); self.fp.zero_(); self.fn.zero_()
        self.brier_sum.zero_(); self.n.zero_()
        for v in self.m.values():
            try:
                v.reset()
            except Exception:
                pass


def pick_metrics_backend(cfg_backend: str) -> str:
    if cfg_backend in ('manual','torcheval','torchmetrics'):
        if cfg_backend == 'torcheval' and not _TORCHEVAL_OK:
            warnings.warn("torcheval no disponible; usando fallback manual")
            return 'manual'
        if cfg_backend == 'torchmetrics' and not _TORCHMETRICS_OK:
            warnings.warn("torchmetrics no disponible; usando fallback manual")
            return 'manual'
        return cfg_backend
    if _TORCHEVAL_OK: return 'torcheval'
    if _TORCHMETRICS_OK: return 'torchmetrics'
    return 'manual'

# ---------------------- Helpers de forma/targets -----------------------------

def ensure_channel_last(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and x.shape[1] < x.shape[2]:
        return x.permute(0, 2, 1)
    if x.ndim == 2 and x.shape[0] < x.shape[1]:
        return x.transpose(0, 1)
    return x


def logits_to_probs_and_labels(logits: torch.Tensor, one_hot_hint: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (p_pos, pred_label) manejando binario (1-logit) y 2 clases (2-logits)
    según la forma real de logits, ignorando el hint si es necesario.
    """
    if logits.ndim == 3:  # (B,T,*)
        last = logits.size(-1)
        if last == 2:
            probs = torch.softmax(logits, dim=-1)
            pos = probs[..., 1]
            pred = probs.argmax(dim=-1)
            return pos, pred
        elif last == 1:
            pos = torch.sigmoid(logits.squeeze(-1))
            pred = (pos >= 0.5).long()
            return pos, pred
        else:
            raise ValueError(f"logits (B,T,{last}) no soportado (esperado 1 ó 2)")
    elif logits.ndim == 2:  # (B,*)
        last = logits.size(-1)
        if last == 2:
            probs = torch.softmax(logits, dim=-1)
            pos = probs[:, 1]
            pred = probs.argmax(dim=-1)
            return pos, pred
        elif last == 1:
            pos = torch.sigmoid(logits.squeeze(-1))
            pred = (pos >= 0.5).long()
            return pos, pred
        else:
            raise ValueError(f"logits (B,{last}) no soportado (esperado 1 ó 2)")
    else:
        raise ValueError("Forma de logits no soportada")

# ---------------------------- Construcción loss ------------------------------

def compute_pos_weight(train_dataset) -> Optional[torch.Tensor]:
    labels = None
    if hasattr(train_dataset, 'specs') and isinstance(train_dataset.specs, list) and len(train_dataset.specs) > 0:
        try:
            labels = np.array([int(getattr(s, 'label', 0)) for s in train_dataset.specs], dtype=np.int64)
        except Exception:
            labels = None
    if labels is None:
        return None
    counts = np.bincount(labels, minlength=2).astype(float)
    neg, pos = counts[0], counts[1]
    if pos <= 0:
        warnings.warn("No hay positivos; pos_weight no aplicable.")
        return None
    return torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)


def build_loss(cfg: TrainConfig, train_dataset, time_step: bool, one_hot: bool, num_classes: int,
               device: Optional[torch.device] = None) -> nn.Module:
    """Crea el módulo de pérdida y lo mueve al device si corresponde.
    (FIX) Si usa pos_weight, asegura que esté en el **mismo device** que logits.
    """
    if cfg.loss_name == 'auto':
        if one_hot:
            loss = nn.CrossEntropyLoss()
        else:
            pos_weight = None
            if cfg.pos_weight_override is not None:
                pos_weight = torch.tensor([float(cfg.pos_weight_override)], dtype=torch.float32)
            elif cfg.pos_weight_auto:
                pw = compute_pos_weight(train_dataset)
                if pw is not None:
                    pos_weight = pw
            loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif cfg.loss_name == 'bce':
        pos_weight = None
        if cfg.pos_weight_override is not None:
            pos_weight = torch.tensor([float(cfg.pos_weight_override)], dtype=torch.float32)
        elif cfg.pos_weight_auto:
            pw = compute_pos_weight(train_dataset)
            if pw is not None:
                pos_weight = pw
        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif cfg.loss_name == 'focal':
        loss = FocalLossWithLogits(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    elif cfg.loss_name == 'ce':
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"loss_name desconocido: {cfg.loss_name}")
    if device is not None:
        loss = loss.to(device)
        # Mover explícitamente pos_weight si existe
        if isinstance(loss, nn.BCEWithLogitsLoss) and loss.pos_weight is not None:
            loss.pos_weight = loss.pos_weight.to(device)
    return loss

# ---------------------------- Epoch runner -----------------------------------

def run_epoch(model: nn.Module, dl: DataLoader, device: str, train: bool, loss_fn: Optional[nn.Module],
              cfg: TrainConfig, use_model_loss: bool, one_hot: bool, time_step: bool) -> Dict[str, Any]:
    model.train(mode=train)
    scaler = torch.amp.GradScaler('cuda') if (cfg.amp and device.startswith('cuda')) else None

    total_loss = 0.0

    backend = pick_metrics_backend(cfg.metrics_backend)
    frame_group = MetricGroup(backend, device, cfg.decision_threshold)
    window_group = MetricGroup(backend, device, cfg.decision_threshold) if time_step else None

    data_time_sum = 0.0
    compute_time_sum = 0.0
    prev = time.perf_counter()
    total_steps = len(dl)
    tick = max(1, total_steps // max(1, cfg.progress_updates))
    t0 = time.perf_counter()

    # Para reportar frame/window loss al final
    all_true, all_prob = [], []
    win_true, win_probs = [], []

    for step, batch in enumerate(dl):
        now = time.perf_counter()
        data_time_sum += now - prev

        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise RuntimeError("Cada batch debe ser (x, y, ...)")

        x, y = batch[0], batch[1]
        x = to_device(x, device); y = to_device(y, device)

        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = ensure_channel_last(x)

        targets = y
        if one_hot:
            if targets.ndim == 3:
                target_labels = targets.argmax(dim=-1)
            elif targets.ndim == 2 and not time_step:
                target_labels = targets.argmax(dim=-1)
            else:
                target_labels = targets.long()
        else:
            target_labels = targets.float()  # BCE → float

        bs = x.size(0)

        with torch.amp.autocast('cuda', enabled=cfg.amp and device.startswith('cuda')):
            logits = model(x)
            if use_model_loss and hasattr(model, 'compute_loss') and callable(getattr(model, 'compute_loss')):
                loss = model.compute_loss(logits, targets)
            else:
                # Selección por forma real de logits (independiente del flag one_hot)
                if logits.ndim == 3:
                    out_ch = logits.size(-1)
                    if out_ch == 2:
                        ce_t = target_labels.long() if target_labels.dtype != torch.long else target_labels
                        loss = nn.CrossEntropyLoss().to(logits.device)(logits.view(-1, 2), ce_t.view(-1))
                    elif out_ch == 1:
                        bce = build_loss(cfg, dl.dataset, True, False, 2, device=logits.device)
                        loss = bce(logits.squeeze(-1), target_labels)
                    else:
                        raise ValueError(f"logits (B,T,{out_ch}) no soportado")
                else:
                    out_ch = logits.size(-1)
                    if out_ch == 2:
                        ce_t = target_labels.long() if target_labels.dtype != torch.long else target_labels
                        loss = nn.CrossEntropyLoss().to(logits.device)(logits, ce_t)
                    elif out_ch == 1:
                        bce = build_loss(cfg, dl.dataset, False, False, 2, device=logits.device)
                        loss = bce(logits.squeeze(-1), target_labels)
                    else:
                        raise ValueError(f"logits (B,{out_ch}) no soportado")

        if train:
            opt = getattr(run_epoch, '_optimizer'); sch = getattr(run_epoch, '_scheduler', None)
            if opt is None:
                raise RuntimeError("Optimizer no inicializado en run_epoch")
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm:
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm: nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                opt.step()
            if sch is not None and isinstance(sch, CosineAnnealingWarmRestarts):
                current_epoch = getattr(run_epoch, '_epoch_float', 0.0)
                sch.step(current_epoch + step / max(len(dl), 1))

        # métricas
        with torch.no_grad():
            pos_prob, _ = logits_to_probs_and_labels(logits, one_hot)
            if time_step:
                # frame-level
                all_prob.append(pos_prob.detach().cpu().flatten().numpy())
                if one_hot:
                    all_true.append(target_labels.detach().cpu().flatten().numpy())
                else:
                    all_true.append(targets.detach().cpu().flatten().numpy())
                # window-level
                p_win = pos_prob.max(dim=1).values if cfg.window_agg == 'max' else pos_prob.mean(dim=1)
                if one_hot:
                    y_win = (target_labels == 1).any(dim=1).long().float()
                else:
                    y_win = targets.max(dim=1).values.float()
                win_probs.append(p_win.detach().cpu().numpy().astype(np.float32))
                win_true.append(y_win.detach().cpu().numpy().astype(np.int64))
                # grupos
                frame_group.update(pos_prob.flatten(), (targets if not one_hot else target_labels).flatten().long())
                window_group.update(p_win, (y_win >= 0.5).long())  # type: ignore
            else:
                # sample-level
                all_prob.append(pos_prob.detach().cpu().numpy())
                if one_hot:
                    all_true.append(target_labels.detach().cpu().numpy())
                else:
                    all_true.append(targets.detach().cpu().numpy())
                frame_group.update(pos_prob.view(-1), (targets if not one_hot else target_labels).view(-1).long())

        total_loss += float(loss.detach().item()) * bs

        end = time.perf_counter()
        compute_time_sum += end - now
        prev = end

        if cfg.show_progress and (step % tick == 0 or step == total_steps - 1):
            done = step + 1
            pct = 100.0 * done / max(total_steps, 1)
            elapsed = end - t0
            rate = done / max(elapsed, 1e-6)
            eta = (total_steps - done) / max(rate, 1e-6)
            bars = int(pct // (100 / 20))
            bar = '#' * bars + '.' * (20 - bars)
            phase = 'TRAIN' if train else 'VAL  '
            print(f"[{phase}] [{bar}] {pct:5.1f}%  elapsed {elapsed:6.1f}s  ETA {eta:6.1f}s\r", end='')
    if cfg.show_progress:
        print()

    # consolidación
    y_true = np.concatenate(all_true, axis=0) if len(all_true) else np.array([])
    y_prob = np.concatenate(all_prob, axis=0) if len(all_prob) else np.array([])
    avg_loss = total_loss / max(len(dl.dataset), 1)

    out: Dict[str, Any] = {
        'loss': float(avg_loss),
        'data_time_sec': float(data_time_sum),
        'compute_time_sec': float(compute_time_sum),
        'data_time_per_batch_ms': float(1000.0 * data_time_sum / max(total_steps, 1)),
        'compute_time_per_batch_ms': float(1000.0 * compute_time_sum / max(total_steps, 1)),
        'data_frac': float(data_time_sum / max(data_time_sum + compute_time_sum, 1e-12)),
        'compute_frac': float(compute_time_sum / max(data_time_sum + compute_time_sum, 1e-12)),
    }

    # métricas + pérdidas por nivel
    def _mean_bce_probs(y_true_np: np.ndarray, y_prob_np: np.ndarray) -> float:
        eps = 1e-7
        p = np.clip(y_prob_np.astype(np.float64), eps, 1.0 - eps)
        yt = y_true_np.astype(np.float64)
        return float((-(yt*np.log(p) + (1-yt)*np.log(1-p))).mean()) if yt.size else float('nan')

    if y_true.size > 0:
        if time_step:
            # frame
            from sklearn.exceptions import UndefinedMetricWarning
            import warnings as _w
            _w.filterwarnings('ignore', category=UndefinedMetricWarning)
            frame = _compute_metrics_binary(y_true, y_prob, thr=cfg.decision_threshold)
            out.update({f'frame_{k}': float(v) for k, v in frame.items()})
            out['frame_loss'] = _mean_bce_probs(y_true, y_prob)
            if len(win_true) > 0:
                y_true_w = np.concatenate(win_true, axis=0)
                y_prob_w = np.concatenate(win_probs, axis=0)
                window = _compute_metrics_binary(y_true_w, y_prob_w, thr=cfg.decision_threshold)
                out.update({f'window_{k}': float(v) for k, v in window.items()})
                out['window_loss'] = _mean_bce_probs(y_true_w, y_prob_w)
        else:
            sample = _compute_metrics_binary(y_true, y_prob, thr=cfg.decision_threshold)
            out.update({k: float(v) for k, v in sample.items()})

    return out

# versión numpy de métricas (para consolidación final)
def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int,int,int,int]:
    tp = int(np.logical_and(y_true==1, y_pred==1).sum())
    tn = int(np.logical_and(y_true==0, y_pred==0).sum())
    fp = int(np.logical_and(y_true==0, y_pred==1).sum())
    fn = int(np.logical_and(y_true==1, y_pred==0).sum())
    return tp, tn, fp, fn

def _compute_metrics_binary(y_true: np.ndarray, y_prob: np.ndarray, thr: float=0.5) -> Dict[str,float]:
    eps = 1e-12
    y_true = y_true.astype(int)
    y_prob = y_prob.astype(float)
    y_pred = (y_prob >= thr).astype(int)
    tp, tn, fp, fn = _confusion_counts(y_true, y_pred)
    tpr = tp / max(tp + fn, eps)
    tnr = tn / max(tn + fp, eps)
    prec = tp / max(tp + fp, eps)
    rec  = tpr
    f1   = 2 * prec * rec / max(prec + rec, eps)
    acc  = (tp + tn) / max(tp + tn + fp + fn, eps)
    bacc = 0.5 * (tpr + tnr)
    brier= float(np.mean((y_prob - y_true) ** 2))
    out = {
        'accuracy': float(acc), 'balanced_accuracy': float(bacc), 'precision': float(prec), 'recall': float(rec),
        'f1': float(f1), 'sensitivity': float(tpr), 'specificity': float(tnr),
        'recall_class_0': float(tnr), 'recall_class_1': float(tpr), 'brier': float(brier),
        'tp': float(tp), 'tn': float(tn), 'fp': float(fp), 'fn': float(fn)
    }
    try:
        uniq = np.unique(y_true)
        if uniq.size == 2:
            from sklearn.metrics import average_precision_score, roc_auc_score
            out['auprc'] = float(average_precision_score(y_true, y_prob))
            out['auroc'] = float(roc_auc_score(y_true, y_prob))
        else:
            out['auprc'] = float('nan'); out['auroc'] = float('nan')
    except Exception:
        out['auprc'] = float('nan'); out['auroc'] = float('nan')
    return out

# ---------------------------- Entrenamiento ----------------------------------

def make_dataloader(ds, cfg: TrainConfig, sampler=None, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        drop_last=False,
    )


def run_training(model: nn.Module, train_dataset, val_dataset, cfg: TrainConfig) -> Tuple[Dict[str, str], List[Dict[str, float]]]:
    set_seed(cfg.seed)

    # inspección rápida
    tmp_loader = DataLoader(train_dataset, batch_size=1)
    x0, y0 = next(iter(tmp_loader))[:2]
    time_step = (y0.ndim >= 2 and (y0.shape[-1] == 2 or (y0.ndim == 2 and y0.numel() > y0.shape[0])))
    if y0.ndim == 3:
        one_hot = (y0.shape[-1] == 2); time_step = True
    elif y0.ndim == 2 and y0.shape[-1] == 2:
        one_hot = True; time_step = False
    else:
        one_hot = False
        time_step = (y0.ndim == 2 and y0.shape[0] == x0.shape[-2])

    run_name = cfg.run_name or f"eeg_torch_{timestamp()}"
    run_dir = os.path.join(cfg.run_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({**asdict(cfg), 'time_step': time_step, 'one_hot': one_hot}, f, indent=2)

    device = cfg.device
    model = model.to(device)

    # Sonda de forma de salida
    try:
        model.eval()
        with torch.no_grad():
            xp = ensure_channel_last(x0)
            if xp.ndim == 2: xp = xp.unsqueeze(0)
            logits_probe = model(xp.to(device))
        out_nc = logits_probe.size(-1) if logits_probe.ndim in (2,3) else 1
        if (not one_hot) and out_nc == 2:
            warnings.warn("Dataset binario (no one_hot) pero el modelo produce 2 logits; se usará CE con targets 0/1. Considera crear el modelo con one_hot=False para BCE 1-logit.")
        if one_hot and out_nc == 1:
            warnings.warn("Dataset one_hot pero el modelo produce 1 logit; considera crear el modelo con one_hot=True para 2 clases.")
    except Exception:
        pass

    if cfg.compile_model and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)  # type: ignore
        except Exception as e:
            warnings.warn(f"torch.compile falló: {e}")

    sampler = None
    if cfg.use_weighted_sampler:
        labels = None
        if hasattr(train_dataset, 'specs') and isinstance(train_dataset.specs, list) and len(train_dataset.specs) > 0:
            try:
                labels = np.array([int(getattr(s, 'label', 0)) for s in train_dataset.specs], dtype=np.int64)
            except Exception:
                labels = None
        if labels is not None:
            counts = np.bincount(labels, minlength=2).astype(float)
            counts[counts==0] = 1.0
            weights = 1.0 / counts[labels]
            sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(labels), replacement=True)

    train_loader = make_dataloader(train_dataset, cfg, sampler=sampler, shuffle=(sampler is None))
    val_loader   = make_dataloader(val_dataset,   cfg, sampler=None,   shuffle=False)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, T_mult=cfg.T_mult, eta_min=cfg.eta_min) if cfg.use_cosine_warm_restarts else None

    run_epoch._optimizer = optimizer
    run_epoch._scheduler = scheduler

    tb = SummaryWriter(log_dir=run_dir) if (_TB and cfg.use_tensorboard) else None

    best_metric_val = -math.inf if cfg.mode == 'max' else math.inf
    best_epoch = -1
    epochs_no_improve = 0

    best_paths: Dict[str, str] = {}
    history: List[Dict[str, float]] = []

    csv_path = os.path.join(run_dir, 'metrics.csv')
    headers_common = ['epoch','time_step','one_hot','epoch_seconds',
                      'train_data_time_sec','train_compute_time_sec','train_data_frac','train_compute_frac',
                      'val_data_time_sec','val_compute_time_sec','val_data_frac','val_compute_frac']
    headers_main = ['loss','accuracy','balanced_accuracy','precision','recall','f1','sensitivity','specificity','brier','tp','tn','fp','fn','auprc','auroc']
    headers_frame = [f'frame_{h}' for h in headers_main]
    headers_window = [f'window_{h}' for h in headers_main]

    with open(csv_path, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        if time_step:
            writer.writerow(headers_common + [f'train_{h}' for h in (headers_frame + headers_window)] + [f'val_{h}' for h in (headers_frame + headers_window)])
        else:
            writer.writerow(headers_common + [f'train_{h}' for h in headers_main] + [f'val_{h}' for h in headers_main])

        for epoch in range(cfg.epochs):
            run_epoch._epoch_float = float(epoch)
            t0 = time.time()
            train_metrics = run_epoch(model, train_loader, device, True,  None, cfg, cfg.use_model_loss, one_hot, time_step)
            val_metrics   = run_epoch(model, val_loader,   device, False, None, cfg, cfg.use_model_loss, one_hot, time_step)
            t1 = time.time()

            if scheduler is not None and not isinstance(scheduler, CosineAnnealingWarmRestarts):
                scheduler.step()

            epoch_seconds = t1 - t0
            row_common = [epoch, int(time_step), int(one_hot), epoch_seconds,
                          train_metrics.get('data_time_sec', float('nan')),
                          train_metrics.get('compute_time_sec', float('nan')),
                          train_metrics.get('data_frac', float('nan')),
                          train_metrics.get('compute_frac', float('nan')),
                          val_metrics.get('data_time_sec', float('nan')),
                          val_metrics.get('compute_time_sec', float('nan')),
                          val_metrics.get('data_frac', float('nan')),
                          val_metrics.get('compute_frac', float('nan'))]

            if tb is not None:
                tb.add_scalar('time/epoch_seconds', epoch_seconds, epoch)
                tb.add_scalar('time/train_data_frac', train_metrics.get('data_frac', float('nan')), epoch)
                tb.add_scalar('time/train_compute_frac', train_metrics.get('compute_frac', float('nan')), epoch)
                tb.add_scalar('time/val_data_frac', val_metrics.get('data_frac', float('nan')), epoch)
                tb.add_scalar('time/val_compute_frac', val_metrics.get('compute_frac', float('nan')), epoch)
                for k, v in train_metrics.items(): tb.add_scalar(f'train/{k}', v, epoch)
                for k, v in val_metrics.items():   tb.add_scalar(f'val/{k}',   v, epoch)

            if time_step:
                row = row_common + [train_metrics.get(f, float('nan')) for f in (headers_frame + headers_window)] + [val_metrics.get(f, float('nan')) for f in (headers_frame + headers_window)]
            else:
                row = row_common + [train_metrics.get(f, float('nan')) for f in headers_main] + [val_metrics.get(f, float('nan')) for f in headers_main]
            writer.writerow(row); fcsv.flush()

            # selección de métrica a monitorear
            monitor_key = cfg.monitor_metric
            if time_step and monitor_key.startswith('val_') and ('frame_' not in monitor_key and 'window_' not in monitor_key):
                base = monitor_key.replace('val_', '')
                monitor_key = f'val_window_{base}' if f'window_{base}' in val_metrics else f'val_frame_{base}'
            if monitor_key.startswith('val_'):
                base = monitor_key.replace('val_', '')
                current = val_metrics.get(base, None)
            else:
                current = val_metrics.get(monitor_key, None)
            if current is None:
                current = val_metrics.get('window_balanced_accuracy', None) if time_step else None
            if current is None:
                current = val_metrics.get('frame_balanced_accuracy', None) if time_step else None
            if current is None:
                current = val_metrics.get('balanced_accuracy', None)

            improved = (current > best_metric_val + cfg.min_delta) if cfg.mode == 'max' else (current < best_metric_val - cfg.min_delta)
            if improved and current is not None:
                best_metric_val = current
                best_epoch = epoch
                p_main = os.path.join(run_dir, f"best_{cfg.monitor_metric}.pth")
                torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_metrics': val_metrics}, p_main)
                best_paths[cfg.monitor_metric] = p_main
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # Mejor por F1 (frame_f1 si time_step)
            f1_key = 'frame_f1' if time_step else 'f1'
            key_tag = 'val_' + f1_key
            prev_f1 = -1.0
            prev_path = best_paths.get(key_tag, None)
            if prev_path and os.path.isfile(prev_path):
                try:
                    prev_state = torch.load(prev_path, map_location='cpu')
                    prev_f1 = float(prev_state.get('val_metrics', {}).get(f1_key, -1.0))
                except Exception:
                    prev_f1 = -1.0
            curr_f1 = float(val_metrics.get(f1_key, -1.0))
            if curr_f1 > prev_f1:
                p2 = os.path.join(run_dir, f"best_{key_tag}.pth")
                torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_metrics': val_metrics}, p2)
                best_paths[key_tag] = p2

            if cfg.save_last:
                torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_metrics': val_metrics}, os.path.join(run_dir, 'last.pth'))

            hist_row = {'epoch': epoch, **{f'train_{k}': float(v) for k, v in train_metrics.items()}, **{f'val_{k}': float(v) for k, v in val_metrics.items()}}
            history.append(hist_row)

            if epochs_no_improve >= cfg.patience:
                print(f"[EarlyStopping] Sin mejora en {cfg.patience} épocas. Mejor en época {best_epoch}: {best_metric_val:.6f}")
                break

    with open(os.path.join(run_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump({'history': history}, f, indent=2)
    if tb is not None:
        tb.close()

    print("Mejores checkpoints:")
    for k, v in best_paths.items():
        print(f"  {k}: {v}")

    return best_paths, history
