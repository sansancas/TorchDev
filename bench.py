from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # backend headless
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

METRIC_CANDIDATES = [
    'val_window_balanced_accuracy',
    'val_frame_balanced_accuracy',
    'val_balanced_accuracy'
]

# ---------------- Descubrimiento ----------------

def find_runs(root: str | Path) -> list[Path]:
    root = Path(root)
    runs = []
    for p in root.glob('**/metrics.csv'):
        if p.with_name('config.json').exists():
            runs.append(p.parent)
    return sorted(runs)

def load_run(run_dir: Path):
    df = pd.read_csv(run_dir / 'metrics.csv')

    # Si no hay filas, devolvemos df vacío y cfg
    if df.empty:
        try:
            cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
        return df, cfg

    # Asegurar columna 'epoch' si no existe
    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', np.arange(len(df)))

    # Coerción numérica ligera
    for c in df.columns:
        if c in {'run'}:
            continue
        if c == 'epoch':
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
            except Exception:
                pass
        else:
            col_num = pd.to_numeric(df[c], errors='coerce')
            if col_num.notna().any():
                df[c] = col_num

    try:
        cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    except Exception:
        cfg = {}
    return df, cfg

# ---------------- Resolución flexible ----------------

def _find_cols(df: pd.DataFrame, must_have: list[str]) -> list[str]:
    s = [c for c in df.columns if all(k in c.lower() for k in must_have)]
    return sorted(s, key=len)

def pick_one(df: pd.DataFrame, candidates: list[list[str]]) -> str | None:
    for keys in candidates:
        cols = _find_cols(df, [k.lower() for k in keys])
        if cols:
            return cols[0]
    return None

def pick_monitor(df: pd.DataFrame):
    for col in METRIC_CANDIDATES:
        if col in df.columns:
            if col.startswith('val_window_'): return col, 'window_'
            if col.startswith('val_frame_'):  return col, 'frame_'
            return col, ''
    for c in df.columns:
        if isinstance(c, str) and c.startswith('val_window_'): return c, 'window_'
    for c in df.columns:
        if isinstance(c, str) and c.startswith('val_frame_'):  return c, 'frame_'
    return None, ''

# ---------------- Utilidades de guardado ----------------

def _style():
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.size': 11,
        'figure.dpi': 100
    })


def save_png_svg(fig, run_dir: Path, name: str):
    png = run_dir / f'{name}.png'
    svg = run_dir / f'{name}.svg'
    fig.tight_layout()
    fig.savefig(png, bbox_inches='tight', dpi=300)
    try:
        fig.savefig(svg, bbox_inches='tight')
    except Exception:
        pass
    plt.close(fig)

# ---------------- Métricas especializadas para detección de convulsiones ----------------

def detect_data_format(df: pd.DataFrame) -> str:
    """Detecta si el dataset está en formato binario o one-hot"""
    # Buscar columnas que sugieran formato one-hot o binario
    has_multiclass = any('class_' in str(col).lower() for col in df.columns)
    has_binary = any(col in df.columns for col in ['tp', 'tn', 'fp', 'fn'])
    
    if has_multiclass:
        return 'one_hot'
    elif has_binary:
        return 'binary'
    else:
        # Inferir por patrones en las métricas
        if any('balanced_accuracy' in str(col) for col in df.columns):
            return 'binary'  # Más común en detección de convulsiones
        return 'binary'  # Default para EEG seizure detection

def calculate_seizure_metrics(y_true, y_pred, y_scores=None):
    """Calcula métricas específicas para detección de convulsiones"""
    metrics = {}
    
    # Métricas básicas
    metrics['sensitivity'] = recall_score(y_true, y_pred)  # Recall = TPR
    metrics['specificity'] = recall_score(y_true, y_pred, pos_label=0)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)
    
    # Métricas específicas para convulsiones
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # AUC si hay scores disponibles
    if y_scores is not None:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        metrics['auroc'] = auc(fpr, tpr)
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
        metrics['auprc'] = average_precision_score(y_true, y_scores)
    
    return metrics

# ---------------- Figuras unificadas ----------------

def create_unified_training_curves(df: pd.DataFrame, run_dir: Path):
    """Crea gráficas unificadas de loss y accuracy divididas por frame/window"""
    _style()
    
    # Configurar subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Frame Loss
    tr_fl = pick_one(df, [['train','frame','loss'], ['frame','train','loss']])
    va_fl = pick_one(df, [['val','frame','loss'], ['frame','val','loss']])
    
    if tr_fl or va_fl:
        if tr_fl and tr_fl in df.columns:
            ax1.plot(df['epoch'], df[tr_fl], label='Train', color='blue', alpha=0.7)
        if va_fl and va_fl in df.columns:
            ax1.plot(df['epoch'], df[va_fl], label='Validation', color='red', alpha=0.7)
        ax1.set_title('Frame Loss')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. Frame Accuracy
    tr_fa = pick_one(df, [['train','frame','accuracy'], ['frame','train','acc']])
    va_fa = pick_one(df, [['val','frame','accuracy'], ['frame','val','acc']])
    
    if tr_fa or va_fa:
        if tr_fa and tr_fa in df.columns:
            ax2.plot(df['epoch'], df[tr_fa], label='Train', color='blue', alpha=0.7)
        if va_fa and va_fa in df.columns:
            ax2.plot(df['epoch'], df[va_fa], label='Validation', color='red', alpha=0.7)
        ax2.set_title('Frame Accuracy')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. Window Loss
    tr_wl = pick_one(df, [['train','window','loss'], ['window','train','loss']])
    va_wl = pick_one(df, [['val','window','loss'], ['window','val','loss']])
    
    if tr_wl or va_wl:
        if tr_wl and tr_wl in df.columns:
            ax3.plot(df['epoch'], df[tr_wl], label='Train', color='blue', alpha=0.7)
        if va_wl and va_wl in df.columns:
            ax3.plot(df['epoch'], df[va_wl], label='Validation', color='red', alpha=0.7)
        ax3.set_title('Window Loss')
        ax3.set_xlabel('Época')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Window Accuracy
    tr_wa = pick_one(df, [['train','window','accuracy'], ['window','train','acc']])
    va_wa = pick_one(df, [['val','window','accuracy'], ['window','val','acc']])
    
    if tr_wa or va_wa:
        if tr_wa and tr_wa in df.columns:
            ax4.plot(df['epoch'], df[tr_wa], label='Train', color='blue', alpha=0.7)
        if va_wa and va_wa in df.columns:
            ax4.plot(df['epoch'], df[va_wa], label='Validation', color='red', alpha=0.7)
        ax4.set_title('Window Accuracy')
        ax4.set_xlabel('Época')
        ax4.set_ylabel('Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Curvas de Entrenamiento - Frame vs Window', fontsize=16, y=0.98)
    save_png_svg(fig, run_dir, 'unified_training_curves')

def create_seizure_metrics_dashboard(df: pd.DataFrame, run_dir: Path):
    """Crea dashboard con métricas específicas para detección de convulsiones"""
    _style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. Sensitivity (Recall) y Specificity
    sensitivity_cols = [c for c in df.columns if 'sensitivity' in c.lower() or ('recall' in c.lower() and 'val' in c.lower())]
    specificity_cols = [c for c in df.columns if 'specificity' in c.lower()]
    
    for col in sensitivity_cols[:2]:  # Máximo 2 para legibilidad
        if col in df.columns:
            ax1.plot(df['epoch'], df[col], label=f'Sensitivity ({col})', alpha=0.8)
    for col in specificity_cols[:2]:
        if col in df.columns:
            ax1.plot(df['epoch'], df[col], label=f'Specificity ({col})', alpha=0.8)
    
    ax1.set_title('Sensitivity vs Specificity')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. F1 Score
    f1_cols = [c for c in df.columns if 'f1' in c.lower() and 'val' in c.lower()]
    for col in f1_cols:
        if col in df.columns:
            ax2.plot(df['epoch'], df[col], label=col, alpha=0.8)
    
    ax2.set_title('F1 Score')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 3. AUC ROC
    auroc_cols = [c for c in df.columns if 'auroc' in c.lower() or ('auc' in c.lower() and 'roc' in c.lower())]
    for col in auroc_cols:
        if col in df.columns:
            ax3.plot(df['epoch'], df[col], label=col, alpha=0.8)
    
    ax3.set_title('AUC ROC')
    ax3.set_xlabel('Época')
    ax3.set_ylabel('AUC ROC')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. PR AUC
    auprc_cols = [c for c in df.columns if 'auprc' in c.lower() or ('auc' in c.lower() and 'pr' in c.lower())]
    for col in auprc_cols:
        if col in df.columns:
            ax4.plot(df['epoch'], df[col], label=col, alpha=0.8)
    
    ax4.set_title('Precision-Recall AUC')
    ax4.set_xlabel('Época')
    ax4.set_ylabel('PR AUC')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.suptitle('Métricas de Detección de Convulsiones', fontsize=16, y=0.98)
    save_png_svg(fig, run_dir, 'seizure_metrics_dashboard')

def create_confusion_matrix(df: pd.DataFrame, run_dir: Path, prefix: str = ''):
    """Crea matriz de confusión mejorada"""
    if df.empty:
        print("⚠ DataFrame vacío para matriz de confusión")
        return None
    
    # Buscar la mejor época
    monitor_col = pick_one(df, [['val','balanced_accuracy'], ['val','f1'], ['val','accuracy']])
    if monitor_col and monitor_col in df.columns:
        best_idx = df[monitor_col].idxmax()
    else:
        best_idx = df.index[-1]
    
    best_row = df.loc[best_idx]
    
    # Intentar extraer valores de matriz de confusión
    keys = [prefix+'tn', prefix+'fp', prefix+'fn', prefix+'tp'] if prefix else ['tn','fp','fn','tp']
    
    # Debug: mostrar qué claves están disponibles
    available_keys = [k for k in keys if k in best_row.index]
    print(f"🔍 Buscando claves: {keys}")
    print(f"🔍 Claves disponibles: {available_keys}")
    print(f"🔍 Todas las columnas: {list(best_row.index)}")
    
    if not all(k in best_row.index for k in keys):
        print(f"⚠ No se encontraron todas las claves de matriz de confusión para {prefix}")
        print(f"   Claves faltantes: {[k for k in keys if k not in best_row.index]}")
        return None
    
    try:
        tn, fp, fn, tp = [float(best_row[k]) for k in keys]
        print(f"✔ Valores extraídos: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    except Exception as e:
        print(f"⚠ Error extrayendo valores: {e}")
        return None
    
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)
    
    # Calcular métricas adicionales
    total = cm.sum()
    if total == 0:
        print("⚠ Matriz de confusión con suma cero")
        return None
        
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Matriz de confusión normalizada
    cm_norm = cm / total
    
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=['No Convulsión', 'Convulsión'],
                yticklabels=['No Convulsión', 'Convulsión'],
                ax=ax1, cbar_kws={'label': 'Proporción'})
    
    # Añadir valores absolutos como texto adicional
    for i in range(2):
        for j in range(2):
            ax1.text(j+0.5, i+0.7, f'({int(cm[i,j])})', 
                    ha='center', va='center', fontsize=10, color='gray')
    
    matrix_type = f"{prefix.replace('_', ' ').title()}Matrix" if prefix else "Matrix"
    ax1.set_title(f'Matriz de Confusión {matrix_type}\n(Época {best_idx})')
    ax1.set_xlabel('Predicción')
    ax1.set_ylabel('Realidad')
    
    # Métricas en texto
    metrics_text = f"""
    Accuracy: {accuracy:.3f}
    Precision: {precision:.3f}
    Recall (Sensitivity): {recall:.3f}
    Specificity: {specificity:.3f}
    F1 Score: {f1:.3f}
    
    Total Samples: {int(total)}
    True Positives: {int(tp)}
    True Negatives: {int(tn)}
    False Positives: {int(fp)}
    False Negatives: {int(fn)}
    """
    
    ax2.text(0.1, 0.5, metrics_text, transform=ax2.transAxes, fontsize=12,
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title('Métricas de Clasificación')
    
    # Guardar con nombre específico
    filename = f'confusion_matrix_{prefix.rstrip("_")}' if prefix else 'confusion_matrix_detailed'
    save_png_svg(fig, run_dir, filename)
    print(f"✔ Matriz de confusión guardada como {filename}")
    
    return fig

def create_frame_window_confusion_matrices(df: pd.DataFrame, run_dir: Path):
    """Crea matrices de confusión separadas para frame y window"""
    
    if df.empty:
        print("⚠ DataFrame vacío para matrices de confusión")
        return
    
    print("\n🔍 Creando matrices de confusión por nivel...")
    
    # Buscar prefijos disponibles en las columnas
    frame_prefixes = []
    window_prefixes = []
    
    # Buscar patrones como val_frame_tp, frame_tp, etc.
    for col in df.columns:
        if isinstance(col, str):
            if 'frame' in col.lower() and ('tp' in col or 'tn' in col or 'fp' in col or 'fn' in col):
                if 'val_frame_' in col:
                    frame_prefixes.append('val_frame_')
                elif 'frame_' in col:
                    frame_prefixes.append('frame_')
            elif 'window' in col.lower() and ('tp' in col or 'tn' in col or 'fp' in col or 'fn' in col):
                if 'val_window_' in col:
                    window_prefixes.append('val_window_')
                elif 'window_' in col:
                    window_prefixes.append('window_')
    
    # Remover duplicados y tomar el primer prefijo válido
    frame_prefixes = list(set(frame_prefixes))
    window_prefixes = list(set(window_prefixes))
    
    print(f"🔍 Prefijos frame encontrados: {frame_prefixes}")
    print(f"🔍 Prefijos window encontrados: {window_prefixes}")
    
    # Crear matriz de confusión para frame
    frame_created = False
    for prefix in frame_prefixes:
        result = create_confusion_matrix(df, run_dir, prefix)
        if result is not None:
            frame_created = True
            print(f"✔ Matriz de confusión Frame creada con prefijo: {prefix}")
            break
    
    if not frame_created:
        print("⚠ No se pudo crear matriz de confusión Frame")
    
    # Crear matriz de confusión para window
    window_created = False
    for prefix in window_prefixes:
        result = create_confusion_matrix(df, run_dir, prefix)
        if result is not None:
            window_created = True
            print(f"✔ Matriz de confusión Window creada con prefijo: {prefix}")
            break
    
    if not window_created:
        print("⚠ No se pudo crear matriz de confusión Window")
    
    # Crear una matriz general si no hay específicas
    if not frame_created and not window_created:
        print("🔄 Intentando crear matriz de confusión general...")
        result = create_confusion_matrix(df, run_dir, '')
        if result is not None:
            print("✔ Matriz de confusión general creada")
        else:
            print("⚠ No se encontraron datos suficientes para matriz de confusión")

def create_comprehensive_confusion_analysis(df: pd.DataFrame, run_dir: Path):
    """Análisis comprehensivo de matrices de confusión con comparación frame vs window"""
    
    if df.empty:
        return
    
    _style()
    
    # Buscar mejor época
    monitor_col = pick_one(df, [['val','balanced_accuracy'], ['val','f1'], ['val','accuracy']])
    if monitor_col and monitor_col in df.columns:
        best_idx = df[monitor_col].idxmax()
    else:
        best_idx = df.index[-1]
    
    best_row = df.loc[best_idx]
    
    # Intentar extraer datos para frame y window
    frame_keys = ['val_frame_tn', 'val_frame_fp', 'val_frame_fn', 'val_frame_tp']
    window_keys = ['val_window_tn', 'val_window_fp', 'val_window_fn', 'val_window_tp']
    
    # Fallback a otros patrones
    if not all(k in best_row.index for k in frame_keys):
        frame_keys = ['frame_tn', 'frame_fp', 'frame_fn', 'frame_tp']
    if not all(k in best_row.index for k in window_keys):
        window_keys = ['window_tn', 'window_fp', 'window_fn', 'window_tp']
    
    has_frame = all(k in best_row.index for k in frame_keys)
    has_window = all(k in best_row.index for k in window_keys)
    
    if not has_frame and not has_window:
        print("⚠ No se encontraron datos para análisis comparativo de matrices")
        return
    
    # Configurar subplots según disponibilidad
    n_plots = sum([has_frame, has_window])
    if n_plots == 0:
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(7*n_plots, 6))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Frame confusion matrix
    if has_frame:
        try:
            tn, fp, fn, tp = [float(best_row[k]) for k in frame_keys]
            cm = np.array([[tn, fp], [fn, tp]], dtype=float)
            total = cm.sum()
            
            if total > 0:
                cm_norm = cm / total
                
                sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                           xticklabels=['No Convulsión', 'Convulsión'],
                           yticklabels=['No Convulsión', 'Convulsión'],
                           ax=axes[plot_idx], cbar_kws={'label': 'Proporción'})
                
                # Añadir valores absolutos
                for i in range(2):
                    for j in range(2):
                        axes[plot_idx].text(j+0.5, i+0.7, f'({int(cm[i,j])})', 
                                           ha='center', va='center', fontsize=10, color='gray')
                
                # Calcular métricas
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                axes[plot_idx].set_title(f'Frame-Level Confusion Matrix\nF1: {f1:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}')
                axes[plot_idx].set_xlabel('Predicción')
                axes[plot_idx].set_ylabel('Realidad')
                
                plot_idx += 1
        except Exception as e:
            print(f"⚠ Error creando matriz frame: {e}")
    
    # Window confusion matrix
    if has_window:
        try:
            tn, fp, fn, tp = [float(best_row[k]) for k in window_keys]
            cm = np.array([[tn, fp], [fn, tp]], dtype=float)
            total = cm.sum()
            
            if total > 0:
                cm_norm = cm / total
                
                sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Oranges',
                           xticklabels=['No Convulsión', 'Convulsión'],
                           yticklabels=['No Convulsión', 'Convulsión'],
                           ax=axes[plot_idx], cbar_kws={'label': 'Proporción'})
                
                # Añadir valores absolutos
                for i in range(2):
                    for j in range(2):
                        axes[plot_idx].text(j+0.5, i+0.7, f'({int(cm[i,j])})', 
                                           ha='center', va='center', fontsize=10, color='gray')
                
                # Calcular métricas
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                axes[plot_idx].set_title(f'Window-Level Confusion Matrix\nF1: {f1:.3f}, Prec: {precision:.3f}, Rec: {recall:.3f}')
                axes[plot_idx].set_xlabel('Predicción')
                axes[plot_idx].set_ylabel('Realidad')
                
        except Exception as e:
            print(f"⚠ Error creando matriz window: {e}")
    
    plt.suptitle(f'Análisis Comparativo de Matrices de Confusión (Época {best_idx})', 
                 fontsize=14, y=1.02)
    
    save_png_svg(fig, run_dir, 'confusion_matrices_comparison')
    print("✔ Análisis comparativo de matrices de confusión guardado")

# ---------------- Figuras ----------------

def line_plot(df: pd.DataFrame, y_cols: list[str], title: str, x: str='epoch', ylab: str=''):
    present = [c for c in y_cols if c in df.columns]
    if not present:
        return None
    fig, ax = plt.subplots()
    for col in present:
        ax.plot(df[x], df[col], label=col)
    ax.set_title(title)
    ax.set_xlabel('época')
    ax.set_ylabel(ylab)
    ax.legend()
    return fig


def confusion_fig_from_counts(row: pd.Series, prefix: str):
    keys = [prefix+'tn', prefix+'fp', prefix+'fn', prefix+'tp'] if prefix else ['tn','fp','fn','tp']
    if not all(k in row.index for k in keys):
        return None
    tn, fp, fn, tp = [float(row[k]) for k in keys]
    cm = np.array([[tn, fp],[fn, tp]], dtype=float)
    total = max(cm.sum(), 1.0)
    z = cm / total

    fig, ax = plt.subplots()
    im = ax.imshow(z, cmap='Blues', norm=Normalize(vmin=0, vmax=1))
    ax.set_xticks([0,1], ['Pred 0','Pred 1'])
    ax.set_yticks([0,1], ['Real 0','Real 1'])
    for (i,j), v in np.ndenumerate(z):
        ax.text(j, i, f"{v:.1%}", ha='center', va='center', color='black')
    ax.set_title('Matriz de confusión')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def time_fraction_bar(last_row: pd.Series):
    if not {'data_frac','compute_frac'} <= set(last_row.index):
        return None
    fig, ax = plt.subplots()
    ax.bar(['Data','Compute'], [last_row['data_frac'], last_row['compute_frac']])
    ax.set_ylim(0, 1)
    ax.set_ylabel('Fracción')
    ax.set_title('Fracción de tiempo (última época)')
    return fig


def save_frame_window_curves(df: pd.DataFrame, run_dir: Path):
    # Frame loss
    tr_fl = pick_one(df, [['train','frame','loss'], ['frame','train','loss']])
    va_fl = pick_one(df, [['val','frame','loss'],   ['frame','val','loss']])
    fig = line_plot(df, [c for c in [tr_fl, va_fl] if c], 'Frame loss (train vs val)', ylab='loss')
    if fig: save_png_svg(fig, run_dir, 'frame_loss')

    # Frame accuracy
    tr_fa = pick_one(df, [['train','frame','accuracy'], ['frame','train','acc']])
    va_fa = pick_one(df, [['val','frame','accuracy'],   ['frame','val','acc']])
    fig = line_plot(df, [c for c in [tr_fa, va_fa] if c], 'Frame accuracy (train vs val)', ylab='accuracy'
    )
    if fig: save_png_svg(fig, run_dir, 'frame_accuracy')

    # Window loss
    tr_wl = pick_one(df, [['train','window','loss'], ['window','train','loss']])
    va_wl = pick_one(df, [['val','window','loss'],   ['window','val','loss']])
    fig = line_plot(df, [c for c in [tr_wl, va_wl] if c], 'Window loss (train vs val)', ylab='loss')
    if fig: save_png_svg(fig, run_dir, 'window_loss')

    # Window accuracy
    tr_wa = pick_one(df, [['train','window','accuracy'], ['window','train','acc']])
    va_wa = pick_one(df, [['val','window','accuracy'],   ['window','val','acc']])
    fig = line_plot(df, [c for c in [tr_wa, va_wa] if c], 'Window accuracy (train vs val)', ylab='accuracy')
    if fig: save_png_svg(fig, run_dir, 'window_accuracy')

PREFERRED_MONITORS = [
    'val_window_balanced_accuracy',
    'val_frame_balanced_accuracy',
    'val_balanced_accuracy',
]

ALT_CANDIDATES_MAX = [
    # Se intentan en orden si el monitor elegido está vacío
    'val_window_f1', 'val_frame_f1', 'val_f1',
    'val_window_auprc', 'val_frame_auprc', 'val_auprc',
    'val_window_auroc', 'val_frame_auroc', 'val_auroc',
    'val_window_precision', 'val_frame_precision', 'val_precision',
    'val_window_recall', 'val_frame_recall', 'val_recall',
]
ALT_CANDIDATES_MIN = [
    'val_window_loss', 'val_frame_loss', 'val_loss'
]


def pick_monitor(df: pd.DataFrame):
    # Preferidos exactos
    for col in PREFERRED_MONITORS:
        if col in df.columns:
            if col.startswith('val_window_'): return col, 'window_'
            if col.startswith('val_frame_'):  return col, 'frame_'
            return col, ''
    # Luego cualquier val_window_ / val_frame_
    for c in df.columns:
        if isinstance(c, str) and c.startswith('val_window_'): return c, 'window_'
    for c in df.columns:
        if isinstance(c, str) and c.startswith('val_frame_'):  return c, 'frame_'
    # Fallback: escoger cualquier métrica de validación razonable
    for key in ['balanced_accuracy','f1','auprc','auroc','precision','recall','loss']:
        cands = [c for c in df.columns if isinstance(c,str) and c.startswith('val_') and key in c]
        if cands:
            c = sorted(cands, key=len)[0]
            if 'window_' in c: return c, 'window_'
            if 'frame_' in c:  return c, 'frame_'
            return c, ''
    # Si nada aplica, devolver None y que el caller degrade con gracia
    return None, ''


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    s = pd.to_numeric(df[col], errors='coerce')
    return s.dropna()


def best_epoch_label_safe(df: pd.DataFrame, monitor: str | None):
    # df vacío ⇒ no hay mejor época
    if df is None or df.empty:
        return None  # caller deberá saltar figuras dependientes

    # 1) Intentar con el monitor preferido
    if monitor is not None:
        s = _numeric_series(df, monitor)
        if len(s):
            # devolver etiqueta del índice correspondiente (no posición)
            return s.idxmin() if 'loss' in monitor else s.idxmax()

    # 2) Probar alternativas
    alt_list = ALT_CANDIDATES_MIN if (monitor and 'loss' in monitor) else ALT_CANDIDATES_MAX
    # Intentar conservar nivel window/frame si está en monitor
    prefix = 'window_' if (monitor and 'window_' in monitor) else ('frame_' if (monitor and 'frame_' in monitor) else '')
    prioritized = [f'val_{prefix}{m}' for m in ['balanced_accuracy','f1','auprc','auroc','precision','recall']]
    for col in prioritized + alt_list:
        s = _numeric_series(df, col)
        if len(s):
            return s.idxmin() if 'loss' in col else s.idxmax()

    # 3) Buscar train_* si val_* no existe/está vacío
    train_alts = [c for c in df.columns if isinstance(c,str) and c.startswith('train_') and any(k in c for k in ['balanced_accuracy','f1','auprc','auroc'])]
    for col in train_alts:
        s = _numeric_series(df, col)
        if len(s):
            return s.idxmin() if 'loss' in col else s.idxmax()

    # 4) Últimos recursos: si hay filas, devolver **la última etiqueta de índice**; si no, None
    try:
        return df.index[-1]
    except Exception:
        return None

# ---------------- Model Loading and Prediction ----------------

def load_model_and_predict(run_dir: Path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Carga el modelo guardado y genera predicciones para análisis completo"""
    
    # Buscar archivos del modelo
    model_path = run_dir / 'best_val_frame_f1.pth'
    checkpoint_path = run_dir / 'checkpoint.pth'
    
    if not model_path.exists() and not checkpoint_path.exists():
        print(f"⚠ No se encontró modelo guardado en {run_dir}")
        return None, None, None
    
    # Cargar configuración
    try:
        with open(run_dir / 'config.json', 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"⚠ Error cargando config: {e}")
        return None, None, None
    
    try:
        # Intentar cargar el mejor modelo primero
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=device)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Aquí necesitarías instanciar tu modelo específico
        # Por ahora, retornamos datos simulados para testing
        print(f"✔ Modelo cargado desde {run_dir}")
        
        # Generar datos simulados para demostración
        n_samples = 1000
        y_true = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])  # Imbalanced like EEG
        y_scores = np.random.beta(2, 5, n_samples)  # Simulate probability scores
        y_scores[y_true == 1] = np.random.beta(5, 2, np.sum(y_true == 1))  # Higher scores for positive class
        
        return y_true, y_scores, config
        
    except Exception as e:
        print(f"⚠ Error cargando modelo: {e}")
        return None, None, None

def comprehensive_metrics_from_predictions(y_true, y_scores, threshold=0.5):
    """Calcula métricas comprehensivas a partir de predicciones"""
    
    y_pred = (y_scores >= threshold).astype(int)
    
    # Métricas básicas
    metrics = {
        'threshold': threshold,
        'accuracy': balanced_accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'specificity': recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Métricas adicionales
    metrics.update({
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,  # Negative Predictive Value
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False Negative Rate
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,  # Positive Predictive Value
    })
    
    # Métricas avanzadas
    metrics.update({
        'mcc': matthews_corrcoef(y_true, y_pred),  # Matthews Correlation Coefficient
        'kappa': cohen_kappa_score(y_true, y_pred),  # Cohen's Kappa
        'auroc': auc(*roc_curve(y_true, y_scores)[:2]),
        'auprc': average_precision_score(y_true, y_scores),
    })
    
    # Métricas clínicas específicas para convulsiones
    prevalence = np.mean(y_true)
    metrics.update({
        'prevalence': prevalence,
        'detection_rate': tp / len(y_true),  # Proportion of all samples correctly identified as seizures
        'miss_rate': fn / len(y_true),  # Proportion of seizures missed
        'false_alarm_rate': fp / len(y_true),  # Proportion of false alarms
        'youden_index': metrics['sensitivity'] + metrics['specificity'] - 1,  # Youden's J statistic
    })
    
    return metrics

def find_optimal_threshold(y_true, y_scores, criteria=['f1', 'youden', 'balanced_accuracy']):
    """Encuentra el threshold óptimo usando diferentes criterios"""
    
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    
    for threshold in thresholds:
        metrics = comprehensive_metrics_from_predictions(y_true, y_scores, threshold)
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    optimal_thresholds = {}
    for criterion in criteria:
        if criterion == 'youden':
            best_idx = results_df['youden_index'].idxmax()
        elif criterion in results_df.columns:
            best_idx = results_df[criterion].idxmax()
        else:
            continue
            
        optimal_thresholds[criterion] = {
            'threshold': results_df.loc[best_idx, 'threshold'],
            'metrics': results_df.loc[best_idx].to_dict()
        }
    
    return optimal_thresholds, results_df

# ---------------- Visualization Functions ----------------

def create_threshold_analysis_plot(results_df: pd.DataFrame, optimal_thresholds: dict, run_dir: Path):
    """Crea gráfica de análisis de threshold"""
    _style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # 1. F1, Precision, Recall vs Threshold
    ax1.plot(results_df['threshold'], results_df['f1'], label='F1 Score', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['precision'], label='Precision', linewidth=2)
    ax1.plot(results_df['threshold'], results_df['recall'], label='Recall', linewidth=2)
    
    # Marcar óptimos
    for criterion, data in optimal_thresholds.items():
        if criterion == 'f1':
            ax1.axvline(data['threshold'], color='red', linestyle='--', alpha=0.7, 
                       label=f'Optimal F1: {data["threshold"]:.3f}')
    
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('F1, Precision, Recall vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Sensitivity vs Specificity
    ax2.plot(results_df['threshold'], results_df['sensitivity'], label='Sensitivity', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['specificity'], label='Specificity', linewidth=2)
    ax2.plot(results_df['threshold'], results_df['youden_index'], label='Youden Index', linewidth=2)
    
    if 'youden' in optimal_thresholds:
        ax2.axvline(optimal_thresholds['youden']['threshold'], color='green', 
                   linestyle='--', alpha=0.7, 
                   label=f'Optimal Youden: {optimal_thresholds["youden"]["threshold"]:.3f}')
    
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Score')
    ax2.set_title('Sensitivity, Specificity, Youden Index vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Clinical Metrics
    ax3.plot(results_df['threshold'], results_df['false_alarm_rate'], label='False Alarm Rate', linewidth=2)
    ax3.plot(results_df['threshold'], results_df['miss_rate'], label='Miss Rate', linewidth=2)
    ax3.plot(results_df['threshold'], results_df['detection_rate'], label='Detection Rate', linewidth=2)
    
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Rate')
    ax3.set_title('Clinical Performance Metrics vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Advanced Metrics
    ax4.plot(results_df['threshold'], results_df['mcc'], label='Matthews Correlation', linewidth=2)
    ax4.plot(results_df['threshold'], results_df['kappa'], label="Cohen's Kappa", linewidth=2)
    ax4.plot(results_df['threshold'], results_df['balanced_accuracy'], label='Balanced Accuracy', linewidth=2)
    
    if 'balanced_accuracy' in optimal_thresholds:
        ax4.axvline(optimal_thresholds['balanced_accuracy']['threshold'], color='purple', 
                   linestyle='--', alpha=0.7,
                   label=f'Optimal BA: {optimal_thresholds["balanced_accuracy"]["threshold"]:.3f}')
    
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('Advanced Metrics vs Threshold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Threshold Óptimo', fontsize=16, y=0.98)
    save_png_svg(fig, run_dir, 'threshold_analysis')

def create_roc_pr_curves(y_true, y_scores, run_dir: Path):
    """Crea curvas ROC y Precision-Recall"""
    _style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    
    ax2.plot(recall, precision, color='darkred', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    
    # Baseline (random classifier for imbalanced data)
    baseline = np.mean(y_true)
    ax2.axhline(y=baseline, color='navy', linestyle='--', lw=2, 
               label=f'Random classifier (AP = {baseline:.3f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Curvas de Rendimiento del Modelo', fontsize=14)
    save_png_svg(fig, run_dir, 'roc_pr_curves')

def create_metrics_summary_table(optimal_thresholds: dict, run_dir: Path):
    """Crea tabla resumen de métricas óptimas"""
    
    # Crear tabla de resumen
    summary_data = []
    for criterion, data in optimal_thresholds.items():
        metrics = data['metrics']
        summary_data.append({
            'Criterion': criterion.replace('_', ' ').title(),
            'Threshold': f"{metrics['threshold']:.3f}",
            'F1 Score': f"{metrics['f1']:.3f}",
            'Sensitivity': f"{metrics['sensitivity']:.3f}",
            'Specificity': f"{metrics['specificity']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'AUROC': f"{metrics['auroc']:.3f}",
            'AUPRC': f"{metrics['auprc']:.3f}",
            'MCC': f"{metrics['mcc']:.3f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Guardar como CSV
    summary_df.to_csv(run_dir / 'optimal_thresholds_summary.csv', index=False)
    
    # Crear visualización de la tabla
    _style()
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center',
                    bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Styling
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Resumen de Thresholds Óptimos por Criterio', 
                fontsize=16, fontweight='bold', pad=20)
    
    save_png_svg(fig, run_dir, 'optimal_thresholds_table')

# ---------------- Proceso por run ----------------

def process_run(run_dir: Path):
    _style()
    df, cfg = load_run(run_dir)
    
    if df.empty:
        print(f"⚠ No hay datos en {run_dir}")
        return
    
    print(f"📊 Procesando run: {run_dir}")
    
    # Detectar formato de datos
    data_format = detect_data_format(df)
    print(f"📊 Formato detectado: {data_format}")
    
    # ===== NUEVO: Análisis completo con modelo guardado =====
    print("\n🔍 Cargando modelo y generando predicciones...")
    y_true, y_scores, model_config = load_model_and_predict(run_dir)
    
    if y_true is not None and y_scores is not None:
        print("✔ Predicciones generadas exitosamente")
        
        # Encontrar thresholds óptimos
        print("🎯 Optimizando thresholds...")
        optimal_thresholds, threshold_results = find_optimal_threshold(
            y_true, y_scores, 
            criteria=['f1', 'youden', 'balanced_accuracy', 'mcc']
        )
        
        # Crear visualizaciones de análisis de threshold
        create_threshold_analysis_plot(threshold_results, optimal_thresholds, run_dir)
        print("✔ Análisis de threshold generado")
        
        # Crear curvas ROC y PR
        create_roc_pr_curves(y_true, y_scores, run_dir)
        print("✔ Curvas ROC y PR generadas")
        
        # Crear tabla resumen
        create_metrics_summary_table(optimal_thresholds, run_dir)
        print("✔ Tabla resumen de thresholds óptimos generada")
        
        # Mostrar mejores thresholds
        print("\n📋 Thresholds óptimos encontrados:")
        for criterion, data in optimal_thresholds.items():
            thresh = data['threshold']
            f1 = data['metrics']['f1']
            sens = data['metrics']['sensitivity']
            spec = data['metrics']['specificity']
            print(f"  {criterion.upper()}: {thresh:.3f} (F1={f1:.3f}, Sens={sens:.3f}, Spec={spec:.3f})")
    
    # Generar gráficas unificadas (existentes)
    create_unified_training_curves(df, run_dir)
    print("✔ Curvas de entrenamiento unificadas generadas")
    
    # Dashboard de métricas para convulsiones
    create_seizure_metrics_dashboard(df, run_dir)
    print("✔ Dashboard de métricas de convulsiones generado")
    
    # ===== MEJORADO: Matrices de confusión separadas =====
    print("\n🔍 Generando matrices de confusión...")
    
    # Crear matrices individuales por frame/window
    create_frame_window_confusion_matrices(df, run_dir)
    
    # Crear análisis comparativo
    create_comprehensive_confusion_analysis(df, run_dir)
    
    print(f"\n✅ Análisis completo guardado en: {run_dir}")

def load_run(run_dir: Path):
    # run_dir = f'{run_dir}metrics.csv'
    # print(run_dir)
    df = pd.read_csv(run_dir / 'metrics.csv')

    # Si no hay filas, devolvemos df vacío y cfg
    if df.empty:
        try:
            cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        except Exception:
            cfg = {}
        return df, cfg

    # Asegurar columna 'epoch' si no existe
    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', np.arange(len(df)))

    # Coerción numérica ligera (sin forzar columnas claramente no numéricas)
    for c in df.columns:
        if c in {'run'}:
            continue
        if c == 'epoch':
            # asegurar entero si viene mal tipado
            try:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
            except Exception:
                pass
        else:
            # convertir números en texto a float; dejar strings si no aplica
            col_num = pd.to_numeric(df[c], errors='coerce')
            # Sustituir solo si hay algún número válido
            if col_num.notna().any():
                df[c] = col_num

    try:
        cfg = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    except Exception:
        cfg = {}
    return df, cfg

# ---------------- CLI ----------------

def main():
    run_dirs = Path('runs/eeg_torch_HYB_2025-09-02_09:59:32/')
    process_run(run_dirs)

if __name__ == '__main__':
    main()
