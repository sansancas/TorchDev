import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ====================================================================================
# TCN OPTIMIZADA PARA DETECCIÓN DE CONVULSIONES EEG
# ====================================================================================

class CausalConv1d(nn.Module):
    """Convolución causal mejorada con padding adaptativo"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=0, dilation=dilation, bias=bias)

    def forward(self, x):  # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class SepConv1d(nn.Module):
    """SeparableConv1D optimizada para EEG con mejor eficiencia"""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=False, depth_multiplier=1):
        super().__init__()
        self.pad = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(in_ch, in_ch*depth_multiplier, kernel_size,
                                   groups=in_ch, dilation=dilation, padding=0, bias=bias)
        self.pointwise = nn.Conv1d(in_ch*depth_multiplier, out_ch, 1, bias=bias)
        # Mejora: BatchNorm para estabilidad en EEG
        self.bn_dw = nn.BatchNorm1d(in_ch*depth_multiplier)
        self.bn_pw = nn.BatchNorm1d(out_ch)

    def forward(self, x):                # x: (B, C, T)
        x = F.pad(x, (self.pad, 0))     # padding causal
        x = self.depthwise(x)
        x = self.bn_dw(x)               # BatchNorm después de depthwise
        x = F.relu(x)                   # Activación
        x = self.pointwise(x)
        x = self.bn_pw(x)               # BatchNorm después de pointwise
        return x


class SE1d(nn.Module):
    """Squeeze-and-Excitation optimizado para señales EEG"""
    def __init__(self, channels: int, se_ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // se_ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.dropout = nn.Dropout(0.1)  # Regularización adicional

    def forward(self, x):  # x: (B, C, T)
        s = self.pool(x).squeeze(-1)
        s = F.relu(self.fc1(s))
        s = self.dropout(s)  # Dropout para regularización
        s = torch.sigmoid(self.fc2(s))
        s = s.unsqueeze(-1)
        return x * s


class EnhancedTCNBlock(nn.Module):
    """Bloque TCN mejorado para detección de convulsiones EEG"""
    def __init__(self, channels: int, kernel_size: int, dilation: int,
                 dropout: float = 0.25, separable: bool = True, use_se: bool = True,
                 use_residual_scaling: bool = True):
        super().__init__()
        Conv = SepConv1d if separable else CausalConv1d
        
        # Capas convolutivas mejoradas
        self.conv1 = Conv(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.ln1 = nn.LayerNorm(channels)
        self.drop1 = nn.Dropout(dropout)
        
        self.conv2 = Conv(channels, channels, kernel_size, dilation=dilation, bias=False)
        self.ln2 = nn.LayerNorm(channels)
        self.drop2 = nn.Dropout(dropout)
        
        # Squeeze-and-Excitation para atención en canales
        self.se = SE1d(channels) if use_se else None
        
        # Residual scaling para mejor gradientes
        self.use_residual_scaling = use_residual_scaling
        if use_residual_scaling:
            self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # Projection layer si hay cambio de receptive field significativo
        self.projection = None
        if dilation > 4:  # Para dilataciones altas, proyección para mejor residual
            self.projection = nn.Conv1d(channels, channels, 1, bias=False)

    def forward(self, x):  # x: (B, T, C)
        residual = x
        
        # Primera convolución
        x = x.permute(0,2,1)  # (B, C, T)
        x = self.conv1(x)
        x = x.permute(0,2,1)  # (B, T, C)
        x = self.ln1(x)
        x = F.gelu(x)  # GELU en lugar de ReLU para mejor rendimiento
        x = self.drop1(x)
        
        # Segunda convolución
        x = x.permute(0,2,1)  # (B, C, T)
        x = self.conv2(x)
        x = x.permute(0,2,1)  # (B, T, C)
        x = self.ln2(x)
        x = F.gelu(x)
        x = self.drop2(x)
        
        # Squeeze-and-Excitation
        if self.se is not None:
            x = x.permute(0,2,1)  # (B, C, T)
            x = self.se(x)
            x = x.permute(0,2,1)  # (B, T, C)
        
        # Ajustar longitudes para residual connection
        if x.size(1) != residual.size(1):
            T = min(x.size(1), residual.size(1))
            x = x[:, :T, :]
            residual = residual[:, :T, :]
        
        # Proyección del residual si es necesario
        if self.projection is not None:
            residual = residual.permute(0,2,1)
            residual = self.projection(residual)
            residual = residual.permute(0,2,1)
        
        # Conexión residual con escalado
        if self.use_residual_scaling:
            return residual + self.residual_scale * x
        else:
            return residual + x


class OptimizedSeizureTCN(nn.Module):
    """TCN optimizada específicamente para detección de convulsiones EEG"""
    
    def __init__(self, 
                 input_dim: int = 22,          # Canales EEG (22 para montaje ar unificado)
                 num_classes: int = 2,         # 2 para binario (normal/convulsión)
                 num_filters: int = 96,        # Incrementado para mejor capacidad
                 kernel_size: int = 7,         # Kernel más grande para patrones EEG
                 num_blocks: int = 8,          # Optimizado para balance complejidad/performance
                 time_step: bool = True,       # Frame-by-frame prediction
                 one_hot: bool = True,         # Compatible con dataset
                 separable: bool = True,       # Eficiencia computacional
                 dropout: float = 0.3,         # Incrementado para regularización
                 use_se: bool = True,          # Atención en canales
                 use_residual_scaling: bool = True,  # Mejor entrenamiento
                 use_adaptive_dilation: bool = True, # Dilataciones adaptativas
                 use_multiscale: bool = True,  # Características multi-escala
                 class_weights: list = [None]):  # Pesos para clases desbalanceadas
        
        super().__init__()
        self.time_step = time_step
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.use_multiscale = use_multiscale
        
        # Proyección de entrada mejorada con normalización
        self.in_fc = nn.Sequential(
            nn.Linear(input_dim, num_filters),
            nn.LayerNorm(num_filters),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Bloques TCN con dilataciones adaptativas
        blocks = []
        if use_adaptive_dilation:
            # Dilataciones que crecen más lentamente para mejor captura temporal
            dilations = [2**min(i, 6) for i in range(num_blocks)]  # Cap at 64
        else:
            dilations = [2**i for i in range(num_blocks)]
        
        for i, dilation in enumerate(dilations):
            blocks.append(EnhancedTCNBlock(
                channels=num_filters,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
                separable=separable,
                use_se=use_se,
                use_residual_scaling=use_residual_scaling
            ))
        
        self.blocks = nn.Sequential(*blocks)
        
        # Características multi-escala opcionales
        if use_multiscale:
            self.multiscale_conv = nn.ModuleList([
                nn.Conv1d(num_filters, num_filters//4, kernel_size=k, padding=k//2)
                for k in [3, 5, 7, 11]  # Diferentes escalas temporales
            ])
            self.multiscale_fusion = nn.Conv1d(num_filters, num_filters, 1)
        
        # Cabeza de clasificación mejorada
        if time_step:
            # Predicción frame-by-frame
            self.head = nn.Sequential(
                nn.Linear(num_filters, num_filters//2),
                nn.LayerNorm(num_filters//2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(num_filters//2, num_classes if one_hot else 1)
            )
        else:
            # Predicción por ventana completa
            self.gap = nn.AdaptiveAvgPool1d(1)
            self.gmp = nn.AdaptiveMaxPool1d(1)  # Pool adicional para captar picos
            self.head = nn.Sequential(
                nn.Linear(num_filters * 2, num_filters),  # *2 por avg+max pooling
                nn.LayerNorm(num_filters),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(num_filters, num_classes if one_hot else 1)
            )
        
        # Inicialización de pesos mejorada
        self._initialize_weights()
        
        # Registrar pesos de clase para loss balanceado
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None

    def _initialize_weights(self):
        """Inicialización de pesos optimizada para EEG"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  # x: (B, T, C) donde C = input_dim (canales EEG)
        # Verificar dimensiones de entrada
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Input dim {x.size(-1)} != expected {self.input_dim}")
        
        # Proyección de entrada
        x = self.in_fc(x)  # (B, T, num_filters)
        
        # Bloques TCN
        x = self.blocks(x)  # (B, T, num_filters)
        
        # Características multi-escala opcionales
        if self.use_multiscale:
            x_ch_first = x.permute(0, 2, 1)  # (B, num_filters, T)
            multiscale_features = []
            for conv in self.multiscale_conv:
                ms_feat = F.relu(conv(x_ch_first))
                multiscale_features.append(ms_feat)
            
            # Concatenar y fusionar
            ms_concat = torch.cat(multiscale_features, dim=1)  # (B, num_filters, T)
            ms_fused = self.multiscale_fusion(ms_concat)
            x_ch_first = x_ch_first + ms_fused  # Residual connection
            x = x_ch_first.permute(0, 2, 1)  # (B, T, num_filters)
        
        # Predicción
        if self.time_step:
            # Frame-by-frame
            logits = self.head(x)  # (B, T, num_classes or 1)
            return logits
        else:
            # Por ventana completa
            x_ch_first = x.permute(0, 2, 1)  # (B, num_filters, T)
            
            # Doble pooling para capturar tanto promedios como picos
            x_avg = self.gap(x_ch_first).squeeze(-1)  # (B, num_filters)
            x_max = self.gmp(x_ch_first).squeeze(-1)  # (B, num_filters)
            x_combined = torch.cat([x_avg, x_max], dim=1)  # (B, num_filters*2)
            
            logits = self.head(x_combined)  # (B, num_classes or 1)
            return logits
        
    def compute_loss(self, logits, targets, mask=None):
        """Función de pérdida optimizada para detección de convulsiones"""
        if self.time_step:
            # Loss frame-by-frame
            if self.one_hot:
                if len(targets.shape) == 3:  # (B, T, C) one-hot
                    targets_labels = targets.argmax(dim=-1)  # (B, T)
                else:
                    targets_labels = targets.long()  # (B, T)
                
                # Cross-entropy con pesos de clase
                if self.class_weights is not None:
                    criterion = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
                else:
                    criterion = nn.CrossEntropyLoss(reduction='none')
                
                loss = criterion(logits.view(-1, self.num_classes), targets_labels.view(-1))
                loss = loss.view(logits.shape[0], logits.shape[1])  # (B, T)
                
                # Aplicar máscara si se proporciona
                if mask is not None:
                    loss = loss * mask
                    return loss.sum() / mask.sum()
                else:
                    return loss.mean()
            else:
                # Regresión binaria
                criterion = nn.BCEWithLogitsLoss()
                return criterion(logits.squeeze(-1), targets.float())
        else:
            # Loss por ventana
            if self.one_hot:
                if self.class_weights is not None:
                    criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                else:
                    criterion = nn.CrossEntropyLoss()
                return criterion(logits, targets.long())
            else:
                criterion = nn.BCEWithLogitsLoss()
                return criterion(logits.squeeze(-1), targets.float())
    
    def predict_proba(self, x):
        """Predicción de probabilidades"""
        with torch.no_grad():
            logits = self.forward(x)
            if self.one_hot:
                return F.softmax(logits, dim=-1)
            else:
                return torch.sigmoid(logits)
    
    def get_receptive_field(self):
        """Calcula el campo receptivo total de la red"""
        rf = 1
        for i in range(len(self.blocks)):
            dilation = 2**min(i, 6)  # Ajustar según adaptive_dilation
            rf += (7 - 1) * dilation  # kernel_size - 1
        return rf
    
# ====================================================================================
# FUNCIONES DE UTILIDAD PARA DETECCIÓN DE CONVULSIONES
# ====================================================================================

def create_seizure_tcn(input_channels=22, window_length_samples=1280, **kwargs):
    """Factory function para crear TCN optimizada para convulsiones"""
    
    # Configuración recomendada para EEG de convulsiones
    default_config = {
        'input_dim': input_channels,
        'num_classes': 2,
        'num_filters': 96,
        'kernel_size': 7,
        'num_blocks': 8,
        'time_step': True,
        'one_hot': True,
        'dropout': 0.3,
        'use_se': True,
        'use_multiscale': True,
        'class_weights': [1.0, 10.0]  # Peso mayor para convulsiones (clase minoritaria)
    }
    
    # Actualizar con parámetros proporcionados
    default_config.update(kwargs)
    
    model = OptimizedSeizureTCN(**default_config)
    
    # Imprimir información del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    receptive_field = model.get_receptive_field()
    
    print(f"🧠 OptimizedSeizureTCN creada:")
    print(f"├── Parámetros totales: {total_params:,}")
    print(f"├── Parámetros entrenables: {trainable_params:,}")
    print(f"├── Campo receptivo: {receptive_field} muestras ({receptive_field/256:.2f}s)")
    print(f"├── Canales entrada: {input_channels}")
    print(f"├── Longitud ventana: {window_length_samples} muestras ({window_length_samples/256:.1f}s)")
    print(f"└── Modo: {'Frame-by-frame' if default_config['time_step'] else 'Por ventana'}")
    
    return model