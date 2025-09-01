import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ====================================================================================
# HYBRID DSCNN + RNN OPTIMIZED FOR EEG SEIZURE DETECTION
# ====================================================================================

class SE1d(nn.Module):
    """Squeeze-and-Excitation optimizado para señales EEG"""
    def __init__(self, channels: int, se_ratio: int = 16):
        super().__init__()
        hidden = max(1, channels // se_ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        # Global average pooling along time dimension
        s = x.mean(dim=1)  # (B, C)
        s = F.relu(self.fc1(s))
        s = self.dropout(s)
        s = torch.sigmoid(self.fc2(s))
        s = s.unsqueeze(1)  # (B, 1, C)
        return x * s


class MultiHeadAttention1D(nn.Module):
    """Multi-Head Attention optimizada para secuencias temporales EEG"""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        
        # Self-attention (query, key, value all from x)
        Q = self.w_q(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        K = self.w_k(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        V = self.w_v(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, V)  # (B, H, T, d_k)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        
        return self.w_o(attn_output)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise Separable Convolution que maneja cualquier número de canales"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False):
        super().__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                  padding=padding, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm1d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn_pw = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = F.relu(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
        return x


class OptimizedHybridSeizureModel(nn.Module):
    """Hybrid DSCNN + RNN optimizada para detección de convulsiones EEG"""
    
    def __init__(self,
                 input_dim: int = 22,                    # Canales EEG
                 num_classes: int = 2,                   # Normal/Convulsión
                 dropout_rate: float = 0.25,
                 one_hot: bool = True,                   # Softmax vs sigmoid
                 time_step: bool = True,                 # Frame-by-frame vs window
                 se_position: str = 'after_conv',        # None, 'after_conv', 'after_fc'
                 attention_position: str = 'final',      # None, 'between_lstm', 'final'
                 rnn_type: str = 'lstm',                 # 'lstm' or 'gru'
                 se_ratio: int = 16,
                 num_heads: int = 4,
                 pool_size_if_ts: int = 1,              # Pool size for time-step mode
                 pool_size_if_win: int = 2,             # Pool size for window mode
                 class_weights: list = None,             # For imbalanced datasets
                 use_layer_norm: bool = True,            # Better for EEG
                 enhanced_rnn: bool = True,              # Bidirectional + larger hidden
                 use_depthwise_separable: bool = True):  # Use depthwise separable conv
        
        super().__init__()
        self.time_step = time_step
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.se_position = se_position
        self.attention_position = attention_position
        self.rnn_type = rnn_type.lower()
        
        # Pool size based on mode
        self.pool_size = pool_size_if_ts if time_step else pool_size_if_win
        
        # --- Convolutional Front-end ---
        if use_depthwise_separable:
            self.conv_front = DepthwiseSeparableConv1d(input_dim, 64, kernel_size=3, padding=1)
        else:
            # Standard convolution as fallback
            self.conv_front = nn.Sequential(
                nn.Conv1d(input_dim, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(64)
            )
        
        # SE after conv
        if se_position == 'after_conv':
            self.se_after_conv = SE1d(64, se_ratio=se_ratio)
        else:
            self.se_after_conv = None
        
        self.pool1 = nn.MaxPool1d(kernel_size=self.pool_size)
        
        # --- Dense layer (feature transformation) ---
        self.fc1 = nn.Linear(64, 256)
        self.ln1 = nn.LayerNorm(256) if use_layer_norm else nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # SE after fc
        if se_position == 'after_fc':
            self.se_after_fc = SE1d(256, se_ratio=se_ratio)
        else:
            self.se_after_fc = None
        
        # --- RNN Layers ---
        rnn_hidden = 128 if enhanced_rnn else 64  # Larger hidden size for seizures
        rnn_bidirectional = enhanced_rnn
        
        if self.rnn_type == 'gru':
            self.rnn1 = nn.GRU(256, rnn_hidden, batch_first=True, 
                              bidirectional=rnn_bidirectional, dropout=dropout_rate if enhanced_rnn else 0)
        else:
            self.rnn1 = nn.LSTM(256, rnn_hidden, batch_first=True,
                               bidirectional=rnn_bidirectional, dropout=dropout_rate if enhanced_rnn else 0)
        
        # Adjust dimensions for bidirectional
        rnn1_output_dim = rnn_hidden * (2 if rnn_bidirectional else 1)
        
        # Attention between RNNs
        if attention_position == 'between_lstm':
            self.mha_between = MultiHeadAttention1D(rnn1_output_dim, num_heads=num_heads, dropout=dropout_rate)
            self.ln_mha_between = nn.LayerNorm(rnn1_output_dim)
        else:
            self.mha_between = None
        
        # Second RNN
        return_seq_for_second = time_step or (attention_position == 'final')
        
        if self.rnn_type == 'gru':
            self.rnn2 = nn.GRU(rnn1_output_dim, rnn_hidden, batch_first=True,
                              bidirectional=rnn_bidirectional, dropout=dropout_rate if enhanced_rnn else 0)
        else:
            self.rnn2 = nn.LSTM(rnn1_output_dim, rnn_hidden, batch_first=True,
                               bidirectional=rnn_bidirectional, dropout=dropout_rate if enhanced_rnn else 0)
        
        rnn2_output_dim = rnn_hidden * (2 if rnn_bidirectional else 1)
        
        # Final attention
        if attention_position == 'final':
            self.mha_final = MultiHeadAttention1D(rnn2_output_dim, num_heads=num_heads, dropout=dropout_rate)
            self.ln_mha_final = nn.LayerNorm(rnn2_output_dim)
        else:
            self.mha_final = None
        
        # Global pooling for window mode when using final attention
        if attention_position == 'final' and not time_step:
            self.gap_final = nn.AdaptiveAvgPool1d(1)
        
        # --- Classification Head ---
        if time_step:
            # Time-distributed head
            self.head_hidden = nn.Linear(rnn2_output_dim, 64)
            self.head_dropout = nn.Dropout(dropout_rate)
            if one_hot:
                self.head_output = nn.Linear(64, num_classes)
            else:
                self.head_output = nn.Linear(64, 1)
        else:
            # Window-level head
            self.fc2 = nn.Linear(rnn2_output_dim, 64)
            self.head_dropout = nn.Dropout(dropout_rate)
            if one_hot:
                self.head_output = nn.Linear(64, num_classes)
            else:
                self.head_output = nn.Linear(64, 1)
        
        # Class weights for imbalanced datasets
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización optimizada para EEG"""
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
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Input dim {x.size(-1)} != expected {self.input_dim}")
        
        # --- Convolutional Front-end ---
        x = x.permute(0, 2, 1)  # (B, C, T) for conv1d
        x = self.conv_front(x)
        x = F.relu(x)
        x = x.permute(0, 2, 1)  # (B, T, C) back to sequence format
        
        # SE after conv
        if self.se_after_conv is not None:
            x = self.se_after_conv(x)
        
        # Pooling
        if self.pool_size > 1:
            x = x.permute(0, 2, 1)  # (B, C, T) for pooling
            x = self.pool1(x)
            x = x.permute(0, 2, 1)  # (B, T', C) back
        
        # --- Dense layer ---
        x = self.fc1(x)
        if isinstance(self.ln1, nn.LayerNorm):
            x = self.ln1(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.ln1(x)
            x = x.permute(0, 2, 1)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # SE after fc
        if self.se_after_fc is not None:
            x = self.se_after_fc(x)
        
        # --- RNN 1 ---
        if self.rnn_type == 'gru':
            x, _ = self.rnn1(x)
        else:
            x, _ = self.rnn1(x)
        
        # Attention between RNNs
        if self.mha_between is not None:
            attn_out = self.mha_between(x)
            x = x + attn_out  # Residual connection
            x = self.ln_mha_between(x)
        
        # --- RNN 2 ---
        if self.rnn_type == 'gru':
            if self.time_step or self.attention_position == 'final':
                x, _ = self.rnn2(x)  # Keep sequences
            else:
                # Take only last output for window mode
                _, hidden = self.rnn2(x)
                if isinstance(hidden, tuple):  # LSTM
                    x = hidden[0][-1]  # Last hidden state
                else:  # GRU
                    x = hidden[-1]
                # Handle bidirectional
                if self.rnn2.bidirectional:
                    x = x.view(x.size(0), 2, -1).mean(dim=1)  # Average forward and backward
        else:  # LSTM
            if self.time_step or self.attention_position == 'final':
                x, _ = self.rnn2(x)  # Keep sequences
            else:
                # Take only last output for window mode
                _, (hidden, _) = self.rnn2(x)
                x = hidden[-1]  # Last hidden state
                # Handle bidirectional
                if self.rnn2.bidirectional:
                    x = x.view(x.size(0), 2, -1).mean(dim=1)  # Average forward and backward
        
        # Final attention
        if self.mha_final is not None:
            attn_out = self.mha_final(x)
            x = x + attn_out  # Residual connection
            x = self.ln_mha_final(x)
            
            if not self.time_step:
                # Window mode: collapse time dimension
                x = x.permute(0, 2, 1)  # (B, C, T)
                x = self.gap_final(x).squeeze(-1)  # (B, C)
        
        # --- Classification Head ---
        if self.time_step:
            # Time-distributed processing
            x = self.head_hidden(x)
            x = F.relu(x)
            x = self.head_dropout(x)
            x = self.head_output(x)
            
            if self.one_hot:
                x = F.softmax(x, dim=-1)  # (B, T', C)
            else:
                x = torch.sigmoid(x)  # (B, T', 1)
        else:
            # Window-level processing
            x = self.fc2(x)
            x = F.relu(x)
            x = self.head_dropout(x)
            x = self.head_output(x)
            
            if self.one_hot:
                x = F.softmax(x, dim=-1)  # (B, C)
            else:
                x = torch.sigmoid(x)  # (B, 1)
        
        return x
    
    def compute_loss(self, logits, targets, mask=None):
        """Función de pérdida optimizada para detección de convulsiones"""
        if self.time_step:
            # Frame-by-frame loss
            if self.one_hot:
                if len(targets.shape) == 3:  # One-hot encoded
                    targets_labels = targets.argmax(dim=-1)
                else:
                    targets_labels = targets.long()
                
                if self.class_weights is not None:
                    criterion = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')
                else:
                    criterion = nn.CrossEntropyLoss(reduction='none')
                
                loss = criterion(logits.view(-1, self.num_classes), targets_labels.view(-1))
                loss = loss.view(logits.shape[0], logits.shape[1])
                
                if mask is not None:
                    loss = loss * mask
                    return loss.sum() / mask.sum()
                else:
                    return loss.mean()
            else:
                criterion = nn.BCELoss()
                return criterion(logits.squeeze(-1), targets.float())
        else:
            # Window-level loss
            if self.one_hot:
                if self.class_weights is not None:
                    criterion = nn.CrossEntropyLoss(weight=self.class_weights)
                else:
                    criterion = nn.CrossEntropyLoss()
                return criterion(logits, targets.long())
            else:
                criterion = nn.BCELoss()
                return criterion(logits.squeeze(-1), targets.float())
    
    def predict_proba(self, x):
        """Predicción de probabilidades"""
        with torch.no_grad():
            return self.forward(x)

# ====================================================================================
# FACTORY FUNCTIONS FOR SEIZURE DETECTION
# ====================================================================================

def create_hybrid_seizure_model(input_channels=22, window_length_samples=256, **kwargs):
    """Factory function para crear modelo híbrido optimizado para convulsiones"""
    
    # Configuración recomendada para detección de convulsiones
    default_config = {
        'input_dim': input_channels,
        'num_classes': 2,
        'dropout_rate': 0.3,  # Mayor dropout para regularización
        'one_hot': True,
        'time_step': True,
        'se_position': 'after_conv',  # SE después de conv para mejor extracción
        'attention_position': 'final',  # Atención final para capturar patrones críticos
        'rnn_type': 'lstm',  # LSTM mejor para secuencias largas EEG
        'se_ratio': 8,  # SE más fuerte para EEG
        'num_heads': 8,  # Más cabezas de atención
        'pool_size_if_ts': 1,  # Sin downsampling temporal
        'pool_size_if_win': 2,
        'class_weights': [1.0, 15.0],  # Peso muy alto para convulsiones
        'use_layer_norm': True,  # LayerNorm mejor para EEG
        'enhanced_rnn': True  # RNN bidireccional + mayor hidden size
    }
    
    # Actualizar con parámetros proporcionados
    default_config.update(kwargs)
    
    model = OptimizedHybridSeizureModel(**default_config)
    
    # Información del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧠 OptimizedHybridSeizureModel creada:")
    print(f"├── Arquitectura: DSCNN + {default_config['rnn_type'].upper()} + Attention")
    print(f"├── Parámetros totales: {total_params:,}")
    print(f"├── Parámetros entrenables: {trainable_params:,}")
    print(f"├── Canales entrada: {input_channels}")
    print(f"├── SE position: {default_config['se_position']}")
    print(f"├── Attention position: {default_config['attention_position']}")
    print(f"├── Enhanced RNN: {default_config['enhanced_rnn']}")
    print(f"└── Modo: {'Frame-by-frame' if default_config['time_step'] else 'Por ventana'}")
    
    return model

def create_lightweight_hybrid(input_channels=22, **kwargs):
    """Versión ligera del modelo híbrido para deployment en tiempo real"""
    lightweight_config = {
        'input_dim': input_channels,
        'num_classes': 2,
        'dropout_rate': 0.2,
        'se_position': None,  # Sin SE para mayor velocidad
        'attention_position': None,  # Sin atención para mayor velocidad
        'rnn_type': 'gru',  # GRU más rápido que LSTM
        'enhanced_rnn': False,  # RNN más simple
        'class_weights': [1.0, 10.0]
    }
    lightweight_config.update(kwargs)
    return OptimizedHybridSeizureModel(**lightweight_config)

def create_robust_hybrid(input_channels=22, **kwargs):
    """Versión robusta que maneja cualquier número de canales de entrada"""
    robust_config = {
        'input_dim': input_channels,
        'num_classes': 2,
        'dropout_rate': 0.25,
        'one_hot': True,
        'time_step': True,
        'se_position': 'after_fc',  # SE después de FC para evitar problemas de dimensionalidad
        'attention_position': 'final',
        'rnn_type': 'lstm',
        'se_ratio': 16,
        'num_heads': 4,
        'pool_size_if_ts': 1,
        'pool_size_if_win': 2,
        'class_weights': [1.0, 15.0],
        'use_layer_norm': True,
        'enhanced_rnn': True
    }
    robust_config.update(kwargs)
    return OptimizedHybridSeizureModel(**robust_config)


