import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# ====================================================================================
# TRANSFORMER OPTIMIZED FOR EEG SEIZURE DETECTION
# ====================================================================================

class PositionalEncoding1D(nn.Module):
    """Positional encoding optimizado para señales EEG temporales"""
    def __init__(self, embed_dim: int, max_len: int = 5000, temperature: float = 10000.0):
        super().__init__()
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(temperature) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (no gradient, saved with model)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, embed_dim)
    
    def forward(self, x):  # x: (B, T, C)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SE1d(nn.Module):
    """Squeeze-and-Excitation para características EEG"""
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


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention optimizada para EEG"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):  # x: (B, T, C)
        B, T, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        
        # Apply mask if provided (for padding)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        
        # Final projection
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out, attn_weights


class FeedForward(nn.Module):
    """Feed-Forward Network optimizada para EEG"""
    def __init__(self, embed_dim: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # GELU mejor que ReLU para Transformers
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    """Bloque Transformer Encoder optimizado para detección de convulsiones"""
    def __init__(self, 
                 embed_dim: int, 
                 num_heads: int, 
                 mlp_dim: int, 
                 dropout: float = 0.1,
                 use_se: bool = False,
                 se_ratio: int = 16):
        super().__init__()
        
        # Multi-Head Self-Attention
        self.mhsa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Feed-Forward Network
        self.ffn = FeedForward(embed_dim, mlp_dim, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Optional SE block
        self.se = SE1d(embed_dim, se_ratio) if use_se else None
        
        # Stochastic depth for better training (optional)
        self.drop_path = nn.Identity()  # Can be replaced with DropPath
        
    def forward(self, x, mask=None):  # x: (B, T, C)
        # Multi-Head Self-Attention with residual connection
        attn_out, attn_weights = self.mhsa(self.norm1(x), mask)
        x = x + self.drop_path(attn_out)
        
        # Optional SE attention
        if self.se is not None:
            x = self.se(x)
        
        # Feed-Forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop_path(ffn_out)
        
        return x, attn_weights


class OptimizedSeizureTransformer(nn.Module):
    """Transformer optimizado para detección de convulsiones EEG"""
    
    def __init__(self,
                 input_dim: int = 22,                    # Canales EEG
                 seq_len: int = 256,                     # Longitud de secuencia
                 num_classes: int = 2,                   # Normal/Convulsión
                 embed_dim: int = 128,                   # Dimensión embedding
                 num_layers: int = 4,                    # Número de capas transformer
                 num_heads: int = 8,                     # Número de cabezas atención
                 mlp_dim: int = 256,                     # Dimensión FFN
                 dropout_rate: float = 0.1,
                 time_step_classification: bool = True,  # Frame-by-frame vs window
                 one_hot: bool = True,                   # Softmax vs sigmoid
                 use_se: bool = True,                    # Squeeze-and-Excitation
                 se_ratio: int = 16,
                 class_weights: list = None,             # Pesos para clases desbalanceadas
                 use_cls_token: bool = False,            # Token CLS para clasificación
                 enhanced_pos_enc: bool = True,          # Positional encoding mejorado
                 use_layer_scale: bool = False):         # Layer scaling para entrenamiento
        
        super().__init__()
        self.time_step_classification = time_step_classification
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, embed_dim)
        
        # Optional SE on input features
        if use_se:
            self.input_se = SE1d(input_dim, se_ratio=se_ratio)
        else:
            self.input_se = None
        
        # Positional encoding
        if enhanced_pos_enc:
            self.pos_encoding = PositionalEncoding1D(embed_dim, max_len=seq_len + 1)
        else:
            # Simple learnable positional embeddings
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, embed_dim) * 0.02)
            self.pos_encoding = None
        
        # CLS token for classification (alternative to pooling)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Transformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout_rate,
                use_se=use_se and (i == 0 or i == num_layers-1),  # SE en primera y última capa
                se_ratio=se_ratio
            ) for i in range(num_layers)
        ])
        
        # Pre-norm para mejor estabilidad
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        if time_step_classification:
            # Frame-by-frame classification
            if one_hot:
                self.classifier = nn.Linear(embed_dim, num_classes)
            else:
                self.classifier = nn.Linear(embed_dim, 1)
        else:
            # Window-level classification
            if not use_cls_token:
                self.global_pool = nn.AdaptiveAvgPool1d(1)
            
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim // 2, num_classes if one_hot else 1)
            )
        
        # Class weights para datasets desbalanceados
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización optimizada para Transformers"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x, mask=None):  # x: (B, T, C)
        B, T, C = x.shape
        
        if C != self.input_dim:
            raise ValueError(f"Input dim {C} != expected {self.input_dim}")
        
        # Optional SE on input features
        if self.input_se is not None:
            x = self.input_se(x)
        
        # Project to embedding dimension
        x = self.input_projection(x)  # (B, T, embed_dim)
        
        # Add CLS token if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, T+1, embed_dim)
            if mask is not None:
                # Extend mask for CLS token
                cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Add positional encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        else:
            seq_len = x.size(1)
            x = x + self.pos_embedding[:, :seq_len, :]
        
        x = self.input_dropout(x)
        
        # Pass through transformer encoder blocks
        attention_weights = []
        for encoder_block in self.encoder_blocks:
            x, attn_weights = encoder_block(x, mask)
            attention_weights.append(attn_weights)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Classification
        if self.time_step_classification:
            # Frame-by-frame prediction
            if self.use_cls_token:
                # Remove CLS token for time-step prediction
                x = x[:, 1:, :]  # (B, T, embed_dim)
            
            logits = self.classifier(x)  # (B, T, num_classes or 1)
            
            if self.one_hot:
                outputs = F.softmax(logits, dim=-1)
            else:
                outputs = torch.sigmoid(logits)
        else:
            # Window-level prediction
            if self.use_cls_token:
                # Use CLS token representation
                x = x[:, 0, :]  # (B, embed_dim)
            else:
                # Global average pooling
                if self.use_cls_token:
                    x = x[:, 1:, :]  # Remove CLS token first
                x = x.permute(0, 2, 1)  # (B, embed_dim, T)
                x = self.global_pool(x).squeeze(-1)  # (B, embed_dim)
            
            logits = self.classifier(x)  # (B, num_classes or 1)
            
            if self.one_hot:
                outputs = F.softmax(logits, dim=-1)
            else:
                outputs = torch.sigmoid(logits)
        
        return outputs
    
    def compute_loss(self, logits, targets, mask=None):
        """Función de pérdida optimizada para detección de convulsiones"""
        if self.time_step_classification:
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
                    # Apply mask (exclude padded positions)
                    if self.use_cls_token and mask.size(1) == logits.size(1) + 1:
                        mask = mask[:, 1:]  # Remove CLS mask
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
    
    def get_attention_weights(self, x, mask=None):
        """Obtener pesos de atención para visualización"""
        attention_weights = []
        
        # Forward pass storing attention weights
        with torch.no_grad():
            B, T, C = x.shape
            
            if self.input_se is not None:
                x = self.input_se(x)
            
            x = self.input_projection(x)
            
            if self.use_cls_token:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)
                if mask is not None:
                    cls_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
                    mask = torch.cat([cls_mask, mask], dim=1)
            
            if self.pos_encoding is not None:
                x = self.pos_encoding(x)
            else:
                seq_len = x.size(1)
                x = x + self.pos_embedding[:, :seq_len, :]
            
            x = self.input_dropout(x)
            
            for encoder_block in self.encoder_blocks:
                x, attn_weights = encoder_block(x, mask)
                attention_weights.append(attn_weights.cpu().numpy())
        
        return attention_weights

# ====================================================================================
# FACTORY FUNCTIONS FOR SEIZURE DETECTION
# ====================================================================================

def create_transformer_seizure_model(input_channels=22, seq_length=256, **kwargs):
    """Factory function para crear Transformer optimizado para convulsiones"""
    
    # Configuración recomendada para detección de convulsiones EEG
    default_config = {
        'input_dim': input_channels,
        'seq_len': seq_length,
        'num_classes': 2,
        'embed_dim': 128,  # Dimensión moderada para EEG
        'num_layers': 6,   # Suficientes capas para capturar dependencias temporales
        'num_heads': 8,    # Múltiples cabezas para diferentes patrones
        'mlp_dim': 256,    # FFN de tamaño moderado
        'dropout_rate': 0.15,  # Dropout moderado
        'time_step_classification': True,
        'one_hot': True,
        'use_se': True,    # SE para enfoque en características importantes
        'se_ratio': 8,     # SE más fuerte
        'class_weights': [1.0, 20.0],  # Peso muy alto para convulsiones
        'use_cls_token': False,  # Sin CLS para time-step classification
        'enhanced_pos_enc': True,
        'use_layer_scale': False
    }
    
    # Actualizar con parámetros proporcionados
    default_config.update(kwargs)
    
    model = OptimizedSeizureTransformer(**default_config)
    
    # Información del modelo
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧠 OptimizedSeizureTransformer creada:")
    print(f"├── Arquitectura: Transformer Encoder")
    print(f"├── Parámetros totales: {total_params:,}")
    print(f"├── Parámetros entrenables: {trainable_params:,}")
    print(f"├── Canales entrada: {input_channels}")
    print(f"├── Longitud secuencia: {seq_length}")
    print(f"├── Embed dim: {default_config['embed_dim']}")
    print(f"├── Capas: {default_config['num_layers']}")
    print(f"├── Cabezas atención: {default_config['num_heads']}")
    print(f"├── SE activado: {default_config['use_se']}")
    print(f"└── Modo: {'Frame-by-frame' if default_config['time_step_classification'] else 'Por ventana'}")
    
    return model


def create_deep_transformer(input_channels=22, seq_length=256, **kwargs):
    """Transformer profundo para máximo rendimiento (cuando hay datos suficientes)"""
    deep_config = {
        'input_dim': input_channels,
        'seq_len': seq_length,
        'num_classes': 2,
        'embed_dim': 256,   # Mayor dimensión
        'num_layers': 12,   # Más capas
        'num_heads': 16,    # Más cabezas
        'mlp_dim': 512,     # FFN más grande
        'dropout_rate': 0.2,  # Mayor dropout para regularización
        'use_se': True,
        'se_ratio': 4,      # SE muy fuerte
        'class_weights': [1.0, 25.0],  # Peso muy alto para convulsiones
        'use_layer_scale': True
    }
    deep_config.update(kwargs)
    return OptimizedSeizureTransformer(**deep_config)
