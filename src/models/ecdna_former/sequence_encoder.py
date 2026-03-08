"""
Sequence Encoder for ecDNA-Former.

Encodes DNA sequences using pre-trained DNA language models.
Supports:
- Nucleotide Transformer (recommended)
- DNABERT-2
- Custom CNN encoder (for fast inference)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class SequenceEncoder(nn.Module):
    """
    DNA sequence encoder using pre-trained language models.

    Encodes genomic context around candidate ecDNA regions using
    DNA foundation models for rich sequence representations.
    """

    def __init__(
        self,
        model_name: str = "nucleotide_transformer",
        pretrained: bool = True,
        hidden_dim: int = 256,
        output_dim: int = 256,
        max_length: int = 6000,
        freeze_encoder: bool = True,
        use_pooling: str = "mean",
    ):
        """
        Initialize sequence encoder.

        Args:
            model_name: Pre-trained model to use
            pretrained: Whether to load pre-trained weights
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            max_length: Maximum sequence length
            freeze_encoder: Whether to freeze pre-trained weights
            use_pooling: Pooling strategy ("mean", "cls", "attention")
        """
        super().__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_length = max_length
        self.use_pooling = use_pooling

        # Initialize encoder
        if model_name == "nucleotide_transformer" and pretrained:
            self.encoder = self._load_nucleotide_transformer()
            self.encoder_dim = 1024  # NT output dim
        elif model_name == "dnabert2" and pretrained:
            self.encoder = self._load_dnabert2()
            self.encoder_dim = 768
        else:
            # Use custom CNN encoder
            self.encoder = DNACNNEncoder(
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )
            self.encoder_dim = hidden_dim

        # Freeze encoder if specified
        if freeze_encoder and pretrained:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(self.encoder_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Attention pooling
        if use_pooling == "attention":
            self.attention_pool = AttentionPooling(output_dim)

        # Circular-aware positional encoding
        self.circular_pe = CircularPositionalEncoding(output_dim, max_length)

    def _load_nucleotide_transformer(self) -> nn.Module:
        """Load Nucleotide Transformer model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            # Use smaller variant for efficiency
            model_id = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            return model
        except ImportError:
            print("Transformers not available, using CNN encoder")
            return DNACNNEncoder(hidden_dim=self.hidden_dim, output_dim=1024)

    def _load_dnabert2(self) -> nn.Module:
        """Load DNABERT-2 model."""
        try:
            from transformers import AutoModel, AutoTokenizer

            model_id = "zhihan1996/DNABERT-2-117M"
            model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            return model
        except ImportError:
            print("Transformers not available, using CNN encoder")
            return DNACNNEncoder(hidden_dim=self.hidden_dim, output_dim=768)

    def forward(
        self,
        sequences: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
        is_circular: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode DNA sequences.

        Args:
            sequences: Input sequences [batch, seq_len] or embeddings [batch, seq_len, dim]
            sequence_mask: Attention mask [batch, seq_len]
            is_circular: Whether each sequence is circular [batch]

        Returns:
            Tuple of:
                - Sequence embeddings [batch, seq_len, output_dim]
                - Pooled embedding [batch, output_dim]
        """
        # Get encoder outputs
        if hasattr(self.encoder, 'config'):
            # Hugging Face model
            if sequence_mask is None:
                sequence_mask = torch.ones(sequences.shape[:2], device=sequences.device)

            outputs = self.encoder(
                input_ids=sequences,
                attention_mask=sequence_mask,
            )
            hidden_states = outputs.last_hidden_state
        else:
            # Custom encoder
            hidden_states = self.encoder(sequences)

        # Project to output dimension
        seq_embeddings = self.projection(hidden_states)

        # Add circular positional encoding if applicable
        if is_circular is not None:
            circular_mask = is_circular.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            circular_pe = self.circular_pe(seq_embeddings.shape[1])
            seq_embeddings = seq_embeddings + circular_mask * circular_pe

        # Pool to get sequence-level representation
        if self.use_pooling == "mean":
            if sequence_mask is not None:
                mask = sequence_mask.unsqueeze(-1).float()
                pooled = (seq_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1)
            else:
                pooled = seq_embeddings.mean(dim=1)
        elif self.use_pooling == "cls":
            pooled = seq_embeddings[:, 0]
        elif self.use_pooling == "attention":
            pooled = self.attention_pool(seq_embeddings, sequence_mask)
        else:
            pooled = seq_embeddings.mean(dim=1)

        return seq_embeddings, pooled


class DNACNNEncoder(nn.Module):
    """
    Custom CNN encoder for DNA sequences.

    Used when pre-trained models are not available or for fast inference.
    """

    def __init__(
        self,
        vocab_size: int = 5,  # A, C, G, T, N
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 6,
        kernel_sizes: Tuple[int, ...] = (7, 7, 5, 5, 3, 3),
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        layers = []
        in_channels = embedding_dim
        for i, kernel_size in enumerate(kernel_sizes):
            out_channels = hidden_dim if i > 0 else hidden_dim // 2
            layers.append(nn.Conv1d(
                in_channels, out_channels, kernel_size,
                padding=kernel_size // 2
            ))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.GELU())
            if i % 2 == 1:
                layers.append(nn.MaxPool1d(2))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        Encode sequences with CNN.

        Args:
            sequences: Token IDs [batch, seq_len]

        Returns:
            Sequence embeddings [batch, seq_len//4, output_dim]
        """
        # Embed tokens
        x = self.embedding(sequences)  # [B, L, D]

        # Transpose for CNN
        x = x.transpose(1, 2)  # [B, D, L]

        # Apply CNN
        x = self.cnn(x)  # [B, H, L']

        # Transpose back
        x = x.transpose(1, 2)  # [B, L', H]

        # Project to output
        x = self.output_proj(x)

        return x


class CircularPositionalEncoding(nn.Module):
    """
    Circular-aware positional encoding.

    Standard positional encodings assume linear sequences. This variant
    adds circular topology information for ecDNA candidate regions.
    """

    def __init__(self, d_model: int, max_len: int = 6000):
        super().__init__()

        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

        # Circular encoding (wraps around)
        circular_pe = torch.zeros(max_len, d_model)
        for i in range(max_len):
            # Use periodic function that wraps
            angle = 2 * math.pi * i / max_len
            for j in range(d_model // 2):
                freq = j + 1
                circular_pe[i, 2*j] = math.sin(freq * angle)
                circular_pe[i, 2*j + 1] = math.cos(freq * angle)

        self.register_buffer('circular_pe', circular_pe)

        # Learned mixing
        self.mix = nn.Parameter(torch.tensor(0.5))

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Get positional encoding.

        Args:
            seq_len: Sequence length

        Returns:
            Positional encoding [seq_len, d_model]
        """
        linear = self.pe[:seq_len]
        circular = self.circular_pe[:seq_len]

        # Mix linear and circular encodings
        mix = torch.sigmoid(self.mix)
        return mix * linear + (1 - mix) * circular


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence with attention.

        Args:
            x: Sequence [batch, seq_len, hidden]
            mask: Attention mask [batch, seq_len]

        Returns:
            Pooled representation [batch, hidden]
        """
        # Compute attention weights
        attn_scores = self.attention(x).squeeze(-1)  # [B, L]

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask.bool(), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L]

        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)

        return pooled
