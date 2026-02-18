import torch
import torch.nn as nn
from gatr.interface import embed_point, embed_scalar, extract_scalar, extract_point
from gatr import GATr, SelfAttentionConfig, MLPConfig
from xformers.ops.fmha import BlockDiagonalMask

class GATrBasicModule(nn.Module):
    """
    GATr wrapper:
      - calorimeter hits → GA points   (x,y,z)
      - tracker hits     → GA vectors (normalized momentum)
      - geometric scalar = det_type_norm (goes to embed_scalar)
      - extra scalars (E, p_mod) passed directly to scalar MLP
    """

    def __init__(
        self,
        hidden_mv_channels=32,
        hidden_s_channels=64,
        num_blocks=2,
        in_s_channels=2,   # E, p_mod
        in_mv_channels=1,
        out_mv_channels=1,
        attention: SelfAttentionConfig = SelfAttentionConfig(),
        mlp: MLPConfig = MLPConfig(),
        dropout=0.1,
        out_s_channels=None
        
        
    ):
        super().__init__()
        # print(in_s_channels)
        if out_s_channels is None:
            out_s_channels = hidden_s_channels
        self.gatr = GATr(
            in_mv_channels=in_mv_channels,
            out_mv_channels=out_mv_channels,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,           # E, p_mod (as real scalars)
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=num_blocks,
            attention=attention,
            mlp=mlp,
        )

        self.dropout = nn.Dropout(dropout)

    # ---------------------------------------------
    # Embedding geométrico (punto o vector)
    # ---------------------------------------------
    def build_geom_embedding(self, mv_v_part, mv_s_part, batch):
        mv_vec = embed_point(mv_v_part)
        mv_scalar = embed_scalar(mv_s_part)
        embedded_mv_vec = mv_vec.unsqueeze(1)   # (N,1,16)
        embedded_mv_scalar = mv_scalar.unsqueeze(1)   # (N,1,16)
        
        embedded_geom = embedded_mv_vec + embedded_mv_scalar

        return embedded_geom

    # ---------------------------------------------
    def build_attention_mask(self, batch):
        return BlockDiagonalMask.from_seqlens(
            torch.bincount(batch.long()).tolist()
        )

    # ---------------------------------------------
    def forward(self, mv_v_part, mv_s_part, scalars, batch, embedded_geom=None):
        if embedded_geom is None:
            embedded_geom = self.build_geom_embedding(mv_v_part, mv_s_part, batch)
            # print(embedded_scalars.shape)
        else:
            assert embedded_geom.shape[1] == 1, "Embedded geom must have shape (N,1,16)"
            assert embedded_geom.shape[2] == 16, "Embedded geom must have shape (N,1,16)"
        mask = self.build_attention_mask(batch)

        mv_out, scalar_out = self.gatr(
            embedded_geom,
            scalars=scalars,
            attention_mask=mask
        )
        # mv_out (B,N,1,1,16)

        # Flatten MV part (take blade-1 representation)
        out = torch.cat([mv_out[:, 0, :], scalar_out], dim=-1)
        dropout_out = self.dropout(out)
        mv_out_final, scalar_out_final = torch.split(dropout_out, [16, dropout_out.shape[1]-16], dim=-1)

        point = extract_point(mv_out_final)          # (N, 3)
        scalar = extract_scalar(mv_out_final.reshape(mv_out_final.shape[0], 1, -1))      # (N, 1, 1)
        scalar = scalar.view(-1, 1) # (N,1)
        # Reshape to (N, 1, 16)
        mv_out_final = mv_out_final.view(mv_out_final.shape[0], 1, -1)
        return mv_out_final, scalar_out_final, point, scalar
