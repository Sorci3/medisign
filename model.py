# Architecture SPOTER
#
# SPOTER (Sign POse-based TransformER) — Bohacek & Hruz, 2022
#
#
# Flux :
#   (B, T, feature_size)
#     -> input_proj   : Linear(feature_size, hidden_dim)
#     -> + pos        : biais positionnel statique appris (1 seul vecteur partagé)
#     -> encoder      : TransformerEncoder (6 couches)
#     -> decoder      : SPOTERDecoder — cross-attention uniquement (pas de self-attn)
#        <- class_query : token appris analogue au [CLS] de BERT
#     -> linear_class : Linear(hidden_dim, n_classes)
#   (B, n_classes)

import copy
import torch
import torch.nn as nn


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class SPOTERDecoderLayer(nn.TransformerDecoderLayer):
    """
    Couche de décodeur sans self-attention.

    Dans SPOTER le décodeur reçoit un seul token (class_query), donc la
    self-attention est inutile. On la supprime du forward mais on garde
    l'attribut self_attn instancié : PyTorch >= 2.x lit self_attn.batch_first
    en interne même si on ne l'appelle pas.
    """
    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                tgt_is_causal=None, memory_is_causal=None):
        # Cross-attention : class_query interroge la séquence encodée
        # tgt/memory : (B, T, hidden_dim) avec batch_first=True
        tgt  = tgt + self.dropout1(tgt)
        tgt  = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt  = tgt + self.dropout2(tgt2)
        tgt  = self.norm2(tgt)
        # Feed-forward
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt  = tgt + self.dropout3(tgt2)
        tgt  = self.norm3(tgt)
        return tgt


class SPOTER(nn.Module):
    """
    SPOTER pour classification de séquences de landmarks.

    Args:
        num_classes    : nombre de classes à prédire
        feature_size   : dimension des features par frame (225 sans visage, 315 avec)
        hidden_dim     : dimension interne du Transformer
        nhead          : nombre de têtes d'attention
        num_encoder_layers / num_decoder_layers : profondeur
        dim_feedforward : dimension du MLP dans chaque couche
        dropout        : taux de dropout
    """
    def __init__(self, num_classes, feature_size=225, hidden_dim=64,
                 nhead=4, num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=256, dropout=0.1):
        super().__init__()

        # Projection d'entrée : réduit feature_size → hidden_dim
        self.input_proj = (nn.Linear(feature_size, hidden_dim)
                           if feature_size != hidden_dim else nn.Identity())

        # Biais positionnel : un seul vecteur appris partagé sur tous les frames.
        # Choix délibéré : pas de PE sinusoïdal, le modèle raisonne sur la forme
        # spatiale des poses, pas sur l'ordre absolu des frames.
        self.pos = nn.Parameter(torch.rand(1, 1, hidden_dim))

        # Token de classe appris
        self.class_query = nn.Parameter(torch.rand(1, 1, hidden_dim))

        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            batch_first=True,
        )
        # Remplace les couches du décodeur par SPOTERDecoderLayer
        custom_dec = SPOTERDecoderLayer(hidden_dim, nhead, dim_feedforward, dropout, "relu",
                                        batch_first=True)
        self.transformer.decoder.layers = _get_clones(custom_dec, num_decoder_layers)

        # Tête de classification
        self.linear_class = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x : (B, T, feature_size)
        h     = self.input_proj(x) + self.pos                  # (B, T, hidden_dim)
        query = self.class_query.expand(x.size(0), 1, -1)      # (B, 1, hidden_dim)
        out   = self.transformer(h, query)                      # (B, 1, hidden_dim)
        return self.linear_class(out.squeeze(1))                # (B, num_classes)
