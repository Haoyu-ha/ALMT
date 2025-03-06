import torch
from torch import nn
from .almt_layer import Transformer, CrossTransformer, HhyperLearningEncoder
from .bert import BertTextEncoder
from einops import repeat


class ALMT(nn.Module):
    def __init__(self, args):
        super(ALMT, self).__init__()

        args = args.model

        self.h_hyper = nn.Parameter(torch.ones(1, args.token_len, args.token_dim))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=args.bert_pretrained)

        self.proj_l = nn.Sequential(
            nn.Linear(args.l_input_dim, args.l_proj_dst_dim),
            Transformer(num_frames=args.l_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_a = nn.Sequential(
            nn.Linear(args.a_input_dim, args.a_proj_dst_dim),
            Transformer(num_frames=args.a_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )
        self.proj_v = nn.Sequential(
            nn.Linear(args.v_input_dim, args.v_proj_dst_dim),
            Transformer(num_frames=args.v_input_length, save_hidden=False, token_len=args.token_length, dim=args.proj_input_dim, depth=args.proj_depth, heads=args.proj_heads, mlp_dim=args.proj_mlp_dim)
        )

        self.l_encoder = Transformer(num_frames=args.token_length, save_hidden=True, token_len=None, dim=args.proj_input_dim, depth=args.AHL_depth-1, heads=args.l_enc_heads, mlp_dim=args.l_enc_mlp_dim)
        self.h_hyper_layer = HhyperLearningEncoder(dim=args.token_dim, depth=args.AHL_depth, heads=args.ahl_heads, dim_head=args.ahl_dim_head, dropout=args.ahl_droup)
        self.fusion_layer = CrossTransformer(source_num_frames=args.token_len, tgt_num_frames=args.token_len, dim=args.proj_input_dim, depth=args.fusion_layer_depth, heads=args.fusion_heads, mlp_dim=args.fusion_mlp_dim)

        self.regression_layer = nn.Sequential(
            nn.Linear(args.token_dim, 1)
        )

    def forward(self, x_visual, x_audio, x_text):
        b = x_visual.size(0)

        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b=b)

        x_text = self.bertmodel(x_text)

        h_v = self.proj_v(x_visual)[:, :self.h_hyper.shape[1]]
        h_a = self.proj_a(x_audio)[:, :self.h_hyper.shape[1]]
        h_l = self.proj_l(x_text)[:, :self.h_hyper.shape[1]]

        h_t_list = self.l_encoder(h_l)
        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]

        output = self.regression_layer(feat)

        return output


def build_model(args):
    model = ALMT(args)

    return model



