'''
* @name: ALMT
* @description: Implementation of ALMT
'''

import torch
from torch import nn
from .almt_layer import Transformer, CrossTransformer, HhyperLearningEncoder
from .bert import BertTextEncoder
from einops import repeat


class ALMT(nn.Module):
    def __init__(self, dataset, AHL_depth=3, fusion_layer_depth=2, bert_pretrained='bert-base-uncased'):
        super(ALMT, self).__init__()

        self.h_hyper = nn.Parameter(torch.ones(1, 8, 128))

        self.bertmodel = BertTextEncoder(use_finetune=True, transformers='bert', pretrained=bert_pretrained)

        # mosi
        if dataset == 'mosi':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(5, 128)
            self.proj_v0 = nn.Linear(20, 128)
        elif dataset == 'mosei':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(74, 128)
            self.proj_v0 = nn.Linear(35, 128)
        elif dataset == 'sims':
            self.proj_l0 = nn.Linear(768, 128)
            self.proj_a0 = nn.Linear(33, 128)
            self.proj_v0 = nn.Linear(709, 128)
        else:
            assert False, "datasetName error"


        self.proj_l = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_a = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)
        self.proj_v = Transformer(num_frames=50, save_hidden=False, token_len=8, dim=128, depth=1, heads=8, mlp_dim=128)

        self.text_encoder = Transformer(num_frames=8, save_hidden=True, token_len=None, dim=128, depth=AHL_depth-1, heads=8, mlp_dim=128)
        self.h_hyper_layer = HhyperLearningEncoder(dim=128, depth=AHL_depth, heads=8, dim_head=16, dropout = 0.)
        self.fusion_layer = CrossTransformer(source_num_frames=8, tgt_num_frames=8, dim=128, depth=fusion_layer_depth, heads=8, mlp_dim=128)

        self.cls_head = nn.Sequential(
            nn.Linear(128, 1)
        )


    def forward(self, x_visual, x_audio, x_text):
        b = x_visual.size(0)

        h_hyper = repeat(self.h_hyper, '1 n d -> b n d', b = b)

        x_text = self.bertmodel(x_text)

        x_visual = self.proj_v0(x_visual)
        x_audio = self.proj_a0(x_audio)
        x_text = self.proj_l0(x_text)

        h_v = self.proj_v(x_visual)[:, :8]
        h_a = self.proj_a(x_audio)[:, :8]
        h_t = self.proj_l(x_text)[:, :8]

        h_t_list = self.text_encoder(h_t)

        h_hyper = self.h_hyper_layer(h_t_list, h_a, h_v, h_hyper)
        feat = self.fusion_layer(h_hyper, h_t_list[-1])[:, 0]
        cls_output = self.cls_head(feat)

        return cls_output


def build_model(opt, mode=None):
    if opt.datasetName == 'sims':
        l_pretrained='bert-base-chinese'
    else:
        l_pretrained='bert-base-uncased'
    model = ALMT(dataset = opt.datasetName, fusion_layer_depth=opt.fusion_layer_depth, bert_pretrained = l_pretrained)

    return model
