import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from MOSEI.utils import cont_NTXentLoss
import os
import sys

path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(path)
sys.path.append(os.path.dirname(path))
import MOSEI.config as cfg
from Text_encoder import TextEncoder
from Vision_encoder import VisionEncoder
from Audio_encoder import AudioEncoder
from trans.transformer import TransformerEncoder
from classifier import BaseClassifier


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class common_feature_extractor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(common_feature_extractor, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class private_feature_extractor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(private_feature_extractor, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=3, drop_out=0.1, config=cfg):
        super(FeatureProjector, self).__init__()
        self.device = config.DEVICE

        self.feed_foward_size = int(output_dim / 2)
        self.project_size = output_dim - self.feed_foward_size
        self.proj1 = nn.Linear(input_dim, self.feed_foward_size, bias=True)

        self.proj2 = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.proj2.append(nn.Linear(input_dim, self.project_size, bias=False))
            else:
                self.proj2.append(nn.Linear(self.project_size, self.project_size, bias=False))
            self.proj2.append(nn.GELU())

        self.layernorm_ff = nn.LayerNorm(self.feed_foward_size)
        self.layernorm = nn.LayerNorm(self.project_size)
        self.MLP = nn.Sequential(*self.proj2)
        self.drop = nn.Dropout(p=drop_out)

    def forward(self, batch):
        # input: list of data samples with different seq length
        dropped = self.drop(batch)
        ff = self.proj1(dropped)
        x = self.MLP(dropped)
        x = torch.cat([self.layernorm(x), self.layernorm_ff(ff)], dim=-1)
        # return x.transpose(0, 1)  # return shape: [seq,batch,fea]
        return x


class TVA_fusion(nn.Module):
    def __init__(self, config=cfg):
        super(TVA_fusion, self).__init__()
        self.config = config
        self.text_dropout = config.MOSEI.downStream.text_drop_out

        encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim
        audio_text_nhead = config.MOSEI.downStream.audio_text_nhead
        audio_text_tf_num_layers = config.MOSEI.downStream.audio_text_tf_num_layers

        # 特征分解
        uni_fea_dim = int(encoder_fea_dim / 2)
        self.T_simi_proj = common_feature_extractor(encoder_fea_dim, uni_fea_dim)
        self.V_simi_proj = common_feature_extractor(encoder_fea_dim, uni_fea_dim)
        self.A_simi_proj = common_feature_extractor(encoder_fea_dim, uni_fea_dim)

        self.T_dissimi_proj = private_feature_extractor(encoder_fea_dim, uni_fea_dim)
        self.V_dissimi_proj = private_feature_extractor(encoder_fea_dim, uni_fea_dim)
        self.A_dissimi_proj = private_feature_extractor(encoder_fea_dim, uni_fea_dim)

        self.audio_fea_dim = config.MOSEI.downStream.audio_fea_dim
        self.a_len = config.MOSEI.downStream.audio_seq_len

        vision_text_nhead = config.MOSEI.downStream.vision_text_nhead
        vision_text_tf_num_layers = config.MOSEI.downStream.vision_text_tf_num_layers

        attn_dropout = config.MOSEI.downStream.drop_out
        attn_mask = config.MOSEI.downStream.attn_mask

        audio_fea_dim = config.MOSEI.downStream.audio_fea_dim
        vision_fea_dim = config.MOSEI.downStream.vision_fea_dim
        text_fea_dim = config.MOSEI.downStream.text_fea_dim

        self.vlen, self.alen = config.MOSEI.downStream.vlen, config.MOSEI.downStream.alen
        #
        self.prompta_m = nn.Parameter(torch.rand(self.alen, encoder_fea_dim))
        self.promptv_m = nn.Parameter(torch.rand(self.vlen, encoder_fea_dim))

        self.text_encoder = TextEncoder(with_projector=False, config=config)
        self.proj_t = nn.Linear(text_fea_dim, encoder_fea_dim)

        self.proj_v = nn.Linear(vision_fea_dim, encoder_fea_dim)

        self.vision_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=vision_text_nhead, layers=vision_text_tf_num_layers,
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        )  # Q:text, KV:vision

        self.proj_a = nn.Linear(audio_fea_dim, encoder_fea_dim)
        self.audio_with_text = TransformerEncoder(
            embed_dim=encoder_fea_dim, num_heads=audio_text_nhead, layers=audio_text_tf_num_layers,
            attn_dropout=attn_dropout, relu_dropout=attn_dropout, res_dropout=attn_dropout, embed_dropout=attn_dropout,
            attn_mask=attn_mask
        )  # Q:text, KV:audio

        self.vision_encoder = VisionEncoder(config=config)
        self.audio_encoder = AudioEncoder(config=config)

        self.p2a = nn.Linear(384, 768)


        self.TVA_decoder = BaseClassifier(
            input_size=encoder_fea_dim * 3,
            hidden_size=[encoder_fea_dim, encoder_fea_dim // 2, encoder_fea_dim // 8],
            output_size=1
        )
        # 试着改一下层数1 2 4 8
        self.mono_decoder = BaseClassifier(input_size=uni_fea_dim,
                                           hidden_size=[encoder_fea_dim // 4, encoder_fea_dim // 8],
                                           output_size=1)
        self.heat = config.MOSEI.downStream.const_heat
        self.ntxent_loss = cont_NTXentLoss(temperature=self.heat)

        self.device = config.DEVICE
        self.MSEcriterion = torch.nn.MSELoss()
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.model_path = config.MOSEI.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)

    def load_froze(self):
        model_path = self.config.MOSEI.path.encoder_path + str(self.config.seed) + '/'
        self.audio_encoder.load_state_dict(
            torch.load(model_path + 'best_loss_audio_encoder.pt', map_location=self.device))
        self.vision_encoder.load_state_dict(
            torch.load(model_path + 'best_loss_vision_encoder.pt', map_location=self.device))
        self.audio_encoder.set_froze()
        self.vision_encoder.set_froze()

    def forward(self, sample1, sample2, mode='train', device=cfg.DEVICE):
        text = sample1['raw_text']
        vision = sample1['vision'].clone().detach().to(device).float()
        audio = sample1['audio'].clone().detach().to(device).float()
        regression_labels = sample1['labels']['M'].clone().detach().to(device).float()
        vision_padding_mask = sample1['vision_padding_mask'].clone().detach().to(device)
        audio_padding_mask = sample1['audio_padding_mask'].clone().detach().to(device)
        if device is None:
            device = self.device
        last_hidden_text, pooler_output = self.text_encoder(text)  # [bs, seq, h] [bs, h]
        pooler_output = pooler_output.squeeze()
        last_hidden_text = F.dropout(self.proj_t(last_hidden_text.permute(1, 0, 2)),
                                     p=self.text_dropout, training=self.training
                                     )
        x_t_embed = last_hidden_text[0]

        proj_vision = self.proj_v(vision).permute(1, 0, 2) + self.promptv_m.unsqueeze(1)
        # proj_vision = self.proj_v(vision).permute(1, 0, 2)
        h_tv = self.vision_with_text(last_hidden_text, proj_vision,
                                     proj_vision)  # [seq-v, bs, 768] [seq-t, bs, 768]--> [seq, bs,h]

        proj_audio = self.proj_a(audio).permute(1, 0, 2) + self.prompta_m.unsqueeze(1)
        # proj_audio = self.proj_a(audio).permute(1, 0, 2)
        h_ta = self.audio_with_text(last_hidden_text, proj_audio, proj_audio)

        vision1 = vision.clone().detach().to(device).float()
        audio1 = audio.clone().detach().to(device).float()
        label1 = regression_labels.clone().detach().to(device).float()
        label_T1 = regression_labels.clone().detach().to(device).float()
        label_A1 = regression_labels.clone().detach().to(device).float()
        label_V1 = regression_labels.clone().detach().to(device).float()
        key_padding_mask_V1, key_padding_mask_A1 = (vision_padding_mask.clone().detach().to(device),
                                                    audio_padding_mask.clone().detach().to(device))
        v_embed, xv = self.vision_encoder(vision1, key_padding_mask=key_padding_mask_V1)
        a_embed, xa = self.audio_encoder(audio1, key_padding_mask=key_padding_mask_A1)
        xv = xv.squeeze()
        xa = xa.squeeze()
        # 相似不相似特征分解
        x_t_simi1 = self.T_simi_proj(pooler_output)  # [bs,encoder_f_d]
        x_v_simi1 = self.V_simi_proj(xv)  # [[batch_size, fea_dim/2]]
        x_v_simi1_p = x_v_simi1
        x_a_simi1 = self.A_simi_proj(xa)  # [[batch_size, fea_dim/2]]
        x_a_simi1_p = x_a_simi1
        x_t_dissimi1 = self.T_dissimi_proj(pooler_output)
        x_v_dissimi1 = self.V_dissimi_proj(xv)
        x_a_dissimi1 = self.A_dissimi_proj(xa)

        x1_s = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1), dim=-1)
        x1_ds = torch.cat((x_t_dissimi1, x_v_dissimi1, x_a_dissimi1), dim=-1)
        x1_all = torch.cat((x1_s, x1_ds), dim=-1)
        x1_sds = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1,), dim=0)
        x_sds = x1_sds
        # 将公共特征连接
        common_feature1 = x_t_simi1 + x_v_simi1 + x_a_simi1
        # <========= common and private feature extractor
        # 得到common_mask
        mask1 = torch.ones(x_t_simi1.unsqueeze(dim=1).shape[:2]).clone().detach().to(device)
        common_mask1 = torch.ones(common_feature1.unsqueeze(dim=1).shape[:2]).clone().detach().to(device)
        label1_sds = torch.cat((label1, label1, label1, label_T1, label_V1, label_A1,), dim=0)
        label_all = label1.squeeze()
        label_sds = label1_sds

        if sample2 is not None:
            text2 = sample2['raw_text']
            vision2 = sample2['vision'].clone().detach().to(device).float()
            audio2 = sample2['audio'].clone().detach().to(device).float()
            label2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_T2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_V2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_A2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            key_padding_mask_V2, key_padding_mask_A2 = (sample2['vision_padding_mask'].clone().detach().to(device),
                                                        sample2['audio_padding_mask'].clone().detach().to(device))

            xt2, x_t_embed2 = self.text_encoder(text2, device=device)
            xv2, x_v_embed2 = self.vision_encoder(vision2, key_padding_mask=key_padding_mask_V2)
            xa2, x_a_embed2 = self.audio_encoder(audio2, key_padding_mask=key_padding_mask_A2)
            x_t_embed2 = x_t_embed2.squeeze()
            x_v_embed2 = x_v_embed2.squeeze()
            x_a_embed2 = x_a_embed2.squeeze()

            x_t_simi2 = self.T_simi_proj(x_t_embed2)
            x_v_simi2 = self.V_simi_proj(x_v_embed2)
            x_a_simi2 = self.A_simi_proj(x_a_embed2)
            x_t_dissimi2 = self.T_dissimi_proj(x_t_embed2)
            x_v_dissimi2 = self.V_dissimi_proj(x_v_embed2)
            x_a_dissimi2 = self.A_dissimi_proj(x_a_embed2)

            x2_s = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2), dim=-1)
            x2_ds = torch.cat((x_t_dissimi2, x_v_dissimi2, x_a_dissimi2), dim=-1)
            x2_all = torch.cat((x2_s, x2_ds), dim=-1)
            x2_sds = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2, x_a_dissimi2,
                                ), dim=0)
            label2_sds = torch.cat((label2, label2, label2, label_T2, label_V2, label_A2,), dim=0)
            # x = torch.cat((x1_all, x2_all), dim=0)
            # x_sds = torch.cat((x1_sds, x2_sds), dim=0)
            # label_sds = torch.cat((label1_sds, label2_sds), dim=0)


        x_v_embed = h_tv[0]  # [batch_size, encoder_fea_dim]
        x_a_embed = h_ta[0]
        x_v_embeds = x_v_embed
        x_a_embeds = x_a_embed


        x_v_simi1 = self.p2a(x_v_simi1)
        x_a_simi1 = self.p2a(x_a_simi1)

        x_v_embed = x_v_embed * x_v_simi1#*
        x_a_embed = x_a_embed * x_a_simi1#*

        x = torch.cat([x_t_embed,x_a_embed,x_v_embed], dim=-1)  # [bs, 3h]
        pred = self.TVA_decoder(x).view(-1)  # [bs]

        if mode == 'train':
            # loss_v = self.get_KL_loss(x_v_embed,x_v_simi1)
            # loss_a = self.get_KL_loss(x_a_embed,x_a_simi1)
            loss_nce = 0
            pred_mono = self.mono_decoder(x_sds)
            sup_const_loss = 0
            # sds_loss = 0
            if sample2 is not None:
                # [Ts,T1s,T2s,T3s,T4s,T5s,T6s,V1s,V2s,V3s,....]
                t1, p, t2, n = torch.tensor([0, 0, 7, 7, 14, 14,
                                             0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                                            device=device), \
                    torch.tensor([1, 2, 8, 9, 15, 16,
                                  7, 14, 8, 15, 9, 16, 10, 17, 11, 18, 12, 19, 13, 20],
                                 device=device), \
                    torch.tensor([0, 0, 0, 0, 7, 7, 7, 7, 14, 14, 14, 14,
                                  0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                                 device=device), \
                    torch.tensor([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20,
                                  21, 28, 35, 22, 29, 36, 23, 30, 37, 24, 31, 38, 25, 32, 39, 26, 33, 40, 27,
                                  34, 41], device=device)

                indices_tuple = (t1, p, t2, n)
                pre_sample_label = torch.tensor([0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4,
                                                 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, ])
                for i in range(len(x1_all)):
                    pre_sample_x = []
                    # print("debug:")
                    # print("shape:"f"{x_t_simi1.shape}"","f"{x_v_simi1.shape}"","f"{x_a_simi1.shape}"f"{x_t_simi2.shape}"","f"{x_v_simi2.shape}"","f"{x_a_simi2.shape}")
                    for fea1, fea2 in zip([x_t_simi1, x_v_simi1_p, x_a_simi1_p, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1, ],
                                          [x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2,
                                           x_a_dissimi2, ]):

                        pre_sample_x.append(torch.cat((fea1[i].unsqueeze(0), fea2[6 * i:6 * (i + 1)]), dim=0))
                        # print("debuf:")
                        # print(pre_sample_x)
                    sup_const_loss += self.ntxent_loss(torch.cat(pre_sample_x, dim=0), pre_sample_label,indices_tuple=indices_tuple)

                sup_const_loss /= len(x1_all)

            pred_loss = self.MSEcriterion(pred.squeeze(), label_all)
            mono_task_loss = self.MSEcriterion(pred_mono.squeeze(), label_sds)

            loss = pred_loss + 0.02 * sup_const_loss + 0.03 * mono_task_loss




            # if return_emb:
            #     return pred, x1_all, loss, pred_loss, sup_const_loss
            # else:
            #     return pred, (pooler_output, v_embed, a_embed), loss, pred_loss, sup_const_loss
            return pred,loss, loss_nce, pred_loss, sup_const_loss

        else:
            # if return_emb:
            #     return x1_all
            # else:
            #     return (pooler_output, v_embed, a_embed)
            return pred, None
        # return pred,loss_nce

    def save_model(self, name=None):
        # save all modules
        if name == None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = self.model_path + str(name) + 'TVA_fusion' + '_model.pt'
        print('model saved at:\n', mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name=None):
        if name == None:
            mode_path = self.model_path + 'TVA_fusion' + '_model.pt'
        else:
            mode_path = name
        print('model loaded from:\n', mode_path)
        # self.load_state_dict(torch.load(mode_path, map_location=self.device))
        checkpoint = torch.load(mode_path, map_location=self.device)
        model_state_dict = self.state_dict()
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_state_dict}
        self.load_state_dict(filtered_checkpoint, strict=False)

    def get_distill_loss(self, input1, input2):
        diff_loss = torch.mean((input1 - input2) * (input1 - input2))
        return diff_loss

    def get_KL_loss(self, x_embed, x_embed_target):
        x_embed1 = F.log_softmax(x_embed, dim=1)
        x_embed_target1 = F.softmax(x_embed_target, dim=1)
        loss = self.criterion(x_embed1, x_embed_target1)
        return loss

    def get_InfoNCE_loss(self, input1, input2):

        x1 = input1 / input1.norm(dim=1, keepdim=True)
        x2 = input2 / input2.norm(dim=1, keepdim=True)

        pos = torch.sum(x1 * x2, dim=-1)  # bs
        neg = torch.logsumexp(torch.matmul(x1, x2.t()), dim=-1)  # bs
        nce_loss = -(pos - neg).mean()

        return nce_loss


