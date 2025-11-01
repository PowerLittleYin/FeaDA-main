
import SIMS.config as cfg
import torch
from torch import nn
import torch.nn.functional as F
import os

from trans.transformer import TransformerEncoder
from classifier import BaseClassifier

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class PositionEncodingTraining(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.

    """

    def __init__(self, fea_size=None, tf_hidden_dim=None, drop_out=None, config=cfg):
        super().__init__()
        if fea_size is None:
            fea_size = config.SIMS.downStream.audio_fea_dim
        if tf_hidden_dim is None:
            tf_hidden_dim = config.SIMS.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.SIMS.downStream.audio_drop_out

        self.cls_token = nn.Parameter(torch.ones(1, 1, tf_hidden_dim))
        num_patches = 400
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings




class TfEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward, num_layers, dropout=0.2, activation='gelu',
                 config=cfg):
        super(TfEncoder, self).__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except:
            raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
        # d_model = int(d_model / 2)
        # dim_feedforward = int(dim_feedforward / 2)
        self.device = config.DEVICE
        self.model_type = 'audio_encoder'
        self.src_mask = None
        self.pos_encoder = PositionEncodingTraining()

        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=activation)

        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, has_mask=True, src_key_padding_mask=None):
        src = self.pos_encoder(src)

        src = src.transpose(0, 1)
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        output = self.transformer_encoder(src, mask=self.src_mask, src_key_padding_mask=src_key_padding_mask)

        return output.transpose(0, 1)


class AudioEncoder(nn.Module):
    def __init__(self, config=cfg):
        super(AudioEncoder, self).__init__()

        self.encoder_fea_dim = config.SIMS.downStream.encoder_fea_dim
        self.audio_fea_dim = config.SIMS.downStream.audio_fea_dim
        self.audio_seq_len = config.SIMS.downStream.audio_seq_len
        self.audio_nhead = config.SIMS.downStream.audio_nhead
        self.dim_feedforward = config.SIMS.downStream.encoder_fea_dim
        self.audio_tf_num_layers = config.SIMS.downStream.audio_tf_num_layers
        self.attn_dropout = config.SIMS.downStream.audio_drop_out
        self.attn_mask = config.SIMS.downStream.audio_attn_mask
        self.layernorm = nn.LayerNorm(self.encoder_fea_dim)
        self.proj_a = nn.Linear(self.audio_fea_dim, self.encoder_fea_dim)

        self.trans_encoder_a = TransformerEncoder(embed_dim=self.encoder_fea_dim,
                                num_heads=self.audio_nhead, # 8
                                layers=self.audio_tf_num_layers, # 2
                                attn_dropout=self.attn_dropout,
                                relu_dropout=self.attn_dropout,
                                res_dropout=self.attn_dropout,
                                embed_dropout=self.attn_dropout,
                                attn_mask=self.attn_mask) # True
        self.encoder = TfEncoder(d_model=self.encoder_fea_dim, nhead=self.audio_nhead, dim_feedforward=self.dim_feedforward,
                                 num_layers=self.audio_tf_num_layers,
                                 dropout=self.attn_dropout, activation='gelu',
                                 config=config)

    def forward(self,audio, key_padding_mask):

        proj_x_a = self.proj_a(audio) # [bs, seq, h]
        proj_x_a = proj_x_a.permute(1, 0, 2)


        h_as = self.trans_encoder_a(proj_x_a)
        if type(h_as) == tuple:
            h_as = h_as[0]# [seq, bs,h]
        last_h_a = h_as[0]   # Take the last output for prediction # [bs, h]


        x = self.encoder(audio, has_mask=False, src_key_padding_mask=key_padding_mask)
        x = self.layernorm(x)
        x = torch.mean(x, dim=-2, keepdim=True)
        return last_h_a,x

    def set_froze(self):
        for param in self.parameters():
            param.requires_grad = False


class AudioPretrain(nn.Module):
    def __init__(self, config=cfg, encoder_fea_dim=None):
        super(AudioPretrain, self).__init__()
        if encoder_fea_dim is None:
            encoder_fea_dim = config.SIMS.downStream.encoder_fea_dim
        self.encoder = AudioEncoder(config)
        self.classifier = BaseClassifier(
            input_size=encoder_fea_dim,
            hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 8)],
            output_size=1, name='AudioRegClassifier',
        )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config = config
        self.model_path = config.SIMS.path.encoder_path + str(config.seed) + '/'
        check_dir(self.model_path)

    def forward(self, audio, label, key_padding_mask,return_loss=True, device=None):
        if device is None:
            device = self.device
        x1,x2 = self.encoder(audio, key_padding_mask)
        pred = self.classifier(x1).squeeze()

        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, x2,loss
        else:
            return pred,x2

    def save_model(self, name='best_loss'):
        # save all modules
        encoder_path = self.model_path + name + '_audio_encoder.pt'
        decoder_path = self.model_path + name + '_audio_decoder.pt'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)

    def load_model(self, name='best_loss', module=None):
        encoder_path = self.model_path + name + '_audio_encoder.pt'
        decoder_path = self.model_path + name + '_audio_decoder.pt'
        print('model loaded from:')

        if module == 'encoder':
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(encoder_path)
        if module == 'decoder':
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(decoder_path)
        if module == 'all' or module is None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(encoder_path)
            print(decoder_path)

