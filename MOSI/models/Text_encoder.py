from transformers import BertModel, BertTokenizer
from torch import nn
from classifier import BaseClassifier
import torch
import os
import MOSI.config as cfg


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


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


class TextEncoder(nn.Module):
    def __init__(self, config=cfg, fea_size=None, with_projector=True, proj_fea_dim=None, name=None):
        super(TextEncoder, self).__init__()
        self.name = name
        language = config.MOSI.downStream.language
        if fea_size is None:
            fea_size = config.MOSI.downStream.text_fea_dim
        if proj_fea_dim is None:
            proj_fea_dim = config.MOSI.downStream.proj_fea_dim
        if language == 'en':
            self.tokenizer = BertTokenizer.from_pretrained('/data/mmsd2/model_state/bert-base-uncased')
            self.extractor = BertModel.from_pretrained('/data/mmsd2/model_state/bert-base-uncased')
        elif language == 'cn':
            self.tokenizer = BertTokenizer.from_pretrained('/data/mmsd2/model_state/bert-base-uncased')
            self.extractor = BertModel.from_pretrained('/data/mmsd2/model_state/bert-base-uncased')
        self.device = config.DEVICE
        self.with_projector = with_projector
        if with_projector:
            self.projector = FeatureProjector(fea_size, proj_fea_dim, config=cfg)

    def forward(self, text, device=None):
        if device is None:
            device = self.device
        x = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        x = self.extractor(**x)
        last_hidden_state = x['last_hidden_state']
        pooler_output = x['pooler_output']
        if self.with_projector:
            pooler_output = self.projector(pooler_output)
        return last_hidden_state, pooler_output
        # [bs, seq, h]


class TextPretrain(nn.Module):
    def __init__(self, config=cfg, encoder_fea_dim=None):
        super(TextPretrain, self).__init__()
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSI.downStream.encoder_fea_dim
        self.encoder = TextEncoder(config)
        self.classifier = BaseClassifier(
            input_size=encoder_fea_dim,
            hidden_size=[int(encoder_fea_dim / 2), int(encoder_fea_dim / 8)],
            output_size=1, name='TextClassifier',
        )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config = config
        self.model_path = config.MOSI.path.encoder_path + str(config.seed) + '/'
        check_dir(self.model_path)

    def forward(self, text, label, return_loss=True, device=None):
        if device is None:
            device = self.device
        x = self.encoder(text)
        pred = self.classifier(x).squeeze()

        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, loss
        else:
            return pred

    def save_model(self, name='best_loss'):
        # save all modules
        encoder_path = self.model_path + name + '_text_encoder.pt'
        decoder_path = self.model_path + name + '_text_decoder.pt'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)

    def load_model(self, name='best_loss', module=None):
        encoder_path = self.model_path + name + '_text_encoder.pt'
        decoder_path = self.model_path + name + '_text_decoder.pt'
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


def main():
    # 配置
    config = cfg
    config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.MOSI.downStream.language = 'en'  # 或者 'cn'
    config.MOSI.downStream.text_fea_dim = 768  # BERT 的输出维度
    config.MOSI.downStream.proj_fea_dim = 256  # 投影后的特征维度

    # 创建 TextEncoder 实例
    text_encoder = TextEncoder(config=config).to(config.DEVICE)

    # 示例文本数据
    sample_texts = [
        "This is a sample text for testing.",
        "Another example to check the output shapes.",
        "Testing the TextEncoder with multiple sentences."
    ]

    # 测试输出
    with torch.no_grad():
        last_hidden_state, pooler_output = text_encoder(sample_texts)
        print("Last Hidden State Shape:", last_hidden_state.shape)
        print("Pooler Output Shape:", pooler_output.shape)


if __name__ == "__main__":
    main()


