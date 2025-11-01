import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from MOSI.data_loader import MOSIDataloader
from models.model import TVA_fusion
import MOSI.config as cfg

# 初始化模型和数据加载器
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = TVA_fusion(config=cfg).to(device)
model.eval()

model.load_model()

# 加载数据集
test_loader = MOSIDataloader('test', batch_size=32, config=cfg)

# 提取视觉特征
all_visual_features = []
all_labels = []
with torch.no_grad():
    for batch in test_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        outputs = model(inputs, None, mode='test', return_all_fea=True)
        if isinstance(outputs, tuple) and len(outputs) > 1:
            visual_features = outputs[1][0].cpu().numpy()
            all_visual_features.append(visual_features)
            all_labels.extend(batch['labels']['M'].cpu().numpy())

# 将特征列表转换为2D numpy数组
all_visual_features = np.vstack(all_visual_features)

# 动态设置perplexity参数
n_samples = all_visual_features.shape[0]
perplexity = min(30, n_samples - 1)  # 确保perplexity小于样本数量

# 使用t-SNE降维
tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42)
all_visual_features_3d = tsne.fit_transform(all_visual_features)

# 绘制散点图
unique_labels = np.unique(all_labels)
colors = plt.cm.get_cmap('tab10', len(unique_labels))
for i, label in enumerate(unique_labels):
    idx = (all_labels == label)
    plt.scatter(all_visual_features_3d[idx, 0], all_visual_features_3d[idx, 1], all_visual_features_3d[idx, 2],
                c=[colors(i)], label=label)

plt.legend()
plt.show()