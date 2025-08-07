import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import shap
from collections import OrderedDict

# omics data
Gene_expression_file = './data_process/data/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'
Methylation_file = './data_process/data/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'
Genomic_mutation_file = './data_process/data/CCLE/genomic_mutation_34673_demap_features.csv'

# ----------------------------
# 模型定义
# ----------------------------
class OmicsFeatureEncoder(nn.Module):
    def __init__(self, gexpr_dim=697, methy_dim=808):
        super().__init__()
        self.mulfc_gexpr1 = nn.Linear(gexpr_dim, 256)
        self.fc_gexpr2 = nn.Linear(256, 128)
        self.mulfc_methy1 = nn.Linear(methy_dim, 256)
        self.fc_methy2 = nn.Linear(256, 128)
        self.fc = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, gexpr, methy):
        gexpr = self.dropout(self.relu(self.mulfc_gexpr1(gexpr)))
        gexpr = self.relu(self.fc_gexpr2(gexpr))

        methy = self.dropout(self.relu(self.mulfc_methy1(methy)))
        methy = self.relu(self.fc_methy2(methy))

        return self.relu(self.fc(torch.cat([gexpr, methy], dim=1))), gexpr, methy


# ----------------------------
# 数据加载函数
# ----------------------------
def load_omics_data(Gene_expression_file, Methylation_file):
    # 读取CSV文件
    gexpr_feature = pd.read_csv(Gene_expression_file, sep=',', header=0, index_col=[0])
    methylation_feature = pd.read_csv(Methylation_file, sep=',', header=0, index_col=[0])

    # 归一化处理
    scaler = StandardScaler()
    gexpr_feature_std = scaler.fit_transform(gexpr_feature)
    methylation_feature_std = scaler.fit_transform(methylation_feature)

    # 转换为PyTorch张量
    gexpr_tensor = torch.FloatTensor(gexpr_feature_std)
    methy_tensor = torch.FloatTensor(methylation_feature_std)

    return gexpr_tensor, methy_tensor


# ----------------------------
# 对比损失（InfoNCE Loss）
# ----------------------------
def contrastive_loss(gexpr_embed, methy_embed, temperature=0.1):
    sim_matrix = torch.matmul(gexpr_embed, methy_embed.T) / temperature
    labels = torch.arange(gexpr_embed.size(0)).to(gexpr_embed.device)
    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    return loss

def augment(gexpr, methy):
    # 高斯噪声 + 随机mask
    gexpr = gexpr * torch.bernoulli(torch.ones_like(gexpr)*0.9) + torch.randn_like(gexpr)*0.1
    methy = methy * torch.bernoulli(torch.ones_like(methy)*0.9) + torch.randn_like(methy)*0.1
    return gexpr, methy


def train_contrastive(model, gexpr_data, methy_data, epochs, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.005)

    num_samples = gexpr_data.size(0)
    indices = np.arange(num_samples)

    for epoch in range(epochs):
        # 手动打乱数据
        np.random.shuffle(indices)
        total_loss = 0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]

            gexpr_batch = gexpr_data[batch_idx].to(device)
            methy_batch = methy_data[batch_idx].to(device)
            gexpr_batch,methy_batch = augment(gexpr_batch,methy_batch)

            optimizer.zero_grad()
            _, gexpr_emb, methy_emb = model(gexpr_batch, methy_batch)

            loss = contrastive_loss(gexpr_emb, methy_emb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model

if __name__ == "__main__":
    # 1. 加载数据
    gexpr_data, methy_data = load_omics_data(Gene_expression_file, Methylation_file)

    # 2. 对比学习预训练
    print("=== 开始对比学习预训练 ===")
    encoder = OmicsFeatureEncoder()
    trained_encoder = train_contrastive(encoder, gexpr_data, methy_data, epochs=150)
    torch.save(trained_encoder.state_dict(), "pretrained_omics_encoder.pth")



