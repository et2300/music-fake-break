"""
Music Generation Script v2.0
功能：基于Transformer的音频生成工具(支持WAV)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from typing import List

# ----------------------
# 1. 配置参数 (按需修改)
# ----------------------
class Config:
    # 数据参数
    audio_dir = "./wav_files"       # WAV文件目录
    seq_length = 100                # 输入序列长度
    batch_size = 32
    sample_rate = 22050             # 音频采样率
    n_mfcc = 20                     # MFCC特征数量
    
    # 模型参数
    d_model = 512                   # 特征维度
    nhead = 8                       # 注意力头数
    num_layers = 6                  # Transformer层数
    dropout = 0.1
    
    # 训练参数
    epochs = 100
    lr = 0.0001
    save_path = "./audio_transformer.pth"

# ----------------------
# 2. 数据预处理模块
# ----------------------
class AudioDataset(Dataset):
    def __init__(self, config):
        self.seq_length = config.seq_length
        self.sample_rate = config.sample_rate
        self.n_mfcc = config.n_mfcc
        self.data = self._load_audio(config.audio_dir)
    
    def _load_audio(self, path):
        """加载音频文件并提取MFCC特征(支持WAV和视频)"""
        features = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if file.endswith('.wav'):
                audio, _ = librosa.load(file_path, sr=self.sample_rate)
            elif file.endswith(('.mp4', '.avi', '.mov')):
                # 从视频提取音频
                import ffmpeg
                try:
                    audio, _ = (
                        ffmpeg.input(file_path)
                        .output('pipe:', format='wav', ac=1, ar=self.sample_rate)
                        .run(capture_stdout=True, capture_stderr=True)
                    )
                    audio = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0
                except ffmpeg.Error as e:
                    print(f"Error processing {file}: {e.stderr.decode()}")
                    continue
            
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            features.append(mfcc.T)  # 转置使时间步在第一个维度
        return self._normalize(np.concatenate(features))
    
    def _normalize(self, data):
        """归一化处理"""
        return (data - np.mean(data)) / np.std(data)
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_length]
        return seq[:-1], seq[-1]  # 输入序列, 目标特征

# ----------------------
# 3. 模型架构模块
# ----------------------
class AudioTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 特征嵌入
        self.feature_embed = nn.Linear(config.n_mfcc, config.d_model)
        
        # Transformer核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # 输出头
        self.output_head = nn.Linear(config.d_model, config.n_mfcc)

    def forward(self, x):
        # 输入x: [batch, seq_len, n_mfcc]
        embed = self.feature_embed(x)
        out = self.transformer(embed)
        return self.output_head(out[:, -1])  # 预测下一个时间步的特征

# ----------------------
# 4. 训练流程模块
# ----------------------
def train(config):
    # 初始化
    dataset = AudioDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = AudioTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(config.epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), config.save_path)

def main():
    config = Config()
    train(config)

if __name__ == "__main__":
    main()
