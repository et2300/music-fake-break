"""
Music Generation Script v1.2
功能：基于Transformer的定制化音乐生成工具
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pretty_midi import PrettyMIDI

# ----------------------
# 1. 配置参数 (按需修改)
# ----------------------
class Config:
    # 数据参数
    midi_dir = "./midi_files"         # MIDI文件目录
    seq_length = 100                  # 输入序列长度
    batch_size = 32
    
    # 模型参数
    d_model = 512                     # 特征维度
    nhead = 8                         # 注意力头数
    num_layers = 6                    # Transformer层数
    dropout = 0.1
    
    # 训练参数
    epochs = 100
    lr = 0.0001
    save_path = "./music_transformer.pth"

# ----------------------
# 2. 数据预处理模块
# ----------------------
class MIDIDataset(Dataset):
    def __init__(self, config):
        self.seq_length = config.seq_length
        self.data = self._load_midi(config.midi_dir)
    
    def _load_midi(self, path):
        """加载MIDI并转换为特征序列"""
        all_notes = []
        for file in os.listdir(path):
            midi = PrettyMIDI(os.path.join(path, file))
            for inst in midi.instruments:
                notes = sorted(inst.notes, key=lambda x: x.start)
                # 转换为[pitch, velocity, start, duration]
                seq = [[n.pitch, n.velocity, n.start, n.end-n.start] for n in notes]
                all_notes.extend(seq)
        return self._normalize(np.array(all_notes))
    
    def _normalize(self, data):
        """归一化处理"""
        data[:, 1] = data[:, 1] / 127.0      # 力度
        data[:, 2] = data[:, 2] - data[0, 2] # 相对开始时间
        data[:, 3] = data[:, 3] / 5.0        # 持续时间
        return data
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_length]
        return seq[:-1], seq[-1]  # 输入序列, 目标音符

# ----------------------
# 3. 模型架构模块
# ----------------------
class MusicTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 特征嵌入
        self.pitch_embed = nn.Embedding(128, config.d_model)  # MIDI音高范围0-127
        self.velo_embed = nn.Linear(1, config.d_model)
        self.time_embed = nn.Linear(1, config.d_model)
        self.dur_embed = nn.Linear(1, config.d_model)
        
        # Transformer核心
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # 输出头
        self.pitch_head = nn.Linear(config.d_model, 128)
        self.dur_head = nn.Linear(config.d_model, 1)

    def forward(self, x):
        # 输入x: [batch, seq_len, 4]
        pitch = x[:, :, 0].long()  # [batch, seq_len]
        velo = x[:, :, 1:2]        # [batch, seq_len, 1]
        time = x[:, :, 2:3]
        duration = x[:, :, 3:4]
        
        # 特征嵌入
        embed = (self.pitch_embed(pitch) + 
                self.velo_embed(velo) + 
                self.time_embed(time) + 
                self.dur_embed(duration))
        
        # Transformer处理
        out = self.transformer(embed)
        
        # 多任务输出
        pitch_pred = self.pitch_head(out[:, -1])  # 最后一个位置的预测
        dur_pred = self.dur_head(out[:, -1])
        return pitch_pred, dur_pred

# ----------------------
# 4. 训练流程模块
# ----------------------
def train(config):
    # 初始化
    dataset = MIDIDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = MusicTransformer(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()  # 音高分类任务
    
    # 训练循环
    for epoch in range(config.epochs):
        total_loss = 0.0
        for inputs, targets in dataloader:
            # 数据转换
            inputs = inputs.float().to(device)
            pitch_targets = targets[:, 0].long().to(device)
            
            # 前向传播
            pitch_pred, dur_pred = model(inputs)
            
            # 损失计算
            pitch_loss = criterion(pitch_pred, pitch_targets)
            dur_loss = nn.MSELoss()(dur_pred.squeeze(), targets[:, 3].to(device))
            loss = pitch_loss + dur_loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), config.save_path)

# ----------------------
# 5. 音乐生成模块
# ----------------------
def generate(model: MusicTransformer,
            seed_sequence: List[float],
            config,
            length: int = 100,
            temperature: float = 1.0) -> List[List[float]]:
    """
    Generate music sequence using the trained model.
    
    Args:
        model (MusicTransformer): Trained music transformer model
        seed_sequence (List[float]): Initial sequence to start generation from
        config: Configuration object containing model parameters
        length (int, optional): Length of sequence to generate. Defaults to 100.
        temperature (float, optional): Sampling temperature - higher values make the output more random,
                                     lower values make it more deterministic. Defaults to 1.0.
    
    Returns:
        List[List[float]]: Generated sequence of musical events, where each event is [pitch, duration]
    
    Example:
        >>> model = MusicTransformer(config)
        >>> model.load_state_dict(torch.load('model.pt'))
        >>> seed = [[60, 0.5], [62, 0.25], [64, 0.25]]  # Example seed sequence
        >>> generated = generate(model, seed, config, length=50)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Convert seed sequence to list if it's not already
    current_seq = seed_sequence.copy()
    
    # Move model to appropriate device if not already done
    device = next(model.parameters()).device
    
    # Initialize list to store generated sequence
    generated_notes = []
    
    # Generate new notes
    with torch.no_grad():  # Disable gradient computation for generation
        for i in range(length):
            try:
                # Take last 'seq_length' notes as context (or pad if too short)
                context = current_seq[-config.seq_length:]
                if len(context) < config.seq_length:
                    padding = [[0, 0]] * (config.seq_length - len(context))
                    context = padding + context
                
                # Prepare input tensor
                x = torch.tensor(context, dtype=torch.float32)
                x = x.unsqueeze(0).to(device)  # Add batch dimension and move to device
                
                # Get model predictions
                pitch_logits, duration_logits = model(x)
                
                # Dynamic temperature scaling based on generation progress
                progress = i / length
                dynamic_temp = temperature + (1.0 - temperature) * progress  # Increase temperature as generation progresses
                pitch_logits = pitch_logits[:, -1, :] / dynamic_temp
                duration_logits = duration_logits[:, -1, :] / dynamic_temp
                
                # Convert logits to probabilities
                pitch_probs = torch.softmax(pitch_logits, dim=-1)
                duration_probs = torch.softmax(duration_logits, dim=-1)
                
                # Scheduled sampling: Sometimes choose random notes for diversity
                sample_random = np.random.random() < (0.1 + 0.9 * progress)  # More random as generation progresses
                if sample_random:
                    pitch = np.random.randint(0, 127)
                    duration = np.random.uniform(0.1, 5.0)
                else:
                    pitch = torch.multinomial(pitch_probs, num_samples=1).item()
                    duration = torch.multinomial(duration_probs, num_samples=1).item()
                
                # Create new note
                new_note = [float(pitch), float(duration)]
                
                # Add to generated sequence
                generated_notes.append(new_note)
                current_seq.append(new_note)
                
            except RuntimeError as e:
                print(f"Error during generation at step {len(generated_notes)}: {str(e)}")
                break
                
    return generated_notes

def main():
    config = Config()
    train(config)

if __name__ == "__main__":
    main()