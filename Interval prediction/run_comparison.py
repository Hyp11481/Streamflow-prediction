#增加了特征选择 缓存处理过的上游径流量数据
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
plt.rc("font",family='SimSun ')
plt.rcParams['axes.unicode_minus'] =False
from torch import nn
import seaborn as sns
from datetime import datetime
import os
import math
from scipy.stats import norm
import json
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from vmd_optimizer import optimize_vmd_parameters,weighted_vmd_wavelet_denoise
import logging
from datetime import datetime, timedelta
import warnings
import pywt
from vmdpy import VMD
from minepy import MINE
from sklearn.feature_selection import SelectKBest
warnings.filterwarnings('ignore')

# 设置数据路径常量
RUNOFF_PATH = r"C:\Users\ls\Desktop\实验数据\汉江站点径流量"
METEO_PATH = r"C:\Users\ls\Desktop\实验数据\汉江站点气象数据"

# 站点列表
STATIONS = ["武侯镇","洋县","安康","白河","黄家港(二)","襄阳","宜城","皇庄","沙洋(三)","仙桃(二)"]

# 气象特征列表
METEO_FEATURES = [
    'pressure', 'temp_avg', 'temp_max', 'temp_min',
    'precipitation', 'dewpoint', 'wind_speed', 'wind_v',
    'wind_u', 'solar_net', 'solar_down'
]

processed_stations_cache = {
    'runoff_data': {},       # 存储原始径流量数据
    'denoised_runoff': {},   # 存储降噪后的径流量数据
    'meteo_data': {},        # 存储气象数据
    'vmd_params': {}         # 存储VMD优化参数
}

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 1. 添加PICP和PINAW指标计算函数
def calculate_picp(targets, lower_bounds, upper_bounds):
    """
    计算预测区间覆盖概率 (PICP)
    
    PICP = 1/n * Σ(I[y_i在预测区间内])
    
    Args:
        targets: 真实值数组
        lower_bounds: 预测下界数组
        upper_bounds: 预测上界数组
    
    Returns:
        float: PICP值 (0~1)
    """
    # 确保输入是numpy数组
    targets = np.array(targets)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    # 计算每个真实值是否在预测区间内
    in_interval = np.logical_and(targets >= lower_bounds, targets <= upper_bounds)
    
    # 计算PICP
    picp = np.mean(in_interval)
    
    return picp

def calculate_pinaw(lower_bounds, upper_bounds, targets, alpha=0.05):
    """
    计算预测区间归一化平均宽度 (PINAW)
    
    PINAW = 1/(R*n) * Σ(上界 - 下界)
    
    其中R是目标变量的范围(max-min)
    
    Args:
        lower_bounds: 预测下界数组
        upper_bounds: 预测上界数组
        targets: 真实值数组
        alpha: 显著性水平，默认0.05 (95%置信区间)
    
    Returns:
        float: PINAW值
    """
    # 确保输入是numpy数组
    targets = np.array(targets)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    # 计算区间宽度
    interval_widths = upper_bounds - lower_bounds
    
    # 目标变量范围
    target_range = np.max(targets) - np.min(targets)
    
    # 防止除零
    if target_range == 0:
        return np.nan
    
    # 计算PINAW
    pinaw = np.mean(interval_widths) / target_range
    
    return pinaw

def calculate_interval_score(targets, lower_bounds, upper_bounds, alpha=0.05):
    """
    计算区间得分 (IS)，评估预测区间的质量
    
    Args:
        targets: 真实值数组
        lower_bounds: 预测下界数组
        upper_bounds: 预测上界数组
        alpha: 显著性水平，默认0.05 (95%置信区间)
    
    Returns:
        float: 区间得分，越低越好
    """
    # 确保输入是numpy数组
    targets = np.array(targets)
    lower_bounds = np.array(lower_bounds)
    upper_bounds = np.array(upper_bounds)
    
    # 计算区间宽度
    interval_widths = upper_bounds - lower_bounds
    
    # 计算惩罚项
    below = lower_bounds - targets
    above = targets - upper_bounds
    
    # 对区间外的预测进行惩罚
    penalty_below = 2.0 / alpha * np.maximum(0, below)
    penalty_above = 2.0 / alpha * np.maximum(0, above)
    
    # 区间得分 = 区间宽度 + 惩罚项
    interval_score = interval_widths + penalty_below + penalty_above
    
    # 返回平均区间得分
    return np.mean(interval_score)

def nse_index(observed, simulated):
    """
    计算Nash-Sutcliffe效率系数（NSE）
    
    Nash-Sutcliffe效率系数是水文模型评价中最常用的效率指标之一，用于评估水文模型
    预测结果与实际观测值的匹配程度。
    
    NSE = 1 - [Σ(Qo - Qs)²] / [Σ(Qo - mean(Qo))²]
    
    其中:
    - Qo是观测值
    - Qs是模拟值
    - mean(Qo)是观测值的平均值
    
    NSE的范围是(-∞, 1]，其中:
    - NSE = 1: 完美预测
    - NSE = 0: 模型预测精度等同于使用观测平均值
    - NSE < 0: 模型预测结果比使用观测平均值作为预测还差
    
    通常，NSE > 0.5被认为是可接受的模型性能，NSE > 0.7表示良好的模型性能。
    
    Args:
        observed: 观测值数组
        simulated: 模拟值数组
        
    Returns:
        float: NSE值
    """
    # 如果输入是列表或pandas Series，转换为numpy数组
    observed = np.array(observed)
    simulated = np.array(simulated)
    
    # 过滤掉NaN值
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    observed = observed[mask]
    simulated = simulated[mask]
    
    # 如果没有有效数据，返回NaN
    if len(observed) == 0:
        return np.nan
    
    # 计算NSE
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    
    # 防止除零
    if denominator == 0:
        return np.nan
    
    nse = 1 - (numerator / denominator)
    return nse

def is_power_of_two(num: int):
    return num != 0 and ((num & (num - 1)) == 0)

def adjust_dilations(dilations: list):
    if all([is_power_of_two(i) for i in dilations]):
        return dilations
    else:
        new_dilations = [2 ** i for i in dilations]
        return new_dilations

class ResidualBlock(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 kernel_size,
                 dilation,
                 padding='causal',
                 dropout_rate=0.1,
                 use_batch_norm=True,
                 use_layer_norm=False,
                 use_weight_norm=False):
        super(ResidualBlock, self).__init__()
        
        # 计算因果填充大小
        # 对于因果卷积，填充大小应该是 (kernel_size - 1) * dilation
        self.padding = (kernel_size - 1) * dilation if padding == 'causal' else 0
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            padding=self.padding,
            dilation=dilation
        )
        
        # 添加归一化层
        if use_batch_norm:
            self.norm1 = nn.BatchNorm1d(n_outputs)
            self.norm2 = nn.BatchNorm1d(n_outputs)
        elif use_layer_norm:
            self.norm1 = nn.LayerNorm(n_outputs)
            self.norm2 = nn.LayerNorm(n_outputs)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            
        if use_weight_norm:
            self.conv1 = nn.utils.weight_norm(self.conv1)
            self.conv2 = nn.utils.weight_norm(self.conv2)
            
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 用于残差连接的1x1卷积，当输入输出维度不同时使用
        self.downsample = None
        if n_inputs != n_outputs:
            self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)
    
    def forward(self, x):
        # 保存原始输入用于残差连接
        res = x if self.downsample is None else self.downsample(x)
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.dropout2(out)
        
        # 如果是因果卷积，需要裁剪序列以对齐
        if self.padding > 0:
            # 移除多余的填充
            out = out[:, :, :-self.padding*2]
            res = res[:, :, -out.size(2):]
        
        # 残差连接
        out = out + res
        
        return F.relu(out), out

class TCN(nn.Module):
    def __init__(self,
                 input_dim,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_rate=0.1,
                 use_batch_norm=True,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 output_dim=7):
        super(TCN, self).__init__()
        
        self.use_skip_connections = use_skip_connections
        
        # 初始投影层
        self.input_proj = nn.Conv1d(input_dim, nb_filters, 1)
        
        # TCN残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(
                n_inputs=nb_filters,
                n_outputs=nb_filters,
                kernel_size=kernel_size,
                dilation=d,
                padding=padding,
                dropout_rate=dropout_rate,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_weight_norm=use_weight_norm
            )
            for _ in range(nb_stacks)
            for d in dilations
        ])
        
        # 输出层 - 现在输出均值和方差参数
        self.mean_output = nn.Linear(nb_filters, output_dim)
        self.var_output = nn.Linear(nb_filters, output_dim)
        
    def forward(self, x):
        # 输入shape: [batch_size, seq_len, features]
        # 转换为[batch_size, features, seq_len]以适应Conv1d
        x = x.transpose(1, 2)
        
        # 初始投影
        x = self.input_proj(x)
        
        # 存储跳跃连接
        skip_connections = []
        
        # 通过TCN层
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
            
        # 合并跳跃连接
        if self.use_skip_connections and skip_connections:
            x = sum([s[:, :, -x.size(2):] for s in skip_connections])
        
        # 只使用最后一个时间步进行预测
        x = x[:, :, -1]
        
        # 输出均值和方差参数
        mean = self.mean_output(x)
        # 使用softplus确保方差为正
        var = F.softplus(self.var_output(x)) + 1e-6
        
        return mean, var

class TCNModel(nn.Module):
    def __init__(self, 
                 input_dim,
                 nb_filters=64,
                 kernel_size=3,
                 nb_stacks=2,
                 dilations=(1, 2, 4, 8, 16, 32),
                 dropout_rate=0.1):
        super(TCNModel, self).__init__()
        
        self.tcn = TCN(
            input_dim=input_dim,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            dilations=dilations,
            padding='causal',
            use_skip_connections=True,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            output_dim=7  #7个提前期预测
        )
        
    def forward(self, x):
        return self.tcn(x)

# 1. LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=64,
                 num_layers=2,
                 dropout_rate=0.1):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        # 修改输出层，分别输出均值和方差
        self.mean_output = nn.Linear(hidden_dim, 7)  # 7个提前期的均值预测
        self.var_output = nn.Linear(hidden_dim, 7)   # 7个提前期的方差预测
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, features]
        lstm_out, _ = self.lstm(x)
        
        # 使用最后一个时间步的输出进行预测
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        
        # 输出均值和方差
        mean = self.mean_output(out)
        # 使用softplus确保方差为正
        var = F.softplus(self.var_output(out)) + 1e-6
        
        return mean, var


# 2. GRU模型
class GRUModel(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=64,
                 num_layers=2,
                 dropout_rate=0.1):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        # 修改输出层，分别输出均值和方差
        self.mean_output = nn.Linear(hidden_dim, 7)  # 7个提前期的均值预测
        self.var_output = nn.Linear(hidden_dim, 7)   # 7个提前期的方差预测
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, features]
        gru_out, _ = self.gru(x)
        
        # 使用最后一个时间步的输出进行预测
        last_out = gru_out[:, -1, :]
        out = self.dropout(last_out)
        
        # 输出均值和方差
        mean = self.mean_output(out)
        # 使用softplus确保方差为正
        var = F.softplus(self.var_output(out)) + 1e-6
        
        return mean, var

# 3. CNN-LSTM混合模型
class CNNLSTMModel(nn.Module):
    def __init__(self, 
                 input_dim,
                 cnn_filters=64,
                 lstm_hidden=64,
                 dropout_rate=0.1):
        super(CNNLSTMModel, self).__init__()
        
        # CNN部分
        self.conv1 = nn.Conv1d(input_dim, cnn_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.batch_norm = nn.BatchNorm1d(cnn_filters)
        
        # 计算CNN输出后的序列长度
        # 假设输入序列长度为30，经过一次maxpool后变为15
        cnn_seq_len = 15
        
        # LSTM部分
        self.lstm = nn.LSTM(
            input_size=cnn_filters,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        # 修改输出层，分别输出均值和方差
        self.mean_output = nn.Linear(lstm_hidden, 7)  # 7个提前期的均值预测
        self.var_output = nn.Linear(lstm_hidden, 7)   # 7个提前期的方差预测
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, features]
        # 转置为[batch_size, features, seq_len]以适应Conv1d
        x = x.transpose(1, 2)
        
        # CNN特征提取
        x = F.relu(self.conv1(x))
        x = self.batch_norm(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        
        # 转置回[batch_size, seq_len, features]以适应LSTM
        x = x.transpose(1, 2)
        
        # LSTM处理
        lstm_out, _ = self.lstm(x)
        
        # 使用最后一个时间步的输出进行预测
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        
        # 输出均值和方差
        mean = self.mean_output(out)
        # 使用softplus确保方差为正
        var = F.softplus(self.var_output(out)) + 1e-6
        
        return mean, var

# 4. Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, 
                 input_dim,
                 d_model=64,
                 nhead=8,
                 num_layers=2,
                 dim_feedforward=128,
                 dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        
        # 特征投影层，将输入维度映射到d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout_rate)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层，分别输出均值和方差
        self.dropout = nn.Dropout(dropout_rate)
        self.mean_output = nn.Linear(d_model, 7)  # 7个提前期的均值预测
        self.var_output = nn.Linear(d_model, 7)   # 7个提前期的方差预测
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, features]
        
        # 投影到d_model维度
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        transformer_out = self.transformer_encoder(x)
        
        # 使用最后一个时间步的输出进行预测
        last_out = transformer_out[:, -1, :]
        out = self.dropout(last_out)
        
        # 输出均值和方差
        mean = self.mean_output(out)
        # 使用softplus确保方差为正
        var = F.softplus(self.var_output(out)) + 1e-6
        
        return mean, var


# Transformer位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x形状: [batch_size, seq_len, d_model]
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

# 5. MLP模型 (简单前馈神经网络)
class MLPModel(nn.Module):
    def __init__(self, 
                 input_dim,
                 hidden_dim=128,
                 dropout_rate=0.1):
        super(MLPModel, self).__init__()
        
        # MLP模型需要将时序数据展平
        self.seq_length = 30  # 假设序列长度为30
        flat_dim = input_dim * self.seq_length
        
        # 定义网络层
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # 输出层，分别输出均值和方差
        self.mean_output = nn.Linear(hidden_dim // 2, 7)  # 7个提前期的均值预测
        self.var_output = nn.Linear(hidden_dim // 2, 7)   # 7个提前期的方差预测
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, features]
        batch_size = x.size(0)
        
        # 展平时序数据
        x = x.view(batch_size, -1)
        
        # 前向传播
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        # 输出均值和方差
        mean = self.mean_output(x)
        # 使用softplus确保方差为正
        var = F.softplus(self.var_output(x)) + 1e-6
        
        return mean, var

def create_model(model_type, input_dim, device):
    """
    创建指定类型的模型实例，支持区间预测
    
    参数:
        model_type: 模型类型，可选值包括'tcn', 'lstm', 'gru', 'cnn_lstm', 'transformer', 'mlp'
        input_dim: 输入特征维度
        device: 训练设备
        
    返回:
        model: 创建的模型实例
    """
    if model_type == 'tcn':
        model = TCNModel(
            input_dim=input_dim,
            nb_filters=64,
            kernel_size=3,
            nb_stacks=2,
            dilations=(1, 2, 4, 8, 16, 32),
            dropout_rate=0.1
        )
    elif model_type == 'lstm':
        model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
            dropout_rate=0.1
        )
    elif model_type == 'gru':
        model = GRUModel(
            input_dim=input_dim,
            hidden_dim=64,
            num_layers=2,
            dropout_rate=0.1
        )
    elif model_type == 'cnn_lstm':
        model = CNNLSTMModel(
            input_dim=input_dim,
            cnn_filters=64,
            lstm_hidden=64,
            dropout_rate=0.1
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            input_dim=input_dim,
            d_model=64,
            nhead=4,  # 4个注意力头
            num_layers=2,
            dim_feedforward=128,
            dropout_rate=0.1
        )
    elif model_type == 'mlp':
        model = MLPModel(
            input_dim=input_dim,
            hidden_dim=128,
            dropout_rate=0.1
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    return model.to(device)
    
class HydrologyDataset(Dataset):
    def __init__(self, features, targets, seq_length=30):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.seq_length = seq_length
        self.forecast_days = [1, 2, 3, 4, 5, 8, 10]  # 定义预测天数
        
    def __len__(self):
        return len(self.features) - self.seq_length - max(self.forecast_days)  # 使用最大预测天数
        
    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = torch.zeros(len(self.forecast_days))  # 创建7维输出
        for i, day in enumerate(self.forecast_days):
            y[i] = self.targets[idx + self.seq_length + day - 1]  # -1 是因为day从1开始
        return x, y

def vmd_wavelet_denoise(signal, K=4, alpha=2000, tau=0, DC=True, init=0, tol=1e-7, wavelet='db4', wavelet_levels=None, mode='soft'):
    """
    使用VMD分解和小波变换相结合的混合降噪方法。
    
    流程:
    1. 首先使用VMD将信号分解为不同频率模态
    2. 对每个模态分别应用小波降噪，采用自适应阈值
    3. 重建净化信号
    
    Args:
        signal (ndarray): 输入的1D信号
        K (int): VMD分解的模态数量
        alpha (float): VMD带宽约束参数
        tau (float): VMD噪声容忍度
        DC (bool): VMD是否包含直流项
        init (int): VMD初始化方法，0用零初始化，1用输入信号的傅里叶变换
        tol (float): VMD收敛容忍度
        wavelet (str): 小波变换使用的母小波类型
        wavelet_levels (int): 小波分解的层数，默认为None自动选择
        mode (str): 小波阈值处理模式，'soft'或'hard'
        
    Returns:
        ndarray: 降噪后的信号
    """
    # 如果信号中存在NaN，用插值填充
    if np.any(np.isnan(signal)):
        nan_indices = np.isnan(signal)
        not_nan_indices = ~nan_indices
        signal = np.interp(
            x=np.arange(len(signal)),
            xp=np.arange(len(signal))[not_nan_indices],
            fp=signal[not_nan_indices]
        )
    
    # 对信号进行标准化处理，便于VMD处理
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std > 0:
        normalized_signal = (signal - signal_mean) / signal_std
    else:
        normalized_signal = signal - signal_mean
    
    # 1. VMD分解
    # 准备VMD参数
    alpha = alpha  # 带宽约束
    tau = tau      # 噪声容忍度
    K = K          # 模态数量
    DC = DC        # 包含直流项
    init = init    # 初始化方法
    tol = tol      # 收敛容忍度
    
    # 执行VMD分解
    try:
        u, u_hat, omega = VMD(normalized_signal, alpha, tau, K, DC, init, tol)
        
        # 2. 对每个模态分别应用小波降噪
        denoised_modes = []
        
        for i, mode_signal in enumerate(u):
            # 根据模态频率自适应选择小波参数
            # 低频模态使用较低阈值，高频模态使用较高阈值
            frequency_ratio = omega[-1, i] / np.max(omega[-1, :])
            
            # 选择小波类型：低频使用对称性好的小波，高频使用短小波
            if frequency_ratio < 0.3:
                mode_wavelet = 'sym8'  # 低频使用对称小波
                threshold_multiplier = 0.6  # 较低阈值以保留趋势
            elif frequency_ratio < 0.7:
                mode_wavelet = wavelet  # 中频使用默认小波
                threshold_multiplier = 1.0  # 标准阈值
            else:
                mode_wavelet = 'db1'  # 高频使用简单小波
                threshold_multiplier = 1.5  # 较高阈值以去除噪声
            
            # 小波分解
            coeffs = pywt.wavedec(mode_signal, mode_wavelet, level=wavelet_levels)
            
            # 基于模态频率的自适应阈值
            # 对于高频模态，我们使用更严格的阈值
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(mode_signal))) * threshold_multiplier
            
            # 对小波系数应用阈值处理
            # 保留第一个近似系数（趋势），只对细节系数去噪
            denoised_coeffs = [coeffs[0]]  # 保留近似系数
            for j in range(1, len(coeffs)):
                denoised_coeffs.append(pywt.threshold(coeffs[j], threshold, mode=mode))
            
            # 小波重构
            denoised_mode = pywt.waverec(denoised_coeffs, mode_wavelet)
            
            # 确保长度匹配
            if len(denoised_mode) != len(mode_signal):
                denoised_mode = denoised_mode[:len(mode_signal)]
                
            denoised_modes.append(denoised_mode)
        
        # 3. 重建信号：对所有模态求和
        denoised_signal = np.sum(denoised_modes, axis=0)
        
        # 反标准化，恢复原始信号的均值和方差
        denoised_signal = denoised_signal * signal_std + signal_mean
        
        # 确保最终信号长度与原始信号匹配
        if len(denoised_signal) != len(signal):
            denoised_signal = denoised_signal[:len(signal)]
            
        return denoised_signal
        
    except Exception as e:
        # 如果VMD分解失败，回退到传统小波降噪
        print(f"VMD分解失败，回退到传统小波降噪: {str(e)}")
        return wavelet_denoise(signal, wavelet=wavelet, level=wavelet_levels)

def wavelet_denoise(signal, wavelet='db1', level=None):
    """
    原始的小波降噪函数，用作备选方案
    """
    # 如果信号中存在 NaN，则用插值法将其填充
    if np.any(np.isnan(signal)):
        nan_indices = np.isnan(signal)
        not_nan_indices = ~nan_indices
        signal = np.interp(
            x=np.arange(len(signal)),
            xp=np.arange(len(signal))[not_nan_indices],
            fp=signal[not_nan_indices]
        )
    
    # 小波分解
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 估计噪声水平
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))  # 确定阈值

    # 对小波系数进行软阈值处理
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # 小波重构
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)

    # 确保返回信号长度与原始信号长度一致
    if len(denoised_signal) != len(signal):
        denoised_signal = denoised_signal[:len(signal)]  # 裁剪
    return denoised_signal

def read_runoff_data(station_name):
    """读取单个站点的径流量数据，不进行降噪处理
    
    Args:
        station_name: 站点名称
    
    Returns:
        DataFrame: 包含站点径流量数据的数据框
    """
    logger.info(f"\n开始读取{station_name}站点的径流量数据...")
    
    station_path = os.path.join(RUNOFF_PATH, station_name)
    all_data = []
    
    # 定义列名映射
    column_names = {
        0: 'lon',
        1: 'lat',
        2: 'runoff',
        3: 'date',
        4: 'remarks'
    }
    
    try:
        for file in sorted(os.listdir(station_path)):
            if file.endswith(('.xlsx', '.xls')):
                file_path = os.path.join(station_path, file)
                
                try:
                    df = pd.read_excel(file_path, names=list(column_names.values()))
                    df['date'] = pd.to_datetime(df['date'])
                    df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
                    df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
                    df['runoff'] = pd.to_numeric(df['runoff'], errors='coerce')
                    
                    all_data.append(df)
                    logger.info(f"成功读取文件：{file}")
                    
                except Exception as e:
                    logger.error(f"处理文件{file}时出错：{str(e)}")
                    continue
        
        if not all_data:
            return None
            
        station_data = pd.concat(all_data, ignore_index=True)
        station_data = station_data.sort_values('date').drop_duplicates(subset=['date'])
        
        return station_data
        
    except Exception as e:
        logger.error(f"读取站点{station_name}径流量数据时出错：{str(e)}")
        return None

def read_meteo_data(station_name):
    """读取单个站点的气象数据
    
    Args:
        station_name: 站点名称
    
    Returns:
        DataFrame: 包含站点气象数据的数据框
    """
    logger.info(f"\n开始读取{station_name}站点的气象数据...")
    
    station_path = os.path.join(METEO_PATH, station_name)
    all_data = []
    
    # 定义列名映射
    column_names = {
        0: 'lon',
        1: 'lat',
        2: 'pressure',
        3: 'temp_avg',
        4: 'temp_max',
        5: 'temp_min',
        6: 'precipitation',
        7: 'dewpoint',
        8: 'wind_speed',
        9: 'wind_v',
        10: 'wind_u',
        11: 'solar_net',
        12: 'solar_down',
        13: 'date',
        14: 'remarks'
    }
    
    try:
        for file in sorted(os.listdir(station_path)):
            if file.endswith(('.xlsx', '.xls')):
                file_path = os.path.join(station_path, file)
                
                try:
                    df = pd.read_excel(file_path, names=list(column_names.values()))
                    df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
                    
                    numeric_columns = list(column_names.values())[:-2]
                    for col in numeric_columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    all_data.append(df)
                    logger.info(f"成功读取文件：{file}")
                    
                except Exception as e:
                    logger.error(f"处理文件{file}时出错：{str(e)}")
                    continue
        
        if not all_data:
            return None
            
        station_data = pd.concat(all_data, ignore_index=True)
        station_data = station_data.sort_values('date').drop_duplicates(subset=['date'])
        
        return station_data
        
    except Exception as e:
        logger.error(f"读取站点{station_name}气象数据时出错：{str(e)}")
        return None

def compute_mic_scores(X, y):
    """
    计算每个特征与目标变量之间的最大信息系数(MIC)。
    MIC能够捕获非线性关系，很适合水文数据的特征选择。
    
    Args:
        X: 特征矩阵 [n_samples, n_features]
        y: 目标向量 [n_samples]
        
    Returns:
        mic_scores: 每个特征的MIC分数
    """
    n_features = X.shape[1]
    mic_scores = np.zeros(n_features)
    
    # 创建MINE对象
    mine = MINE(alpha=0.6, c=15)
    
    # 计算每个特征的MIC
    for i in range(n_features):
        mine.compute_score(X[:, i], y)
        mic_scores[i] = mine.mic()
    
    return mic_scores

def select_features_mic(X, y, feature_names, k=None, threshold=0.1):
    """
    基于MIC进行特征选择
    
    Args:
        X: 特征矩阵 [n_samples, n_features]
        y: 目标向量 [n_samples]
        feature_names: 特征名称列表
        k: 选择的特征数量，如果为None则使用threshold
        threshold: MIC阈值，只选择大于此阈值的特征(当k为None时使用)
        
    Returns:
        selected_indices: 选择的特征索引
        selected_names: 选择的特征名称
        mic_scores: 所有特征的MIC分数
    """
    # 计算MIC分数
    mic_scores = compute_mic_scores(X, y)
    
    # 创建特征名称、索引和MIC分数的映射
    feature_info = [(i, name, mic_scores[i]) for i, name in enumerate(feature_names)]
    
    # 按MIC分数降序排序
    sorted_features = sorted(feature_info, key=lambda x: x[2], reverse=True)
    
    # 选择特征
    if k is not None:
        # 选择前k个特征
        selected_features = sorted_features[:k]
    else:
        # 选择MIC大于阈值的特征
        selected_features = [feat for feat in sorted_features if feat[2] > threshold]
    
    # 提取选择的特征索引和名称
    selected_indices = [feat[0] for feat in selected_features]
    selected_names = [feat[1] for feat in selected_features]
    
    # 打印选择的特征及其MIC分数
    print(f"选择了{len(selected_indices)}个特征:")
    for name, score in zip([feat[1] for feat in selected_features], 
                           [feat[2] for feat in selected_features]):
        print(f"{name}: MIC={score:.4f}")
    
    return selected_indices, selected_names, mic_scores

def plot_mic_feature_importance(feature_names, mic_scores, save_path=None, top_n=20):
    """
    可视化特征的MIC分数
    
    Args:
        feature_names: 特征名称列表
        mic_scores: MIC分数列表
        save_path: 图像保存路径，默认为None(不保存)
        top_n: 显示前n个特征
    """
    # 创建特征名称和MIC分数的映射
    feature_mic = list(zip(feature_names, mic_scores))
    
    # 按MIC分数降序排序
    sorted_features = sorted(feature_mic, key=lambda x: x[1], reverse=True)
    
    # 取前top_n个
    if top_n and top_n < len(sorted_features):
        sorted_features = sorted_features[:top_n]
    
    names = [feat[0] for feat in sorted_features]
    scores = [feat[1] for feat in sorted_features]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(names)), scores, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('MIC分数')
    plt.title('基于MIC的特征重要性')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{scores[i]:.4f}', va='center')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.close()

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """训练模型的完整流程，支持区间预测
    
    这个函数实现了神经网络模型的完整训练过程，包括：
    1. 损失函数和优化器的设置
    2. 学习率调度
    3. 训练和验证循环
    4. 早停机制
    5. 模型保存
    
    Args:
        model: 待训练的模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数，默认为100
        device: 训练设备（'cuda'或'cpu'）
        
    Returns:
        训练后的模型实例
    """
    # 设置损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失，适用于回归问题
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,  # 初始学习率
        weight_decay=0.01  # L2正则化系数，防止过拟合
    )
    
    # 设置学习率调度器，使用余弦退火策略以实现更平滑的学习率下降
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,            # 第一次重启的周期
        T_mult=2,          # 每次重启后周期长度的倍乘因子
        eta_min=1e-6,      # 最小学习率
    )
    
    # 初始化早停相关变量
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 15  # 增加容忍验证损失不下降的轮数
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'best_epoch': 0,
        'learning_rates': []
    }
    
    # 开始训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()  # 设置模型为训练模式
        train_loss = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            # 将数据移到指定设备
            features = features.to(device)
            targets = targets.to(device)
            
            # 前向传播
            optimizer.zero_grad()  # 清除先前的梯度
            mean, var = model(features)  # 模型预测，返回均值和方差
            
            # 计算负对数似然损失 (NLL)，适合区间预测
            loss = criterion(mean, targets)  # 仅使用均值进行训练
            
            # 反向传播
            loss.backward()  # 计算梯度
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # 更新模型参数
            
            train_loss += loss.item()
            
        # 每个epoch结束后更新学习率
        scheduler.step()
            
        # 计算平均训练损失
        train_loss /= len(train_loader)
        training_history['train_losses'].append(train_loss)
        training_history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        # 验证阶段
        model.eval()  # 设置模型为评估模式
        val_loss = 0
        
        with torch.no_grad():  # 不计算梯度
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                mean, var = model(features)  # 模型预测，返回均值和方差
                loss = criterion(mean, targets)  # 仅使用均值进行验证
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        training_history['val_losses'].append(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            logger.info(f'验证损失改善: {best_val_loss:.6f} -> {val_loss:.6f}')
            best_val_loss = val_loss
            early_stopping_counter = 0
            training_history['best_epoch'] = epoch
            
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'training_history': training_history
            }, 'best_model.pth')
        else:
            early_stopping_counter += 1
            
        # 检查是否触发早停
        if early_stopping_counter >= patience:
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            logger.info(f'Best validation loss: {best_val_loss:.6f} at epoch {training_history["best_epoch"]}')
            break
            
        # 定期输出训练信息
        if (epoch + 1) % 5 == 0:
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            logger.info(f'Train Loss: {train_loss:.6f}')
            logger.info(f'Val Loss: {val_loss:.6f}')
            logger.info(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
    # 加载最佳模型
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info('\n训练完成!')
    logger.info(f'最佳验证损失: {best_val_loss:.6f}')
    logger.info(f'最佳轮次: {training_history["best_epoch"]}')
    
    return model, training_history

def plot_prediction_results_with_intervals(results_df, station_name, suffix, save_path):
    """绘制改进的预测结果对比图，包含区间预测和NSE"""
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形和轴对象
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
    
    # 设置主题风格
    plt.style.use('seaborn')
    
    # 更新预测期映射字典
    days_map = {
        '1d': '1天',
        '2d': '2天',
        '3d': '3天',
        '4d': '4天',
        '5d': '5天',
        '8d': '8天',
        '10d': '10天'
    }
    
    # 绘制第一个子图：预测值与实际值对比，加上预测区间
    ax1.plot(results_df['date'], results_df['actual'], 
            label='实际值', color='#2878B5', linewidth=2, alpha=0.8)
    ax1.plot(results_df['date'], results_df['predicted'], 
            label='预测值', color='#F8766D', linewidth=2, 
            linestyle='--', alpha=0.8)
    
    # 添加预测区间
    ax1.fill_between(results_df['date'], 
                    results_df['lower_bound'],
                    results_df['upper_bound'],
                    color='#F8766D', alpha=0.2, label='预测区间')
    
    # 设置网格线
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 设置标题和标签
    title = f'{station_name}站点 {days_map.get(suffix, suffix)} 预测结果对比'
    ax1.set_title(title, fontsize=16, pad=20)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('径流量(m³/s)', fontsize=12)
    
    # 优化图例
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax1.tick_params(axis='x', rotation=45)
    
    # 绘制误差图
    error = results_df['predicted'] - results_df['actual']
    ax2.fill_between(results_df['date'], error, 
                     color='#9AC9DB', alpha=0.5, label='预测误差')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 设置误差图的标签
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('误差(m³/s)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['predicted']))
    r2 = r2_score(results_df['actual'], results_df['predicted'])
    
    # 添加新的指标：NSE
    nse = nse_index(results_df['actual'], results_df['predicted'])
    
    
    # 计算区间预测指标
    picp = calculate_picp(results_df['actual'], results_df['lower_bound'], results_df['upper_bound'])
    pinaw = calculate_pinaw(results_df['lower_bound'], results_df['upper_bound'], results_df['actual'])
    
    # 添加统计信息，包括NSE和区间预测指标
    stats_text = f'RMSE: {rmse:.2f}\nR²: {r2:.3f}\nNSE: {nse:.3f}\nPICP: {picp:.3f}\nPINAW: {pinaw:.3f}'
    ax1.text(0.02, 0.98, stats_text, 
             transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top',
             fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_models_comparison_with_intervals(dates, actual_values, model_predictions, model_lower_bounds, model_upper_bounds, 
                                         station_name, days, model_types, save_path):
    """
    绘制多个模型的预测结果对比图，包含预测区间
    
    Args:
        dates: 日期序列
        actual_values: 实际值数组
        model_predictions: 字典，包含各模型的预测均值
        model_lower_bounds: 字典，包含各模型的预测下界
        model_upper_bounds: 字典，包含各模型的预测上界
        station_name: 站点名称
        days: 预测天数
        model_types: 模型类型列表
        save_path: 图像保存路径
    """
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(15, 10))
    
    # 绘制实际值
    plt.plot(dates, actual_values, 'k-', label='实际值', linewidth=2)
    
    # 为每种模型绘制预测值和区间
    colors = ['#2878B5', '#F8766D', '#00BA38', '#C77CFF', '#FFA500', '#8B4513']
    for i, model_type in enumerate(model_types):
        color = colors[i % len(colors)]
        
        # 绘制预测均值
        plt.plot(dates, model_predictions[model_type], 
                linestyle='--', color=color, 
                label=f'{model_type.upper()}预测值', linewidth=1.5, alpha=0.8)
        
        # 绘制预测区间
        plt.fill_between(dates, 
                         model_lower_bounds[model_type],
                         model_upper_bounds[model_type],
                         color=color, alpha=0.1)
    
    plt.title(f'{station_name}站点各模型{days}天预测对比（含预测区间）', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('径流量(m³/s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='best', fontsize=12)
    plt.xticks(rotation=45)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_interval_metrics_comparison(metrics_df, station_name, suffix, save_path):
    """
    绘制区间预测指标的对比条形图
    
    Args:
        metrics_df: 包含各模型指标的DataFrame
        station_name: 站点名称
        suffix: 预测期后缀
        save_path: 图像保存路径
    """
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 模型名称和对应的颜色
    models = metrics_df.index
    colors = ['#2878B5', '#F8766D', '#00BA38', '#C77CFF', '#FFA500', '#8B4513']
    
    # PICP条形图
    picp_values = metrics_df['PICP'].values
    bars1 = ax1.bar(models, picp_values, color=colors[:len(models)])
    ax1.axhline(y=0.95, color='red', linestyle='--', label='目标覆盖率 (95%)')
    ax1.set_ylim(0, 1.05)
    ax1.set_title(f'{station_name}站点各模型{suffix}预测PICP对比', fontsize=14)
    ax1.set_ylabel('PICP', fontsize=12)
    ax1.legend()
    
    # 为每个柱子添加数值标签
    for i, v in enumerate(picp_values):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # PINAW条形图
    pinaw_values = metrics_df['PINAW'].values
    bars2 = ax2.bar(models, pinaw_values, color=colors[:len(models)])
    ax2.set_title(f'{station_name}站点各模型{suffix}预测PINAW对比', fontsize=14)
    ax2.set_ylabel('PINAW', fontsize=12)
    
    # 为每个柱子添加数值标签
    for i, v in enumerate(pinaw_values):
        ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_predictions(model, test_loader, device='cuda', alpha=0.05):
    """评估模型预测结果，包含区间预测"""
    def smape(y_true, y_pred):
        # 过滤全零样本防止除零
        mask = (np.abs(y_true) + np.abs(y_pred)) > 1e-6
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        return 200 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

    model.eval()
    all_means = []
    all_vars = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            mean, var = model(features)
            
            all_means.extend(mean.cpu().numpy())
            all_vars.extend(var.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    means = np.array(all_means)
    vars = np.array(all_vars)
    actual_values = np.array(all_targets)
    
    # 计算预测区间
    z_value = norm.ppf(1 - alpha/2)  # 对应于给定的置信水平
    lower_bounds = means - z_value * np.sqrt(vars)
    upper_bounds = means + z_value * np.sqrt(vars)
    
    # 计算总体评估指标
    metrics = {
        "总体评估指标": {
            "MSE": float(mean_squared_error(actual_values, means)),
            "RMSE": float(np.sqrt(mean_squared_error(actual_values, means))),
            "MAE": float(mean_absolute_error(actual_values, means)),
            "R2": float(r2_score(actual_values.flatten(), means.flatten())),
            "sMAPE": float(smape(actual_values.flatten(), means.flatten())),
            "NSE": float(nse_index(actual_values.flatten(), means.flatten())),
            "PICP": float(calculate_picp(actual_values.flatten(), lower_bounds.flatten(), upper_bounds.flatten())),
            "PINAW": float(calculate_pinaw(lower_bounds.flatten(), upper_bounds.flatten(), actual_values.flatten())),
            "IS": float(calculate_interval_score(actual_values.flatten(), lower_bounds.flatten(), upper_bounds.flatten(), alpha))
        }
    }
    
    # 计算各预测期的指标
    periods = [
        (0, "1天"), (1, "2天"), (2, "3天"), (3, "4天"), 
        (4, "5天"), (5, "8天"), (6, "10天")
    ]
    period_metrics = {}
    
    for idx, name in periods:
        period_means = means[:, idx]
        period_vars = vars[:, idx]
        period_actuals = actual_values[:, idx]
        period_lower = lower_bounds[:, idx]
        period_upper = upper_bounds[:, idx]
        
        period_metrics[f"{name}预测"] = {
            "MSE": float(mean_squared_error(period_actuals, period_means)),
            "RMSE": float(np.sqrt(mean_squared_error(period_actuals, period_means))),
            "MAE": float(mean_absolute_error(period_actuals, period_means)),
            "R2": float(r2_score(period_actuals, period_means)),
            "sMAPE": float(smape(period_actuals, period_means)),
            "NSE": float(nse_index(period_actuals, period_means)),
            "PICP": float(calculate_picp(period_actuals, period_lower, period_upper)),
            "PINAW": float(calculate_pinaw(period_lower, period_upper, period_actuals)),
            "IS": float(calculate_interval_score(period_actuals, period_lower, period_upper, alpha))
        }
    
    metrics["各提前期评估指标"] = period_metrics
    return means, actual_values, lower_bounds, upper_bounds, metrics

def plot_prediction_results(results_df, station_name, suffix, save_path):
    """绘制改进的预测结果对比图，包含NSE"""
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形和轴对象
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[3, 1])
    
    # 设置主题风格
    plt.style.use('seaborn')
    
    # 更新预测期映射字典
    days_map = {
        '1d': '1天',
        '2d': '2天',
        '3d': '3天',
        '4d': '4天',
        '5d': '5天',
        '8d': '8天',
        '10d': '10天'
    }
    
    # 绘制第一个子图：预测值与实际值对比
    ax1.plot(results_df['date'], results_df['actual'], 
            label='实际值', color='#2878B5', linewidth=2, alpha=0.8)
    ax1.plot(results_df['date'], results_df['predicted'], 
            label='预测值', color='#F8766D', linewidth=2, 
            linestyle='--', alpha=0.8)
    
    # 设置网格线
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 设置标题和标签
    title = f'{station_name}站点 {days_map.get(suffix, suffix)} 预测结果对比'
    ax1.set_title(title, fontsize=16, pad=20)
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('径流量(m³/s)', fontsize=12)
    
    # 优化图例
    ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax1.tick_params(axis='x', rotation=45)
    
    # 绘制误差图
    error = results_df['predicted'] - results_df['actual']
    ax2.fill_between(results_df['date'], error, 
                     color='#9AC9DB', alpha=0.5, label='预测误差')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # 设置误差图的标签
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('误差(m³/s)', fontsize=12)
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(results_df['actual'], results_df['predicted']))
    r2 = r2_score(results_df['actual'], results_df['predicted'])
    
    # 添加新的指标：NSE
    nse = nse_index(results_df['actual'], results_df['predicted'])
    
    # 添加统计信息，包括NSE
    stats_text = f'RMSE: {rmse:.2f}\nR²: {r2:.3f}\nNSE: {nse:.3f}\n'
    ax1.text(0.02, 0.98, stats_text, 
             transform=ax1.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'),
             verticalalignment='top',
             fontsize=12)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_station_with_model_comparison(station_name, upstream_stations, 
                                         model_types=None, 
                                         alpha=0.05):
    """处理单个站点的预测任务，对比不同模型的性能，带缓存机制，避免重复处理上游站点数据
    支持区间预测，不使用任何降噪方法，直接使用原始数据
    
    Args:
        station_name: 目标站点名称
        upstream_stations: 上游站点列表
        model_types: 要对比的模型类型列表，默认为['tcn', 'lstm', 'gru', 'cnn_lstm', 'transformer', 'mlp']
        alpha: 预测区间显著性水平，默认0.05 (95%置信区间)
    
    Returns:
        comparative_metrics: 不同模型的对比指标
    """
    global processed_stations_cache
    
    # 如果未指定模型类型，使用默认的模型列表
    if model_types is None:
        model_types = ['tcn', 'lstm', 'gru', 'cnn_lstm', 'transformer', 'mlp']
    
    logger.info(f"\n{'='*50}")
    logger.info(f"开始处理站点: {station_name}")
    logger.info(f"将对比以下模型: {', '.join(model_types)}")
    logger.info(f"预测区间显著性水平: {alpha} ({(1-alpha)*100:.0f}%置信区间)")
    logger.info("注意：本次试验不使用任何降噪方法，直接使用原始径流量数据")
    
    try:
        # 创建结果保存目录
        station_dir = f"results_{station_name}_model_comparison_interval_no_denoising"
        os.makedirs(station_dir, exist_ok=True)
        
        # 读取目标站点的径流量数据（仅用作标签）
        if station_name in processed_stations_cache['runoff_data']:
            logger.info(f"从缓存中获取{station_name}站点的径流量数据")
            target_runoff_data = processed_stations_cache['runoff_data'][station_name].copy()
        else:
            target_runoff_data = read_runoff_data(station_name)
            if target_runoff_data is not None:
                processed_stations_cache['runoff_data'][station_name] = target_runoff_data.copy()
            
        if target_runoff_data is None:
            logger.error(f"目标站点{station_name}径流量数据读取失败")
            return None
        
        # 初始化一个空的DataFrame来存储所有上游站点数据
        all_data = pd.DataFrame()
        all_data['date'] = target_runoff_data['date']
        logger.info(f"开始处理上游站点数据，共{len(upstream_stations)}个上游站点")
        
        # 读取上游站点的径流量数据和气象数据，不进行降噪处理
        for upstream_station in upstream_stations:
            logger.info(f"\n处理上游站点: {upstream_station}")
            
            # 检查是否已经缓存了该上游站点的原始径流量数据
            if upstream_station in processed_stations_cache['runoff_data']:
                logger.info(f"从缓存获取{upstream_station}站点的原始径流量数据")
                upstream_runoff_data = processed_stations_cache['runoff_data'][upstream_station].copy()
            else:
                upstream_runoff_data = read_runoff_data(upstream_station)
                if upstream_runoff_data is not None:
                    processed_stations_cache['runoff_data'][upstream_station] = upstream_runoff_data.copy()
            
            # 如果有径流量数据，合并到all_data
            if upstream_station in processed_stations_cache['runoff_data']:
                processed_runoff_data = processed_stations_cache['runoff_data'][upstream_station][['date', 'runoff']].rename(
                    columns={'runoff': f'runoff_{upstream_station}'}
                )
                all_data = pd.merge(all_data, processed_runoff_data, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的原始径流量数据")
            else:
                logger.warning(f"上游站点{upstream_station}径流量数据不可用，将忽略该站点数据")
            
            # 检查是否已经处理过该上游站点的气象数据
            if upstream_station in processed_stations_cache['meteo_data']:
                logger.info(f"从缓存获取{upstream_station}站点的气象数据")
                upstream_meteo_data = processed_stations_cache['meteo_data'][upstream_station].copy()
            else:
                # 读取上游站点的气象数据
                upstream_meteo_data = read_meteo_data(upstream_station)
                if upstream_meteo_data is not None:
                    # 缓存气象数据
                    processed_stations_cache['meteo_data'][upstream_station] = upstream_meteo_data.copy()
            
            # 如果有气象数据，合并到all_data
            if upstream_station in processed_stations_cache['meteo_data']:
                meteo_data_to_merge = processed_stations_cache['meteo_data'][upstream_station][['date'] + METEO_FEATURES].rename(
                    columns={col: f'{col}_{upstream_station}' for col in METEO_FEATURES}
                )
                all_data = pd.merge(all_data, meteo_data_to_merge, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的气象数据")
            else:
                logger.warning(f"上游站点{upstream_station}气象数据不可用，将忽略该站点数据")
        
        # 添加时间特征
        all_data['month_sin'] = np.sin(2 * np.pi * all_data['date'].dt.month / 12)
        all_data['month_cos'] = np.cos(2 * np.pi * all_data['date'].dt.month / 12)
        all_data['day_sin'] = np.sin(2 * np.pi * all_data['date'].dt.day / 31)
        all_data['day_cos'] = np.cos(2 * np.pi * all_data['date'].dt.day / 31)
        
        # 为每个上游站点创建特征
        for upstream_station in upstream_stations:
            # 创建上游站点径流量的滞后特征
            runoff_col = f'runoff_{upstream_station}'
            if runoff_col in all_data.columns:
                for lag in range(1, 8):
                    all_data[f'{runoff_col}_lag_{lag}'] = all_data[runoff_col].shift(lag)
            
            # 创建上游站点气象数据的滞后特征
            for col in METEO_FEATURES:
                col_name = f'{col}_{upstream_station}'
                if col_name in all_data.columns:
                    for lag in range(1, 8):
                        all_data[f'{col_name}_lag_{lag}'] = all_data[col_name].shift(lag)
                    
                    # 为降水量创建累积特征
                    if col == 'precipitation':
                        all_data[f'precip_sum_7d_{upstream_station}'] = all_data[col_name].rolling(window=7).sum()
                        all_data[f'precip_sum_30d_{upstream_station}'] = all_data[col_name].rolling(window=30).sum()

        # 合并目标变量（仅用于训练和评估）
        all_data = pd.merge(
            all_data,
            target_runoff_data[['date', 'runoff']],
            on='date',
            how='inner'
        )
        
        # 清理缺失值
        all_data = all_data.dropna()
        logger.info(f"数据预处理完成，最终样本数量: {len(all_data)}")
        
        # 准备特征和目标
        feature_cols = [col for col in all_data.columns if col not in ['date', 'runoff']]
        features = all_data[feature_cols].values
        targets = all_data['runoff'].values

        # 使用MIC进行特征选择
        logger.info("\n开始使用MIC进行特征选择...")
        mic_dir = os.path.join(station_dir, "mic_analysis")
        os.makedirs(mic_dir, exist_ok=True)

        # 为了避免数据泄露，只在训练集上进行特征选择
        train_end = int(0.7 * len(features))
        X_train, y_train = features[:train_end], targets[:train_end]

        # 进行基于MIC的特征选择
        selected_indices, selected_names, mic_scores = select_features_mic(
            X_train, y_train, feature_cols, 
            k=None,  # 使用阈值而不是固定数量
            threshold=0.1  # MIC阈值，可以根据具体数据调整
        )

        # 绘制特征重要性图
        plot_mic_feature_importance(
            feature_cols, mic_scores,
            save_path=os.path.join(mic_dir, f"{station_name}_mic_importance.png"),
            top_n=30  # 显示前30个重要特征
        )

        # 保存MIC分析结果
        mic_results = {
            "feature_names": feature_cols,
            "mic_scores": mic_scores.tolist(),
            "selected_features": selected_names,
            "threshold": 0.1
        }

        with open(os.path.join(mic_dir, f"{station_name}_mic_results.json"), 'w', encoding='utf-8') as f:
            json.dump(mic_results, f, indent=2, ensure_ascii=False)

        # 筛选特征
        features = features[:, selected_indices]
        logger.info(f"特征选择完成，从{len(feature_cols)}个特征中选择了{len(selected_indices)}个特征")

        # 数据标准化
        feature_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        features = feature_scaler.fit_transform(features)
        targets = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # 创建数据集
        dataset = HydrologyDataset(features, targets, seq_length=30)
        
        # 时间顺序划分
        total_len = len(dataset)
        train_end = int(0.7 * total_len)
        val_end = train_end + int(0.15 * total_len)

        # 按时间顺序创建数据集分割
        indices = list(range(total_len))
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # 创建子数据集
        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        test_dataset = torch.utils.data.Subset(dataset, test_indices)

        # 记录划分信息
        logger.info(f"按时间顺序划分数据集:")
        logger.info(f"训练集: {len(train_dataset)}个样本")
        logger.info(f"验证集: {len(val_dataset)}个样本")
        logger.info(f"测试集: {len(test_dataset)}个样本")

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 用于存储各模型的评估结果
        model_metrics = {}
        model_predictions = {}
        model_lower_bounds = {}
        model_upper_bounds = {}
        
        # 对每种模型类型进行训练和评估
        for model_type in model_types:
            logger.info(f"\n开始训练和评估{model_type.upper()}模型...")
            
            # 创建模型
            model = create_model(model_type, len(selected_indices), device)
            
            # 训练模型
            model, training_history = train_model(model, train_loader, val_loader, num_epochs=100, device=device)
            
            # 评估模型
            logger.info(f"开始评估{model_type.upper()}模型...")
            means, actual_values, lower_bounds, upper_bounds, metrics = evaluate_predictions(
                model, test_loader, device=device, alpha=alpha
            )
            
            # 反标准化预测结果
            means = target_scaler.inverse_transform(means)
            actual_values = target_scaler.inverse_transform(actual_values)
            lower_bounds = target_scaler.inverse_transform(lower_bounds)
            upper_bounds = target_scaler.inverse_transform(upper_bounds)
            
            # 存储模型评估结果
            model_metrics[model_type] = metrics
            model_predictions[model_type] = means
            model_lower_bounds[model_type] = lower_bounds
            model_upper_bounds[model_type] = upper_bounds
            
            # 创建模型结果子目录
            model_dir = os.path.join(station_dir, model_type)
            os.makedirs(model_dir, exist_ok=True)
            
            # 创建可视化目录
            plots_dir = os.path.join(model_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # 保存预测结果并生成可视化
            prediction_periods = [
                (1, "1d"), (2, "2d"), (3, "3d"), (4, "4d"), 
                (5, "5d"), (8, "8d"), (10, "10d")
            ]
            
            for idx, (days, suffix) in enumerate(prediction_periods):
                try:
                    # 创建结果DataFrame
                    results_df = pd.DataFrame({
                        'date': all_data['date'].values[-len(means):],
                        'actual': actual_values[:, idx],
                        'predicted': means[:, idx],
                        'lower_bound': lower_bounds[:, idx],
                        'upper_bound': upper_bounds[:, idx],
                        'error': means[:, idx] - actual_values[:, idx]
                    })
                    
                    # 保存预测结果到CSV
                    csv_file = os.path.join(model_dir, f"{station_name}_{model_type}_{suffix}_predictions.csv")
                    results_df.to_csv(csv_file, index=False, encoding='utf-8')
                    
                    # 生成可视化
                    plot_file = os.path.join(plots_dir, f"{station_name}_{model_type}_{suffix}_plot.png")
                    plot_prediction_results_with_intervals(
                        results_df=results_df,
                        station_name=f"{station_name} ({model_type.upper()})",
                        suffix=suffix,
                        save_path=plot_file
                    )
                    
                    logger.info(f"成功生成{model_type}模型{days}天预测的图像和CSV文件")
                except Exception as e:
                    logger.error(f"处理{model_type}模型{days}天预测结果时出错: {str(e)}")
                
                # 输出每个时间尺度的预测指标
                logger.info(f"\n{model_type.upper()}模型{days}天预测结果:")
                logger.info(f"RMSE: {metrics['各提前期评估指标'][f'{days}天预测']['RMSE']:.4f}")
                logger.info(f"R2: {metrics['各提前期评估指标'][f'{days}天预测']['R2']:.4f}")
                logger.info(f"NSE: {metrics['各提前期评估指标'][f'{days}天预测']['NSE']:.4f}")
                logger.info(f"PICP: {metrics['各提前期评估指标'][f'{days}天预测']['PICP']:.4f}")
                logger.info(f"PINAW: {metrics['各提前期评估指标'][f'{days}天预测']['PINAW']:.4f}")
        
        # 生成模型对比报告
        logger.info("\n生成模型对比报告...")
        
        # 创建对比目录
        comparison_dir = os.path.join(station_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        # 生成各预测期的模型对比图
        for idx, (days, suffix) in enumerate(prediction_periods):
            try:
                # 创建对比折线图（包含预测区间）
                dates = all_data['date'].values[-len(model_predictions[model_types[0]]):]
                
                plot_file = os.path.join(comparison_dir, f"{station_name}_models_comparison_{suffix}.png")
                plot_models_comparison_with_intervals(
                    dates=dates,
                    actual_values=actual_values[:, idx],
                    model_predictions={model: predictions[:, idx] for model, predictions in model_predictions.items()},
                    model_lower_bounds={model: bounds[:, idx] for model, bounds in model_lower_bounds.items()},
                    model_upper_bounds={model: bounds[:, idx] for model, bounds in model_upper_bounds.items()},
                    station_name=station_name,
                    days=days,
                    model_types=model_types,
                    save_path=plot_file
                )
                
                logger.info(f"成功生成{days}天预测的模型对比图")
                
                # 创建指标对比表格
                metrics_comparison = {}
                for model_type in model_types:
                    metrics_dict = model_metrics[model_type]['各提前期评估指标'][f'{days}天预测']
                    metrics_comparison[model_type] = {
                        'RMSE': metrics_dict['RMSE'],
                        'MAE': metrics_dict['MAE'],
                        'R2': metrics_dict['R2'],
                        'NSE': metrics_dict['NSE'],
                        'sMAPE': metrics_dict['sMAPE'],
                        'PICP': metrics_dict['PICP'],
                        'PINAW': metrics_dict['PINAW'],
                        'IS': metrics_dict['IS']
                    }
                
                # 转换为DataFrame便于保存
                metrics_df = pd.DataFrame(metrics_comparison).T
                
                # 添加排名信息
                for col in metrics_df.columns:
                    # 对RMSE, MAE, sMAPE, PINAW, IS是越小越好
                    if col in ['RMSE', 'MAE', 'sMAPE', 'PINAW', 'IS']:
                        metrics_df[f'{col}_rank'] = metrics_df[col].rank()
                    # 对R2, NSE, PICP是越大越好
                    else:
                        metrics_df[f'{col}_rank'] = metrics_df[col].rank(ascending=False)
                
                # 计算平均排名
                rank_cols = [col for col in metrics_df.columns if col.endswith('_rank')]
                metrics_df['avg_rank'] = metrics_df[rank_cols].mean(axis=1)
                
                # 保存指标比较表
                csv_file = os.path.join(comparison_dir, f"{station_name}_metrics_comparison_{suffix}.csv")
                metrics_df.to_csv(csv_file, encoding='utf-8')
                
                # 生成区间预测指标对比图
                interval_metrics_df = metrics_df[['PICP', 'PINAW']]
                interval_plot_file = os.path.join(comparison_dir, f"{station_name}_interval_metrics_{suffix}.png")
                plot_interval_metrics_comparison(
                    metrics_df=interval_metrics_df,
                    station_name=station_name,
                    suffix=f"{days}天",
                    save_path=interval_plot_file
                )
                
                # 生成指标对比条形图
                plt.figure(figsize=(18, 12))
                
                # 创建多个子图
                metrics_to_plot = ['RMSE', 'R2', 'NSE', 'PICP', 'PINAW']
                n_metrics = len(metrics_to_plot)
                fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5*n_metrics))
                
                # 颜色方案
                colors = ['#2878B5', '#F8766D', '#00BA38', '#C77CFF', '#FFA500', '#8B4513']
                
                for i, metric in enumerate(metrics_to_plot):
                    ax = axes[i]
                    values = [metrics_comparison[model_type][metric] for model_type in model_types]
                    bars = ax.bar(model_types, values, color=colors[:len(model_types)])
                    
                    # 设置标题和标签
                    ax.set_title(f'{days}天预测 - {metric}对比', fontsize=14)
                    ax.set_ylabel(metric, fontsize=12)
                    
                    # 为条形添加数值标签
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.4f}', ha='center', va='bottom', fontsize=10)
                    
                    # 设置网格
                    ax.grid(True, linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                barplot_file = os.path.join(comparison_dir, f"{station_name}_metrics_barplot_{suffix}.png")
                plt.savefig(barplot_file, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.error(f"生成{days}天预测的模型对比图时出错: {str(e)}")
        
        # 创建总体模型评估对比表
        overall_comparison = {}
        for model_type in model_types:
            overall_comparison[model_type] = {
                'RMSE': model_metrics[model_type]['总体评估指标']['RMSE'],
                'MAE': model_metrics[model_type]['总体评估指标']['MAE'],
                'R2': model_metrics[model_type]['总体评估指标']['R2'],
                'NSE': model_metrics[model_type]['总体评估指标']['NSE'],
                'sMAPE': model_metrics[model_type]['总体评估指标']['sMAPE'],
                'PICP': model_metrics[model_type]['总体评估指标']['PICP'],
                'PINAW': model_metrics[model_type]['总体评估指标']['PINAW'],
                'IS': model_metrics[model_type]['总体评估指标']['IS']
            }
        
        # 转换为DataFrame
        overall_df = pd.DataFrame(overall_comparison).T
        
        # 添加排名信息
        for col in overall_df.columns:
            if col in ['RMSE', 'MAE', 'sMAPE', 'PINAW', 'IS']:
                overall_df[f'{col}_rank'] = overall_df[col].rank()
            else:
                overall_df[f'{col}_rank'] = overall_df[col].rank(ascending=False)
        
        # 计算平均排名
        rank_cols = [col for col in overall_df.columns if col.endswith('_rank')]
        overall_df['avg_rank'] = overall_df[rank_cols].mean(axis=1)
        
        # 找出最佳模型
        best_model = overall_df['avg_rank'].idxmin()
        
        # 保存总体评估表
        overall_csv = os.path.join(comparison_dir, f"{station_name}_overall_comparison.csv")
        overall_df.to_csv(overall_csv, encoding='utf-8')
        
        # 生成总体评估区间预测指标对比图
        interval_metrics_df = overall_df[['PICP', 'PINAW']]
        interval_plot_file = os.path.join(comparison_dir, f"{station_name}_overall_interval_metrics.png")
        plot_interval_metrics_comparison(
            metrics_df=interval_metrics_df,
            station_name=station_name,
            suffix="总体",
            save_path=interval_plot_file
        )
        
        # 生成总体评估条形图
        plt.figure(figsize=(18, 12))
        metrics_to_plot = ['RMSE', 'R2', 'NSE', 'PICP', 'PINAW']
        n_metrics = len(metrics_to_plot)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(15, 5*n_metrics))
        
        # 颜色方案
        colors = ['#2878B5', '#F8766D', '#00BA38', '#C77CFF', '#FFA500', '#8B4513']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            values = [overall_comparison[model_type][metric] for model_type in model_types]
            bars = ax.bar(model_types, values, color=colors[:len(model_types)])
            
            ax.set_title(f'总体评估 - {metric}对比', fontsize=14)
            ax.set_ylabel(metric, fontsize=12)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=10)
            
            ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        overall_barplot = os.path.join(comparison_dir, f"{station_name}_overall_metrics_barplot.png")
        plt.savefig(overall_barplot, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 创建汇总报告
        summary = {
            "站点名称": station_name,
            "上游站点数量": len(upstream_stations),
            "上游站点列表": upstream_stations,
            "训练样本数": len(train_dataset),
            "验证样本数": len(val_dataset),
            "测试样本数": len(test_dataset),
            "预测区间显著性水平": alpha,
            "对比模型": model_types,
            "最佳模型": {
                "模型类型": best_model,
                "平均排名": float(overall_df.loc[best_model, 'avg_rank']),
                "总体指标": {
                    metric: float(overall_df.loc[best_model, metric])
                    for metric in ['RMSE', 'MAE', 'R2', 'NSE', 'sMAPE', 'PICP', 'PINAW', 'IS']
                }
            },
            "预测时间范围": {
                "起始": pd.Timestamp(all_data['date'].values[-len(model_predictions[model_types[0]])]).strftime('%Y-%m-%d'),
                "结束": pd.Timestamp(all_data['date'].values[-1]).strftime('%Y-%m-%d')
            },
            "降噪方法": "无降噪，直接使用原始数据"
        }
        
        # 保存汇总报告
        summary_file = os.path.join(station_dir, f"{station_name}_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # 生成区间预测评估报告
        interval_dir = os.path.join(station_dir, "interval_evaluation")
        os.makedirs(interval_dir, exist_ok=True)
        
        # 收集所有提前期的PICP和PINAW值
        picp_values = {}
        pinaw_values = {}
        
        # 预测期列表
        prediction_periods = [
            (1, "1d"), (2, "2d"), (3, "3d"), (4, "4d"), 
            (5, "5d"), (8, "8d"), (10, "10d")
        ]
        
        # 创建图表比较各预测期的区间评估指标（针对最佳模型）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 条形图数据
        period_names = []
        picps = []
        pinaws = []
        
        for idx, (days, suffix) in enumerate(prediction_periods):
            period_metrics = model_metrics[best_model]['各提前期评估指标'][f'{days}天预测']
            picp = period_metrics['PICP']
            pinaw = period_metrics['PINAW']
            
            # 添加到列表
            period_names.append(f"{days}天")
            picps.append(picp)
            pinaws.append(pinaw)
            
            # 记录所有值
            picp_values[f"{days}天"] = picp
            pinaw_values[f"{days}天"] = pinaw
        
        # 绘制PICP条形图
        ax1.bar(period_names, picps, color='skyblue')
        ax1.axhline(y=1-alpha, color='red', linestyle='--', label=f'目标覆盖率 ({(1-alpha)*100:.0f}%)')
        ax1.set_ylim(0, 1.05)
        ax1.set_title(f'{station_name}站点 {best_model.upper()} 模型各预测期PICP (预测区间覆盖概率)')
        ax1.set_xlabel('预测提前期')
        ax1.set_ylabel('PICP')
        ax1.legend()
        
        # 为每个柱子添加数值标签
        for i, v in enumerate(picps):
            ax1.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # 绘制PINAW条形图
        ax2.bar(period_names, pinaws, color='lightgreen')
        ax2.set_title(f'{station_name}站点 {best_model.upper()} 模型各预测期PINAW (预测区间归一化平均宽度)')
        ax2.set_xlabel('预测提前期')
        ax2.set_ylabel('PINAW')
        
        # 为每个柱子添加数值标签
        for i, v in enumerate(pinaws):
            ax2.text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(interval_dir, f"{station_name}_{best_model}_interval_metrics.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算PICP和PINAW的汇总统计
        interval_summary = {
            "站点名称": station_name,
            "最佳模型": best_model,
            "预测区间显著性水平": alpha,
            "目标覆盖率": 1-alpha,
            "平均PICP": np.mean(list(picp_values.values())),
            "最大PICP": np.max(list(picp_values.values())),
            "最小PICP": np.min(list(picp_values.values())),
            "PICP标准差": np.std(list(picp_values.values())),
            "平均PINAW": np.mean(list(pinaw_values.values())),
            "最大PINAW": np.max(list(pinaw_values.values())),
            "最小PINAW": np.min(list(pinaw_values.values())),
            "PINAW标准差": np.std(list(pinaw_values.values())),
            "各预测期指标": {
                period: {
                    "PICP": picp_values[period],
                    "PINAW": pinaw_values[period]
                }
                for period in picp_values.keys()
            }
        }
        
        # 保存区间预测汇总报告
        interval_summary_file = os.path.join(interval_dir, f"{station_name}_interval_summary.json")
        with open(interval_summary_file, 'w', encoding='utf-8') as f:
            json.dump(interval_summary, f, indent=2, ensure_ascii=False)
        
        # 输出总结信息
        logger.info(f"\n站点{station_name}模型对比结果:")
        logger.info(f"最佳模型: {best_model}")
        logger.info(f"总体RMSE: {overall_df.loc[best_model, 'RMSE']:.4f}")
        logger.info(f"总体R2: {overall_df.loc[best_model, 'R2']:.4f}")
        logger.info(f"总体NSE: {overall_df.loc[best_model, 'NSE']:.4f}")
        logger.info(f"总体PICP: {overall_df.loc[best_model, 'PICP']:.4f}")
        logger.info(f"总体PINAW: {overall_df.loc[best_model, 'PINAW']:.4f}")
        logger.info(f"目标覆盖率: {(1-alpha)*100:.0f}%")
        logger.info(f"结果保存在目录: {station_dir}")
        
        return {
            'metrics': model_metrics,
            'best_model': best_model,
            'summary': summary,
            'interval_summary': interval_summary
        }
        
    except Exception as e:
        logger.error(f"处理站点{station_name}时出错: {str(e)}")
        logger.exception("详细错误信息:")
        return None

def compare_models_on_selected_stations(stations_to_compare=None, model_types=None, 
                                   alpha=0.05):
    """对选定的站点运行模型对比试验，支持区间预测
    
    Args:
        stations_to_compare: 要进行对比的站点列表，默认为None（使用全部站点）
        model_types: 要对比的模型类型列表，默认为['tcn', 'lstm', 'gru', 'cnn_lstm', 'transformer', 'mlp']
        nrbo_population: NRBO算法种群大小
        nrbo_iterations: NRBO算法迭代次数
        alpha: 预测区间显著性水平，默认0.05 (95%置信区间)
    
    Returns:
        dict: 包含各站点对比结果的字典
    """
    # 建立站点上下游关系
    station_locations = {
        "白河": {"lon": 110.05, "lat": 32.85},
        "安康": {"lon": 109.05, "lat": 32.75},
        "洋县": {"lon": 107.55, "lat": 33.15},
        "皇庄": {"lon": 112.55, "lat": 31.15},
        "武侯镇": {"lon": 106.65, "lat": 33.15},
        "襄阳": {"lon": 112.15, "lat": 32.05},
        "宜城": {"lon": 112.25, "lat": 31.75},
        "黄家港(二)": {"lon": 111.55, "lat": 32.45},
        "沙洋(三)": {"lon": 112.55, "lat": 30.65},
        "仙桃(二)": {"lon": 113.45, "lat": 30.35}
    }
    
    # 根据经度排序站点
    ordered_stations = sorted(
        station_locations.items(), 
        key=lambda x: x[1]["lon"]
    )
    ordered_stations = [station[0] for station in ordered_stations]
    
    # 建立上游站点字典
    upstream_dict = {}
    for i, station in enumerate(ordered_stations):
        upstream_dict[station] = ordered_stations[:i]
    
    # 如果未指定站点，使用除最上游外的所有站点
    if stations_to_compare is None:
        stations_to_compare = ordered_stations[1:]  # 跳过最上游站点
    
    # 默认模型类型
    if model_types is None:
        model_types = ['tcn', 'lstm', 'gru', 'cnn_lstm', 'transformer', 'mlp']
    
    # 处理每个站点
    all_results = {}
    all_interval_summaries = {}
    
    for station in stations_to_compare:
        logger.info(f"\n开始对站点{station}进行模型对比试验...")
        
        # 确保站点有上游站点
        if station in upstream_dict and upstream_dict[station]:
            results = process_station_with_model_comparison(
                station, 
                upstream_dict[station],
                model_types,
                alpha
            )
            
            if results:
                all_results[station] = results
                all_interval_summaries[station] = results['interval_summary']
                
                # 输出当前站点的最佳模型
                logger.info(f"站点{station}的最佳模型: {results['best_model']}")
                logger.info(f"平均排名: {results['summary']['最佳模型']['平均排名']:.4f}")
                logger.info(f"RMSE: {results['summary']['最佳模型']['总体指标']['RMSE']:.4f}")
                logger.info(f"NSE: {results['summary']['最佳模型']['总体指标']['NSE']:.4f}")
                logger.info(f"PICP: {results['summary']['最佳模型']['总体指标']['PICP']:.4f}")
                logger.info(f"PINAW: {results['summary']['最佳模型']['总体指标']['PINAW']:.4f}")
        else:
            logger.warning(f"站点{station}没有上游站点或不在站点列表中，跳过")
    
    # 创建总体汇总报告
    all_summary = {
        "对比站点": list(all_results.keys()),
        "对比模型": model_types,
        "预测区间显著性水平": alpha,
        "目标覆盖率": 1-alpha,
        "各站点最佳模型": {
            station: {
                "模型类型": results['best_model'],
                "总体指标": results['summary']['最佳模型']['总体指标']
            }
            for station, results in all_results.items()
        },
        "各站点区间预测汇总": all_interval_summaries
    }
    
    # 保存总体汇总报告
    with open("model_comparison_interval_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    
    # 创建所有站点区间预测指标对比图
    stations = list(all_results.keys())
    picps = [all_interval_summaries[station]["平均PICP"] for station in stations]
    pinaws = [all_interval_summaries[station]["平均PINAW"] for station in stations]
    
    plt.figure(figsize=(15, 10))
    
    # PICP对比
    plt.subplot(2, 1, 1)
    bars1 = plt.bar(stations, picps, color='skyblue')
    plt.axhline(y=1-alpha, color='red', linestyle='--', label=f'目标覆盖率 ({(1-alpha)*100:.0f}%)')
    plt.ylim(0, 1.05)
    plt.title(f'各站点预测区间覆盖概率(PICP)对比')
    plt.ylabel('PICP')
    plt.xticks(rotation=45)
    plt.legend()
    
    # 为每个柱子添加数值标签
    for i, v in enumerate(picps):
        plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
    
    # PINAW对比
    plt.subplot(2, 1, 2)
    bars2 = plt.bar(stations, pinaws, color='lightgreen')
    plt.title(f'各站点预测区间归一化平均宽度(PINAW)对比')
    plt.ylabel('PINAW')
    plt.xticks(rotation=45)
    
    # 为每个柱子添加数值标签
    for i, v in enumerate(pinaws):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig("all_stations_interval_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("\n所有站点模型对比试验完成!")
    logger.info(f"对比站点: {', '.join(all_results.keys())}")
    logger.info(f"对比模型: {', '.join(model_types)}")
    
    # 输出各站点最佳模型统计
    model_counts = {}
    for station, results in all_results.items():
        best = results['best_model']
        model_counts[best] = model_counts.get(best, 0) + 1
    
    logger.info("\n各模型作为最佳模型的站点数量:")
    for model, count in model_counts.items():
        logger.info(f"{model}: {count}站点")
    
    # 输出区间预测总体情况
    all_picps = [summary["平均PICP"] for summary in all_interval_summaries.values()]
    all_pinaws = [summary["平均PINAW"] for summary in all_interval_summaries.values()]
    
    logger.info(f"\n区间预测总体情况 (目标覆盖率: {(1-alpha)*100:.0f}%):")
    logger.info(f"平均PICP: {np.mean(all_picps):.4f}")
    logger.info(f"平均PINAW: {np.mean(all_pinaws):.4f}")
    
    return all_results

def run_vmd_parameter_search():
    """运行VMD参数搜索测试，用于单站点验证"""
    # 选择一个测试站点
    test_station = "安康"
    
    # 读取测试站点的径流量数据
    test_data = read_runoff_data(test_station)
    if test_data is None:
        logger.error(f"测试站点{test_station}数据读取失败")
        return
    
    # 获取径流量数据
    runoff_signal = test_data['runoff'].values
    
    # 定义简单预测函数
    def simple_forecast(signal):
        # 使用过去7天的平均值预测
        return np.array([signal[i-7:i].mean() for i in range(7, len(signal))])
    
    # 准备目标数据
    target = runoff_signal[7:]
    
    # 设置优化参数
    population = 20
    max_iter = 30
    
    logger.info(f"开始优化{test_station}站点VMD参数...")
    
    # 运行参数优化
    best_params, best_score, convergence = optimize_vmd_parameters(
        runoff_signal,
        forecast_function=simple_forecast,
        target_signal=target,
        population=population,
        max_iter=max_iter
    )
    
    # 输出结果
    K = int(best_params[0])
    alpha = best_params[1]
    tau = best_params[2]
    weights = best_params[3:3+K]
    
    logger.info(f"最优VMD参数: K={K}, alpha={alpha:.2f}, tau={tau:.4f}")
    logger.info(f"模态权重: {weights}")
    
    # 应用最优参数
    denoised_signal, metrics = weighted_vmd_wavelet_denoise(
        runoff_signal, 
        best_params,
        simple_forecast, 
        target
    )
    
    # 比较结果
    logger.info(f"优化后的预测指标: RMSE={metrics.get('rmse', 'N/A')}, R2={metrics.get('r2', 'N/A')}")
    
    # 创建结果目录
    os.makedirs("vmd_test_results", exist_ok=True)
    
    # 可视化原始信号、降噪后信号和各模态
    plt.figure(figsize=(15, 10))
    
    # 原始信号和降噪后信号
    plt.subplot(2, 1, 1)
    plt.plot(runoff_signal, 'b-', label='原始信号')
    plt.plot(denoised_signal, 'r-', label='降噪后信号')
    plt.title(f'{test_station}站点原始信号与优化VMD降噪后信号对比')
    plt.legend()
    plt.grid(True)
    
    # 收敛曲线
    plt.subplot(2, 1, 2)
    plt.plot(convergence, 'g-')
    plt.title('NRBO优化收敛曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数值')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"vmd_test_results/{test_station}_vmd_optimization.png")
    
    # 保存参数
    params_info = {
        'K': int(K),
        'alpha': float(alpha),
        'tau': float(tau),
        'weights': weights.tolist(),
        'best_score': float(best_score),
        'metrics': metrics
    }
    
    with open(f"vmd_test_results/{test_station}_vmd_params.json", 'w', encoding='utf-8') as f:
        json.dump(params_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"测试结果已保存到vmd_test_results目录")
    
    return params_info
    
if __name__ == "__main__":
    try:
        # 设置随机种子保证可重复性
        np.random.seed(42)
        torch.manual_seed(42)
        
        # 显示所有可用站点
        available_stations = ["武侯镇","洋县","安康","白河","黄家港(二)","襄阳","宜城","皇庄","沙洋(三)","仙桃(二)"]
        available_models = ['tcn', 'lstm', 'gru', 'cnn_lstm', 'transformer', 'mlp']
        
        print("\n可用站点:")
        for i, station in enumerate(available_stations):
            print(f"{i+1}. {station}")
        
        print("\n可用模型:")
        for i, model in enumerate(available_models):
            print(f"{i+1}. {model}")
        
        # 询问用户是否使用区间预测
        use_interval = input("\n是否使用区间预测? (y/n): ").lower() == 'y'
        
        if use_interval:
            alpha = float(input("请输入预测区间显著性水平 (0.05表示95%置信区间): ") or "0.05")
            logger.info(f"已启用区间预测，显著性水平: {alpha} ({(1-alpha)*100:.0f}%置信区间)")
        else:
            alpha = 0.05  # 默认值
            logger.info("未启用区间预测，仅进行点预测")
        
        # 用户选择站点
        print("\n请输入要比较的站点编号，多个站点用逗号分隔，或输入'all'选择全部站点（除最上游站点外）:")
        station_input = input().strip()
        
        if station_input.lower() == 'all':
            selected_stations = available_stations[1:]  # 跳过最上游站点
        else:
            station_indices = [int(idx.strip()) - 1 for idx in station_input.split(',')]
            selected_stations = [available_stations[idx] for idx in station_indices if 0 <= idx < len(available_stations)]
        
        # 用户选择模型
        print("\n请输入要比较的模型编号，多个模型用逗号分隔，或输入'all'选择全部模型:")
        model_input = input().strip()
        
        if model_input.lower() == 'all':
            selected_models = available_models
        else:
            model_indices = [int(idx.strip()) - 1 for idx in model_input.split(',')]
            selected_models = [available_models[idx] for idx in model_indices if 0 <= idx < len(available_models)]
        
        # 用户确认
        print(f"\n已选择的站点: {', '.join(selected_stations)}")
        print(f"已选择的模型: {', '.join(selected_models)}")
        print("\n是否继续运行模型对比试验? (y/n): ")
        
        user_input = input().strip().lower()
        
        if user_input == 'y':
            # 运行模型对比试验
            logger.info("\n开始运行模型对比试验...")
            
            if use_interval:
                # 使用区间预测模式
                results = compare_models_on_selected_stations(
                    stations_to_compare=selected_stations,
                    model_types=selected_models,
                    alpha=alpha
                )
                logger.info("区间预测模型对比试验完成")
            else:
                # 使用点预测模式
                results = compare_models_on_selected_stations(
                    stations_to_compare=selected_stations,
                    model_types=selected_models,
                    nrbo_population=30,
                    nrbo_iterations=30
                )
                logger.info("点预测模型对比试验完成")
        else:
            logger.info("程序已取消")
    
    except Exception as e:
        logger.error("程序执行出错")
        logger.exception("详细错误信息:")