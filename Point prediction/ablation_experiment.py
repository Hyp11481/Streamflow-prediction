# ablation_experiment.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import logging
from datetime import datetime
import warnings
import pywt
import seaborn as sns
from vmdpy import VMD
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from main import (
    TCNModel, HydrologyDataset, nse_index, vmd_wavelet_denoise, 
    wavelet_denoise, read_runoff_data, read_meteo_data, evaluate_predictions,
    plot_prediction_results, select_features_mic, plot_mic_feature_importance
)
from vmd_optimizer import optimize_vmd_parameters, weighted_vmd_wavelet_denoise
from torch.utils.data import DataLoader
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

# 全局缓存（仅用于实验A）
processed_stations_cache = {
    'runoff_data': {},       # 存储原始径流量数据
    'denoised_runoff': {},   # 存储降噪后的径流量数据
    'meteo_data': {},        # 存储气象数据
    'vmd_params': {}         # 存储VMD优化参数
}

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ablation_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda'):
    """训练模型的完整流程
    
    从main2.py复制，使消融实验使用相同的训练过程
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6,
    )
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    patience = 15
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'best_epoch': 0,
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        scheduler.step()
            
        train_loss /= len(train_loader)
        training_history['train_losses'].append(train_loss)
        training_history['learning_rates'].append(scheduler.get_last_lr()[0])
        
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        training_history['val_losses'].append(val_loss)
        
        if val_loss < best_val_loss:
            logger.info(f'验证损失改善: {best_val_loss:.6f} -> {val_loss:.6f}')
            best_val_loss = val_loss
            early_stopping_counter = 0
            training_history['best_epoch'] = epoch
            
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
            
        if early_stopping_counter >= patience:
            logger.info(f'Early stopping triggered at epoch {epoch+1}')
            logger.info(f'Best validation loss: {best_val_loss:.6f} at epoch {training_history["best_epoch"]}')
            break
            
        if (epoch + 1) % 5 == 0:
            logger.info(f'Epoch {epoch+1}/{num_epochs}:')
            logger.info(f'Train Loss: {train_loss:.6f}')
            logger.info(f'Val Loss: {val_loss:.6f}')
            logger.info(f'Learning Rate: {scheduler.get_last_lr()[0]:.6f}')
            
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info('\n训练完成!')
    logger.info(f'最佳验证损失: {best_val_loss:.6f}')
    logger.info(f'最佳轮次: {training_history["best_epoch"]}')
    
    return model, training_history

def process_station_experiment_a(station_name, upstream_stations, nrbo_population=30, nrbo_iterations=50):
    """实验A: 完整模型 (NRBO优化+VMD分解+小波降噪)
    
    与main2.py的process_station_with_vmd_optimization相同，保留缓存机制
    """
    global processed_stations_cache
    
    logger.info(f"\n{'='*50}")
    logger.info(f"实验A: 完整模型 - 开始处理站点: {station_name}")
    
    try:
        # 创建结果保存目录
        station_dir = f"results_A_{station_name}"
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
        
        # 为每个上游站点优化VMD参数
        vmd_params = {}  # 存储每个站点的最优VMD参数
        
        # 读取上游站点的径流量数据和气象数据
        for upstream_station in upstream_stations:
            logger.info(f"\n处理上游站点: {upstream_station}")
            
            # 检查是否已经处理过该上游站点的径流量数据
            if upstream_station in processed_stations_cache['denoised_runoff'] and upstream_station in processed_stations_cache['vmd_params']:
                logger.info(f"从缓存获取{upstream_station}站点的降噪径流量数据和VMD参数")
                upstream_runoff_data = processed_stations_cache['denoised_runoff'][upstream_station].copy()
                vmd_params[upstream_station] = processed_stations_cache['vmd_params'][upstream_station]
            else:
                # 需要处理该上游站点的数据
                if upstream_station in processed_stations_cache['runoff_data']:
                    upstream_runoff_data = processed_stations_cache['runoff_data'][upstream_station].copy()
                else:
                    upstream_runoff_data = read_runoff_data(upstream_station)
                    if upstream_runoff_data is not None:
                        processed_stations_cache['runoff_data'][upstream_station] = upstream_runoff_data.copy()
                
                if upstream_runoff_data is not None:
                    # 获取径流量数据
                    original_runoff = upstream_runoff_data['runoff'].values
                    
                    logger.info(f"开始优化{upstream_station}站点的VMD参数...")
                    
                    # 定义简单预测函数用于优化
                    def simple_prediction(signal):
                        # 一个简单的基于滞后的预测函数
                        return np.array([signal[i-7:i].mean() for i in range(7, len(signal))])
                    
                    # 准备目标信号
                    target_signal = original_runoff[7:]  # 与预测函数匹配
                    
                    # 优化VMD参数
                    best_params, best_score, convergence = optimize_vmd_parameters(
                        original_runoff, 
                        forecast_function=simple_prediction,
                        target_signal=target_signal,
                        population=nrbo_population,
                        max_iter=nrbo_iterations
                    )
                    
                    # 记录优化参数
                    K = int(best_params[0])
                    alpha = best_params[1]
                    tau = best_params[2]
                    weights = best_params[3:3+K]
                    
                    vmd_params[upstream_station] = {
                        'K': K,
                        'alpha': alpha,
                        'tau': tau,
                        'weights': weights.tolist()
                    }
                    
                    # 缓存VMD参数
                    processed_stations_cache['vmd_params'][upstream_station] = vmd_params[upstream_station]
                    
                    logger.info(f"{upstream_station}站点最优VMD参数: K={K}, alpha={alpha:.2f}, tau={tau:.4f}")
                    logger.info(f"模态权重: {weights}")
                    
                    # 使用优化后的参数处理信号
                    try:
                        # 使用带权重的VMD处理
                        denoised_runoff, _ = weighted_vmd_wavelet_denoise(
                            original_runoff, best_params
                        )
                        upstream_runoff_data['runoff'] = denoised_runoff
                        
                        # 缓存降噪后的数据
                        processed_stations_cache['denoised_runoff'][upstream_station] = upstream_runoff_data.copy()
                        
                        logger.info(f"成功使用优化的VMD参数处理{upstream_station}站点的径流量数据")
                    except Exception as e:
                        logger.warning(f"使用优化VMD参数处理{upstream_station}站点数据失败: {str(e)}")
                        # 如果失败，尝试传统方法
                        try:
                            denoised_runoff = wavelet_denoise(original_runoff)
                            upstream_runoff_data['runoff'] = denoised_runoff
                            
                            # 缓存降噪后的数据
                            processed_stations_cache['denoised_runoff'][upstream_station] = upstream_runoff_data.copy()
                            
                            logger.info(f"回退到传统小波降噪，成功处理{upstream_station}站点数据")
                        except Exception as e2:
                            logger.warning(f"传统小波降噪也失败: {str(e2)}")
                    
                    # 保存优化后的VMD参数
                    params_file = os.path.join(station_dir, f"{upstream_station}_vmd_params.json")
                    with open(params_file, 'w', encoding='utf-8') as f:
                        json.dump(vmd_params[upstream_station], f, indent=2, ensure_ascii=False)
                    
                    # 保存收敛曲线
                    plt.figure(figsize=(10, 6))
                    plt.plot(convergence, 'b-')
                    plt.title(f'{upstream_station}站点VMD参数优化收敛曲线')
                    plt.xlabel('迭代次数')
                    plt.ylabel('目标函数值')
                    plt.grid(True)
                    plt.savefig(os.path.join(station_dir, f"{upstream_station}_convergence.png"))
                    plt.close()
            
            # 如果有径流量数据，合并到all_data
            if upstream_station in processed_stations_cache['denoised_runoff']:
                # 从缓存获取处理后的数据
                processed_runoff_data = processed_stations_cache['denoised_runoff'][upstream_station][['date', 'runoff']].rename(
                    columns={'runoff': f'runoff_{upstream_station}'}
                )
                all_data = pd.merge(all_data, processed_runoff_data, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的径流量数据")
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
        
        # 保存所有站点的VMD参数
        all_params_file = os.path.join(station_dir, "all_vmd_params.json")
        with open(all_params_file, 'w', encoding='utf-8') as f:
            json.dump(vmd_params, f, indent=2, ensure_ascii=False)
        
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
        
        # 创建模型
        model = TCNModel(
            input_dim=len(selected_indices),
            nb_filters=64,
            kernel_size=3,
            nb_stacks=2,
            dilations=(1, 2, 4, 8, 16, 32),
            dropout_rate=0.1
        ).to(device)
        
        # 训练模型
        logger.info("\n开始模型训练...")
        model, training_history = train_model(model, train_loader, val_loader, num_epochs=100, device=device)
        
        # 评估模型
        logger.info("\n开始模型评估...")
        predictions, actual_values, metrics = evaluate_predictions(model, test_loader, device=device)
        
        # 反标准化预测结果
        predictions = target_scaler.inverse_transform(predictions)
        actual_values = target_scaler.inverse_transform(actual_values)
        
        # 创建可视化结果目录
        plots_dir = os.path.join(station_dir, "plots")
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
                    'date': all_data['date'].values[-len(predictions):],
                    'actual': actual_values[:, idx],
                    'predicted': predictions[:, idx],
                    'error': predictions[:, idx] - actual_values[:, idx]
                })
                
                # 保存预测结果到CSV
                csv_file = os.path.join(station_dir, f"{station_name}_{suffix}_predictions.csv")
                results_df.to_csv(csv_file, index=False, encoding='utf-8')
                
                # 生成可视化
                plot_file = os.path.join(plots_dir, f"{station_name}_{suffix}_plot.png")
                plot_prediction_results(
                    results_df=results_df,
                    station_name=station_name,
                    suffix=suffix,
                    save_path=plot_file
                )
                
                logger.info(f"成功生成{days}天预测的图像和CSV文件")
            except Exception as e:
                logger.error(f"处理{days}天预测结果时出错: {str(e)}")
            
            # 输出每个时间尺度的预测指标
            logger.info(f"\n{days}天预测结果:")
            logger.info(f"RMSE: {metrics['各提前期评估指标'][f'{days}天预测']['RMSE']:.4f}")
            logger.info(f"R2: {metrics['各提前期评估指标'][f'{days}天预测']['R2']:.4f}")
            logger.info(f"sMAPE: {metrics['各提前期评估指标'][f'{days}天预测']['sMAPE']:.2f}%")
            logger.info(f"NSE: {metrics['各提前期评估指标'][f'{days}天预测']['NSE']:.4f}")
            
        
        # 创建汇总报告，包含VMD优化信息
        summary = {
            "站点名称": station_name,
            "上游站点数量": len(upstream_stations),
            "上游站点列表": upstream_stations,
            "训练样本数": len(train_dataset),
            "验证样本数": len(val_dataset),
            "测试样本数": len(test_dataset),
            "总体评估指标": metrics["总体评估指标"],
            "各时间尺度指标": metrics["各提前期评估指标"],
            "预测时间范围": {
                "起始": pd.Timestamp(all_data['date'].values[-len(predictions)]).strftime('%Y-%m-%d'),
                "结束": pd.Timestamp(all_data['date'].values[-1]).strftime('%Y-%m-%d')
            },
            "VMD参数优化信息": vmd_params
        }
        
        # 保存评估指标和汇总报告
        metrics_file = os.path.join(station_dir, f"{station_name}_metrics.json")
        summary_file = os.path.join(station_dir, f"{station_name}_summary.json")
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # 比较优化前后的结果
        logger.info(f"\n站点{station_name}处理完成!")
        logger.info(f"预测指标:")
        logger.info(f"总体RMSE: {metrics['总体评估指标']['RMSE']:.4f}")
        logger.info(f"总体R2: {metrics['总体评估指标']['R2']:.4f}")
        logger.info(f"总体sMAPE: {metrics['总体评估指标']['sMAPE']:.2f}%")
        logger.info(f"总体NSE: {metrics['总体评估指标']['NSE']:.4f}")
        logger.info(f"结果保存在目录: {station_dir}")
        
        return metrics, vmd_params
        
    except Exception as e:
        logger.error(f"处理站点{station_name}时出错: {str(e)}")
        logger.exception("详细错误信息:")
        return None

def process_station_experiment_b(station_name, upstream_stations):
    """实验B: 固定VMD参数 (移除NRBO优化)
    
    使用固定的VMD参数代替NRBO优化的参数
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"实验B: 固定VMD参数 - 开始处理站点: {station_name}")
    
    try:
        # 创建结果保存目录
        station_dir = f"results_B_{station_name}"
        os.makedirs(station_dir, exist_ok=True)
        
        # 读取目标站点的径流量数据（仅用作标签）
        target_runoff_data = read_runoff_data(station_name)
        if target_runoff_data is None:
            logger.error(f"目标站点{station_name}径流量数据读取失败")
            return None
            
        # 初始化一个空的DataFrame来存储所有上游站点数据
        all_data = pd.DataFrame()
        all_data['date'] = target_runoff_data['date']
        logger.info(f"开始处理上游站点数据，共{len(upstream_stations)}个上游站点")
        
        # 为每个上游站点设置固定的VMD参数
        vmd_params = {}
        
        # 读取上游站点的径流量数据和气象数据
        for upstream_station in upstream_stations:
            logger.info(f"\n处理上游站点: {upstream_station}")
            
            # 读取该上游站点的径流量数据
            upstream_runoff_data = read_runoff_data(upstream_station)
            
            if upstream_runoff_data is not None:
                # 获取径流量数据
                original_runoff = upstream_runoff_data['runoff'].values
                
                # 设置固定的VMD参数 (不通过NRBO优化)
                K = 4  # 固定模态数
                alpha = 2000  # 带宽约束
                tau = 0  # 噪声容忍度
                
                vmd_params[upstream_station] = {
                    'K': K,
                    'alpha': alpha,
                    'tau': tau,
                    'weights': [1.0] * K  # 所有模态权重相等
                }
                
                logger.info(f"{upstream_station}站点固定VMD参数: K={K}, alpha={alpha:.2f}, tau={tau:.4f}")
                
                # 使用固定参数处理信号
                try:
                    # 标准VMD分解 + 小波降噪
                    denoised_runoff = vmd_wavelet_denoise(
                        original_runoff,
                        K=K,
                        alpha=alpha,
                        tau=tau
                    )
                    upstream_runoff_data['runoff'] = denoised_runoff
                    
                    logger.info(f"成功使用固定的VMD参数处理{upstream_station}站点的径流量数据")
                except Exception as e:
                    logger.warning(f"使用VMD处理{upstream_station}站点数据失败: {str(e)}")
                    # 如果失败，尝试传统方法
                    try:
                        denoised_runoff = wavelet_denoise(original_runoff)
                        upstream_runoff_data['runoff'] = denoised_runoff
                        
                        logger.info(f"回退到传统小波降噪，成功处理{upstream_station}站点数据")
                    except Exception as e2:
                        logger.warning(f"传统小波降噪也失败: {str(e2)}")
                
                # 合并径流量数据到all_data
                processed_runoff_data = upstream_runoff_data[['date', 'runoff']].rename(
                    columns={'runoff': f'runoff_{upstream_station}'}
                )
                all_data = pd.merge(all_data, processed_runoff_data, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的径流量数据")
            else:
                logger.warning(f"上游站点{upstream_station}径流量数据不可用，将忽略该站点数据")
            
            # 读取气象数据
            upstream_meteo_data = read_meteo_data(upstream_station)
            
            # 如果有气象数据，合并到all_data
            if upstream_meteo_data is not None:
                meteo_data_to_merge = upstream_meteo_data[['date'] + METEO_FEATURES].rename(
                    columns={col: f'{col}_{upstream_station}' for col in METEO_FEATURES}
                )
                all_data = pd.merge(all_data, meteo_data_to_merge, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的气象数据")
            else:
                logger.warning(f"上游站点{upstream_station}气象数据不可用，将忽略该站点数据")
        
        # 保存所有站点的VMD参数
        all_params_file = os.path.join(station_dir, "all_vmd_params.json")
        with open(all_params_file, 'w', encoding='utf-8') as f:
            json.dump(vmd_params, f, indent=2, ensure_ascii=False)
        
        # 添加时间特征和其他派生特征
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

        # 合并目标变量
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
            k=None,  
            threshold=0.1  
        )

        # 绘制特征重要性图
        plot_mic_feature_importance(
            feature_cols, mic_scores,
            save_path=os.path.join(mic_dir, f"{station_name}_mic_importance.png"),
            top_n=30
        )

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

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model = TCNModel(
            input_dim=len(selected_indices),
            nb_filters=64,
            kernel_size=3,
            nb_stacks=2,
            dilations=(1, 2, 4, 8, 16, 32),
            dropout_rate=0.1
        ).to(device)
        
        # 训练模型
        logger.info("\n开始模型训练...")
        model, training_history = train_model(model, train_loader, val_loader, num_epochs=100, device=device)
        
        # 评估模型
        logger.info("\n开始模型评估...")
        predictions, actual_values, metrics = evaluate_predictions(model, test_loader, device=device)
        
        # 反标准化预测结果
        predictions = target_scaler.inverse_transform(predictions)
        actual_values = target_scaler.inverse_transform(actual_values)
        
        # 创建可视化结果目录
        plots_dir = os.path.join(station_dir, "plots")
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
                    'date': all_data['date'].values[-len(predictions):],
                    'actual': actual_values[:, idx],
                    'predicted': predictions[:, idx],
                    'error': predictions[:, idx] - actual_values[:, idx]
                })
                
                # 保存预测结果到CSV
                csv_file = os.path.join(station_dir, f"{station_name}_{suffix}_predictions.csv")
                results_df.to_csv(csv_file, index=False, encoding='utf-8')
                
                # 生成可视化
                plot_file = os.path.join(plots_dir, f"{station_name}_{suffix}_plot.png")
                plot_prediction_results(
                    results_df=results_df,
                    station_name=station_name,
                    suffix=suffix,
                    save_path=plot_file
                )
                
                logger.info(f"成功生成{days}天预测的图像和CSV文件")
            except Exception as e:
                logger.error(f"处理{days}天预测结果时出错: {str(e)}")
        
        # 创建汇总报告
        summary = {
            "站点名称": station_name,
            "上游站点数量": len(upstream_stations),
            "上游站点列表": upstream_stations,
            "训练样本数": len(train_dataset),
            "验证样本数": len(val_dataset),
            "测试样本数": len(test_dataset),
            "总体评估指标": metrics["总体评估指标"],
            "各时间尺度指标": metrics["各提前期评估指标"],
            "预测时间范围": {
                "起始": pd.Timestamp(all_data['date'].values[-len(predictions)]).strftime('%Y-%m-%d'),
                "结束": pd.Timestamp(all_data['date'].values[-1]).strftime('%Y-%m-%d')
            },
            "信号处理方法": "固定参数VMD+小波降噪"
        }
        
        # 保存评估指标和汇总报告
        metrics_file = os.path.join(station_dir, f"{station_name}_metrics.json")
        summary_file = os.path.join(station_dir, f"{station_name}_summary.json")
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\n站点{station_name}处理完成!")
        logger.info(f"预测指标:")
        logger.info(f"总体RMSE: {metrics['总体评估指标']['RMSE']:.4f}")
        logger.info(f"总体R2: {metrics['总体评估指标']['R2']:.4f}")
        logger.info(f"总体NSE: {metrics['总体评估指标']['NSE']:.4f}")
        logger.info(f"结果保存在目录: {station_dir}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"处理站点{station_name}时出错: {str(e)}")
        logger.exception("详细错误信息:")
        return None

def process_station_experiment_c(station_name, upstream_stations):
    """实验C: 仅使用小波降噪 (移除VMD分解)
    
    只使用小波进行降噪，没有VMD分解
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"实验C: 仅使用小波降噪 - 开始处理站点: {station_name}")
    
    try:
        # 创建结果保存目录
        station_dir = f"results_C_{station_name}"
        os.makedirs(station_dir, exist_ok=True)
        
        # 读取目标站点的径流量数据（仅用作标签）
        target_runoff_data = read_runoff_data(station_name)
        if target_runoff_data is None:
            logger.error(f"目标站点{station_name}径流量数据读取失败")
            return None
            
        # 初始化一个空的DataFrame来存储所有上游站点数据
        all_data = pd.DataFrame()
        all_data['date'] = target_runoff_data['date']
        logger.info(f"开始处理上游站点数据，共{len(upstream_stations)}个上游站点")
        
        # 读取上游站点的径流量数据和气象数据
        for upstream_station in upstream_stations:
            logger.info(f"\n处理上游站点: {upstream_station}")
            
            # 读取该上游站点的径流量数据
            upstream_runoff_data = read_runoff_data(upstream_station)
            
            if upstream_runoff_data is not None:
                # 获取径流量数据
                original_runoff = upstream_runoff_data['runoff'].values
                
                # 仅使用小波降噪，不使用VMD
                try:
                    denoised_runoff = wavelet_denoise(original_runoff, wavelet='db4')
                    upstream_runoff_data['runoff'] = denoised_runoff
                    
                    logger.info(f"成功使用小波降噪处理{upstream_station}站点的径流量数据")
                except Exception as e:
                    logger.warning(f"小波降噪处理{upstream_station}站点数据失败: {str(e)}")
                
                # 合并径流量数据到all_data
                processed_runoff_data = upstream_runoff_data[['date', 'runoff']].rename(
                    columns={'runoff': f'runoff_{upstream_station}'}
                )
                all_data = pd.merge(all_data, processed_runoff_data, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的径流量数据")
            else:
                logger.warning(f"上游站点{upstream_station}径流量数据不可用，将忽略该站点数据")
            
            # 读取气象数据
            upstream_meteo_data = read_meteo_data(upstream_station)
            
            # 如果有气象数据，合并到all_data
            if upstream_meteo_data is not None:
                meteo_data_to_merge = upstream_meteo_data[['date'] + METEO_FEATURES].rename(
                    columns={col: f'{col}_{upstream_station}' for col in METEO_FEATURES}
                )
                all_data = pd.merge(all_data, meteo_data_to_merge, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的气象数据")
            else:
                logger.warning(f"上游站点{upstream_station}气象数据不可用，将忽略该站点数据")
        
        # 添加时间特征和其他派生特征
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

        # 合并目标变量
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
            k=None,  
            threshold=0.1  
        )

        # 绘制特征重要性图
        plot_mic_feature_importance(
            feature_cols, mic_scores,
            save_path=os.path.join(mic_dir, f"{station_name}_mic_importance.png"),
            top_n=30
        )

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

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model = TCNModel(
            input_dim=len(selected_indices),
            nb_filters=64,
            kernel_size=3,
            nb_stacks=2,
            dilations=(1, 2, 4, 8, 16, 32),
            dropout_rate=0.1
        ).to(device)
        
        # 训练模型
        logger.info("\n开始模型训练...")
        model, training_history = train_model(model, train_loader, val_loader, num_epochs=100, device=device)
        
        # 评估模型
        logger.info("\n开始模型评估...")
        predictions, actual_values, metrics = evaluate_predictions(model, test_loader, device=device)
        
        # 反标准化预测结果
        predictions = target_scaler.inverse_transform(predictions)
        actual_values = target_scaler.inverse_transform(actual_values)
        
        # 创建可视化结果目录
        plots_dir = os.path.join(station_dir, "plots")
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
                    'date': all_data['date'].values[-len(predictions):],
                    'actual': actual_values[:, idx],
                    'predicted': predictions[:, idx],
                    'error': predictions[:, idx] - actual_values[:, idx]
                })
                
                # 保存预测结果到CSV
                csv_file = os.path.join(station_dir, f"{station_name}_{suffix}_predictions.csv")
                results_df.to_csv(csv_file, index=False, encoding='utf-8')
                
                # 生成可视化
                plot_file = os.path.join(plots_dir, f"{station_name}_{suffix}_plot.png")
                plot_prediction_results(
                    results_df=results_df,
                    station_name=station_name,
                    suffix=suffix,
                    save_path=plot_file
                )
                
                logger.info(f"成功生成{days}天预测的图像和CSV文件")
            except Exception as e:
                logger.error(f"处理{days}天预测结果时出错: {str(e)}")
        
        # 创建汇总报告
        summary = {
            "站点名称": station_name,
            "上游站点数量": len(upstream_stations),
            "上游站点列表": upstream_stations,
            "训练样本数": len(train_dataset),
            "验证样本数": len(val_dataset),
            "测试样本数": len(test_dataset),
            "总体评估指标": metrics["总体评估指标"],
            "各时间尺度指标": metrics["各提前期评估指标"],
            "预测时间范围": {
                "起始": pd.Timestamp(all_data['date'].values[-len(predictions)]).strftime('%Y-%m-%d'),
                "结束": pd.Timestamp(all_data['date'].values[-1]).strftime('%Y-%m-%d')
            },
            "信号处理方法": "仅使用小波降噪"
        }
        
        # 保存评估指标和汇总报告
        metrics_file = os.path.join(station_dir, f"{station_name}_metrics.json")
        summary_file = os.path.join(station_dir, f"{station_name}_summary.json")
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\n站点{station_name}处理完成!")
        logger.info(f"预测指标:")
        logger.info(f"总体RMSE: {metrics['总体评估指标']['RMSE']:.4f}")
        logger.info(f"总体R2: {metrics['总体评估指标']['R2']:.4f}")
        logger.info(f"总体NSE: {metrics['总体评估指标']['NSE']:.4f}")
        logger.info(f"结果保存在目录: {station_dir}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"处理站点{station_name}时出错: {str(e)}")
        logger.exception("详细错误信息:")
        return None

def process_station_experiment_d(station_name, upstream_stations):
    """实验D: 基线模型 (无信号处理)
    
    不使用任何信号处理方法，直接使用原始数据
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"实验D: 基线模型 - 开始处理站点: {station_name}")
    
    try:
        # 创建结果保存目录
        station_dir = f"results_D_{station_name}"
        os.makedirs(station_dir, exist_ok=True)
        
        # 读取目标站点的径流量数据（仅用作标签）
        target_runoff_data = read_runoff_data(station_name)
        if target_runoff_data is None:
            logger.error(f"目标站点{station_name}径流量数据读取失败")
            return None
            
        # 初始化一个空的DataFrame来存储所有上游站点数据
        all_data = pd.DataFrame()
        all_data['date'] = target_runoff_data['date']
        logger.info(f"开始处理上游站点数据，共{len(upstream_stations)}个上游站点")
        
        # 读取上游站点的径流量数据和气象数据，不进行任何降噪处理
        for upstream_station in upstream_stations:
            logger.info(f"\n处理上游站点: {upstream_station}")
            
            # 读取该上游站点的径流量数据
            upstream_runoff_data = read_runoff_data(upstream_station)
            
            if upstream_runoff_data is not None:
                # 直接合并径流量数据到all_data，不进行降噪
                processed_runoff_data = upstream_runoff_data[['date', 'runoff']].rename(
                    columns={'runoff': f'runoff_{upstream_station}'}
                )
                all_data = pd.merge(all_data, processed_runoff_data, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的原始径流量数据")
            else:
                logger.warning(f"上游站点{upstream_station}径流量数据不可用，将忽略该站点数据")
            
            # 读取气象数据
            upstream_meteo_data = read_meteo_data(upstream_station)
            
            # 如果有气象数据，合并到all_data
            if upstream_meteo_data is not None:
                meteo_data_to_merge = upstream_meteo_data[['date'] + METEO_FEATURES].rename(
                    columns={col: f'{col}_{upstream_station}' for col in METEO_FEATURES}
                )
                all_data = pd.merge(all_data, meteo_data_to_merge, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的气象数据")
            else:
                logger.warning(f"上游站点{upstream_station}气象数据不可用，将忽略该站点数据")
        
        # 添加时间特征和其他派生特征
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

        # 合并目标变量
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
            k=None,  
            threshold=0.1  
        )

        # 绘制特征重要性图
        plot_mic_feature_importance(
            feature_cols, mic_scores,
            save_path=os.path.join(mic_dir, f"{station_name}_mic_importance.png"),
            top_n=30
        )

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

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型
        model = TCNModel(
            input_dim=len(selected_indices),
            nb_filters=64,
            kernel_size=3,
            nb_stacks=2,
            dilations=(1, 2, 4, 8, 16, 32),
            dropout_rate=0.1
        ).to(device)
        
        # 训练模型
        logger.info("\n开始模型训练...")
        model, training_history = train_model(model, train_loader, val_loader, num_epochs=100, device=device)
        
        # 评估模型
        logger.info("\n开始模型评估...")
        predictions, actual_values, metrics = evaluate_predictions(model, test_loader, device=device)
        
        # 反标准化预测结果
        predictions = target_scaler.inverse_transform(predictions)
        actual_values = target_scaler.inverse_transform(actual_values)
        
        # 创建可视化结果目录
        plots_dir = os.path.join(station_dir, "plots")
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
                    'date': all_data['date'].values[-len(predictions):],
                    'actual': actual_values[:, idx],
                    'predicted': predictions[:, idx],
                    'error': predictions[:, idx] - actual_values[:, idx]
                })
                
                # 保存预测结果到CSV
                csv_file = os.path.join(station_dir, f"{station_name}_{suffix}_predictions.csv")
                results_df.to_csv(csv_file, index=False, encoding='utf-8')
                
                # 生成可视化
                plot_file = os.path.join(plots_dir, f"{station_name}_{suffix}_plot.png")
                plot_prediction_results(
                    results_df=results_df,
                    station_name=station_name,
                    suffix=suffix,
                    save_path=plot_file
                )
                
                logger.info(f"成功生成{days}天预测的图像和CSV文件")
            except Exception as e:
                logger.error(f"处理{days}天预测结果时出错: {str(e)}")
        
        # 创建汇总报告
        summary = {
            "站点名称": station_name,
            "上游站点数量": len(upstream_stations),
            "上游站点列表": upstream_stations,
            "训练样本数": len(train_dataset),
            "验证样本数": len(val_dataset),
            "测试样本数": len(test_dataset),
            "总体评估指标": metrics["总体评估指标"],
            "各时间尺度指标": metrics["各提前期评估指标"],
            "预测时间范围": {
                "起始": pd.Timestamp(all_data['date'].values[-len(predictions)]).strftime('%Y-%m-%d'),
                "结束": pd.Timestamp(all_data['date'].values[-1]).strftime('%Y-%m-%d')
            },
            "信号处理方法": "无信号处理"
        }
        
        # 保存评估指标和汇总报告
        metrics_file = os.path.join(station_dir, f"{station_name}_metrics.json")
        summary_file = os.path.join(station_dir, f"{station_name}_summary.json")
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        logger.info(f"\n站点{station_name}处理完成!")
        logger.info(f"预测指标:")
        logger.info(f"总体RMSE: {metrics['总体评估指标']['RMSE']:.4f}")
        logger.info(f"总体R2: {metrics['总体评估指标']['R2']:.4f}")
        logger.info(f"总体NSE: {metrics['总体评估指标']['NSE']:.4f}")
        logger.info(f"结果保存在目录: {station_dir}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"处理站点{station_name}时出错: {str(e)}")
        logger.exception("详细错误信息:")
        return None

def process_station_experiment_e(station_name, upstream_stations, nrbo_population=30, nrbo_iterations=50):
    """实验E: 仅NRBO优化+VMD分解 (移除小波降噪)
    
    使用NRBO优化VMD参数，但不应用小波降噪
    保留缓存机制以避免重复计算
    """
    global processed_stations_cache
    
    logger.info(f"\n{'='*50}")
    logger.info(f"实验E: 仅NRBO优化+VMD分解 - 开始处理站点: {station_name}")
    
    try:
        # 创建结果保存目录
        station_dir = f"results_E_{station_name}"
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
        
        # 为每个上游站点优化VMD参数
        vmd_params = {}  # 存储每个站点的最优VMD参数
        
        # 读取上游站点的径流量数据和气象数据
        for upstream_station in upstream_stations:
            logger.info(f"\n处理上游站点: {upstream_station}")
            
            # 检查是否已经处理过该上游站点的径流量数据和VMD参数
            # 这里我们使用相同的缓存，但后续处理不同
            if upstream_station in processed_stations_cache['runoff_data'] and upstream_station in processed_stations_cache['vmd_params']:
                logger.info(f"从缓存获取{upstream_station}站点的径流量数据和VMD参数")
                upstream_runoff_data = processed_stations_cache['runoff_data'][upstream_station].copy()
                vmd_params[upstream_station] = processed_stations_cache['vmd_params'][upstream_station]
            else:
                # 需要处理该上游站点的数据
                if upstream_station in processed_stations_cache['runoff_data']:
                    upstream_runoff_data = processed_stations_cache['runoff_data'][upstream_station].copy()
                else:
                    upstream_runoff_data = read_runoff_data(upstream_station)
                    if upstream_runoff_data is not None:
                        processed_stations_cache['runoff_data'][upstream_station] = upstream_runoff_data.copy()
                
                if upstream_runoff_data is not None and upstream_station not in processed_stations_cache['vmd_params']:
                    # 获取径流量数据
                    original_runoff = upstream_runoff_data['runoff'].values
                    
                    logger.info(f"开始优化{upstream_station}站点的VMD参数...")
                    
                    # 定义简单预测函数用于优化
                    def simple_prediction(signal):
                        # 一个简单的基于滞后的预测函数
                        return np.array([signal[i-7:i].mean() for i in range(7, len(signal))])
                    
                    # 准备目标信号
                    target_signal = original_runoff[7:]  # 与预测函数匹配
                    
                    # 优化VMD参数
                    best_params, best_score, convergence = optimize_vmd_parameters(
                        original_runoff, 
                        forecast_function=simple_prediction,
                        target_signal=target_signal,
                        population=nrbo_population,
                        max_iter=nrbo_iterations
                    )
                    
                    # 记录优化参数
                    K = int(best_params[0])
                    alpha = best_params[1]
                    tau = best_params[2]
                    weights = best_params[3:3+K]
                    
                    vmd_params[upstream_station] = {
                        'K': K,
                        'alpha': alpha,
                        'tau': tau,
                        'weights': weights.tolist()
                    }
                    
                    # 缓存VMD参数
                    processed_stations_cache['vmd_params'][upstream_station] = vmd_params[upstream_station]
                    
                    logger.info(f"{upstream_station}站点最优VMD参数: K={K}, alpha={alpha:.2f}, tau={tau:.4f}")
                    logger.info(f"模态权重: {weights}")
                    
                    # 保存优化后的VMD参数
                    params_file = os.path.join(station_dir, f"{upstream_station}_vmd_params.json")
                    with open(params_file, 'w', encoding='utf-8') as f:
                        json.dump(vmd_params[upstream_station], f, indent=2, ensure_ascii=False)
                    
                    # 保存收敛曲线
                    plt.figure(figsize=(10, 6))
                    plt.plot(convergence, 'b-')
                    plt.title(f'{upstream_station}站点VMD参数优化收敛曲线')
                    plt.xlabel('迭代次数')
                    plt.ylabel('目标函数值')
                    plt.grid(True)
                    plt.savefig(os.path.join(station_dir, f"{upstream_station}_convergence.png"))
                    plt.close()
            
            # 不论是否从缓存获取，都需要进行VMD处理但不进行小波降噪
            # 这是本实验与实验A的主要区别
            if upstream_runoff_data is not None:
                # 获取径流量数据
                original_runoff = upstream_runoff_data['runoff'].values
                
                if upstream_station in vmd_params:
                    # 使用优化后的参数进行VMD分解，但不进行小波降噪
                    try:
                        K = vmd_params[upstream_station]['K']
                        alpha = vmd_params[upstream_station]['alpha']
                        tau = vmd_params[upstream_station]['tau']
                        
                        # 对信号进行VMD分解，但不应用小波降噪
                        # 使用信号标准化
                        signal_mean = np.mean(original_runoff)
                        signal_std = np.std(original_runoff)
                        if signal_std > 0:
                            normalized_signal = (original_runoff - signal_mean) / signal_std
                        else:
                            normalized_signal = original_runoff - signal_mean
                        
                        # VMD分解
                        u, _, _ = VMD(normalized_signal, alpha, tau, K, DC=True, init=0, tol=1e-7)
                        
                        # 直接重建信号，不进行小波降噪
                        # 如果有权重信息，使用权重
                        if 'weights' in vmd_params[upstream_station]:
                            weights = np.array(vmd_params[upstream_station]['weights'])
                            vmd_signal = np.zeros_like(normalized_signal)
                            for i in range(K):
                                vmd_signal += u[i] * weights[i]
                        else:
                            # 如果没有权重，直接求和
                            vmd_signal = np.sum(u, axis=0)
                        
                        # 反标准化
                        vmd_signal = vmd_signal * signal_std + signal_mean
                        
                        # 更新径流量数据
                        upstream_runoff_data['runoff'] = vmd_signal
                        
                        logger.info(f"成功使用VMD分解处理{upstream_station}站点的径流量数据（无小波降噪）")
                    except Exception as e:
                        logger.warning(f"使用VMD处理{upstream_station}站点数据失败: {str(e)}")
                
                # 合并径流量数据到all_data
                processed_runoff_data = upstream_runoff_data[['date', 'runoff']].rename(
                    columns={'runoff': f'runoff_{upstream_station}'}
                )
                all_data = pd.merge(all_data, processed_runoff_data, on='date', how='left')
                logger.info(f"成功合并{upstream_station}站点的径流量数据")
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
        
        # 保存所有站点的VMD参数
        all_params_file = os.path.join(station_dir, "all_vmd_params.json")
        with open(all_params_file, 'w', encoding='utf-8') as f:
            json.dump(vmd_params, f, indent=2, ensure_ascii=False)
        
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
        
        # 创建模型
        model = TCNModel(
            input_dim=len(selected_indices),
            nb_filters=64,
            kernel_size=3,
            nb_stacks=2,
            dilations=(1, 2, 4, 8, 16, 32),
            dropout_rate=0.1
        ).to(device)
        
        # 训练模型
        logger.info("\n开始模型训练...")
        model, training_history = train_model(model, train_loader, val_loader, num_epochs=100, device=device)
        
        # 评估模型
        logger.info("\n开始模型评估...")
        predictions, actual_values, metrics = evaluate_predictions(model, test_loader, device=device)
        
        # 反标准化预测结果
        predictions = target_scaler.inverse_transform(predictions)
        actual_values = target_scaler.inverse_transform(actual_values)
        
        # 创建可视化结果目录
        plots_dir = os.path.join(station_dir, "plots")
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
                    'date': all_data['date'].values[-len(predictions):],
                    'actual': actual_values[:, idx],
                    'predicted': predictions[:, idx],
                    'error': predictions[:, idx] - actual_values[:, idx]
                })
                
                # 保存预测结果到CSV
                csv_file = os.path.join(station_dir, f"{station_name}_{suffix}_predictions.csv")
                results_df.to_csv(csv_file, index=False, encoding='utf-8')
                
                # 生成可视化
                plot_file = os.path.join(plots_dir, f"{station_name}_{suffix}_plot.png")
                plot_prediction_results(
                    results_df=results_df,
                    station_name=station_name,
                    suffix=suffix,
                    save_path=plot_file
                )
                
                logger.info(f"成功生成{days}天预测的图像和CSV文件")
            except Exception as e:
                logger.error(f"处理{days}天预测结果时出错: {str(e)}")
            
            # 输出每个时间尺度的预测指标
            logger.info(f"\n{days}天预测结果:")
            logger.info(f"RMSE: {metrics['各提前期评估指标'][f'{days}天预测']['RMSE']:.4f}")
            logger.info(f"R2: {metrics['各提前期评估指标'][f'{days}天预测']['R2']:.4f}")
            logger.info(f"sMAPE: {metrics['各提前期评估指标'][f'{days}天预测']['sMAPE']:.2f}%")
            logger.info(f"NSE: {metrics['各提前期评估指标'][f'{days}天预测']['NSE']:.4f}")
        
        # 创建汇总报告，包含VMD优化信息
        summary = {
            "站点名称": station_name,
            "上游站点数量": len(upstream_stations),
            "上游站点列表": upstream_stations,
            "训练样本数": len(train_dataset),
            "验证样本数": len(val_dataset),
            "测试样本数": len(test_dataset),
            "总体评估指标": metrics["总体评估指标"],
            "各时间尺度指标": metrics["各提前期评估指标"],
            "预测时间范围": {
                "起始": pd.Timestamp(all_data['date'].values[-len(predictions)]).strftime('%Y-%m-%d'),
                "结束": pd.Timestamp(all_data['date'].values[-1]).strftime('%Y-%m-%d')
            },
            "信号处理方法": "NRBO优化+VMD分解，无小波降噪",
            "VMD参数优化信息": vmd_params
        }
        
        # 保存评估指标和汇总报告
        metrics_file = os.path.join(station_dir, f"{station_name}_metrics.json")
        summary_file = os.path.join(station_dir, f"{station_name}_summary.json")
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        # 比较优化前后的结果
        logger.info(f"\n站点{station_name}处理完成!")
        logger.info(f"预测指标:")
        logger.info(f"总体RMSE: {metrics['总体评估指标']['RMSE']:.4f}")
        logger.info(f"总体R2: {metrics['总体评估指标']['R2']:.4f}")
        logger.info(f"总体sMAPE: {metrics['总体评估指标']['sMAPE']:.2f}%")
        logger.info(f"总体NSE: {metrics['总体评估指标']['NSE']:.4f}")
        logger.info(f"结果保存在目录: {station_dir}")
        
        return metrics, vmd_params
        
    except Exception as e:
        logger.error(f"处理站点{station_name}时出错: {str(e)}")
        logger.exception("详细错误信息:")
        return None
    
def run_ablation_study(target_stations=None, nrbo_population=30, nrbo_iterations=50):
    """运行消融实验
    
    参数:
        target_stations: 要处理的目标站点列表，如果为None则处理所有站点
        nrbo_population: NRBO算法种群大小
        nrbo_iterations: NRBO算法迭代次数
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
    
    # 根据经度排序站点（从上游到下游）
    ordered_stations = sorted(
        station_locations.items(), 
        key=lambda x: x[1]["lon"]
    )
    ordered_stations = [station[0] for station in ordered_stations]
    
    # 建立上游站点字典
    upstream_dict = {}
    for i, station in enumerate(ordered_stations):
        upstream_dict[station] = ordered_stations[:i]
    
    # 如果没有指定目标站点，使用除最上游站点外的所有站点
    if target_stations is None:
        target_stations = ordered_stations[1:]  # 跳过最上游站点
    
    # 创建结果目录
    os.makedirs("ablation_results", exist_ok=True)
    
    # 初始化实验结果字典
    experiment_results = {
        "实验A": {},  # 完整模型: NRBO优化+VMD分解+小波降噪
        "实验B": {},  # 移除NRBO优化: 固定VMD参数+VMD分解+小波降噪
        "实验C": {},  # 移除VMD分解: 仅使用小波降噪
        "实验D": {},  # 基线模型: 无信号处理
        "实验E": {}   # 新增: 仅NRBO优化+VMD分解 (移除小波降噪)
    }
    
    # 运行每个站点的每个实验
    for station in target_stations:
        logger.info(f"\n\n{'#'*80}")
        logger.info(f"开始处理站点 {station} 的所有消融实验")
        logger.info(f"{'#'*80}\n")
        
        # 获取上游站点
        upstream_stations = upstream_dict[station]
        if not upstream_stations:
            logger.warning(f"站点 {station} 没有上游站点，跳过")
            continue
            
        logger.info(f"站点 {station} 的上游站点: {', '.join(upstream_stations)}")
        
        # 实验A: 完整模型 (NRBO优化+VMD分解+小波降噪)
        logger.info(f"\n\n开始实验A: 完整模型 (NRBO优化+VMD分解+小波降噪)")
        results_a, _ = process_station_experiment_a(station, upstream_stations, nrbo_population, nrbo_iterations)
        if results_a is not None:
            experiment_results["实验A"][station] = {
                "RMSE": results_a["总体评估指标"]["RMSE"],
                "R2": results_a["总体评估指标"]["R2"],
                "NSE": results_a["总体评估指标"]["NSE"],
            }
        
        # 实验B: 固定VMD参数 (移除NRBO优化)
        logger.info(f"\n\n开始实验B: 固定VMD参数 (移除NRBO优化)")
        results_b = process_station_experiment_b(station, upstream_stations)
        if results_b is not None:
            experiment_results["实验B"][station] = {
                "RMSE": results_b["总体评估指标"]["RMSE"],
                "R2": results_b["总体评估指标"]["R2"],
                "NSE": results_b["总体评估指标"]["NSE"],
            }
        
        # 实验C: 仅使用小波降噪 (移除VMD分解)
        logger.info(f"\n\n开始实验C: 仅使用小波降噪 (移除VMD分解)")
        results_c = process_station_experiment_c(station, upstream_stations)
        if results_c is not None:
            experiment_results["实验C"][station] = {
                "RMSE": results_c["总体评估指标"]["RMSE"],
                "R2": results_c["总体评估指标"]["R2"],
                "NSE": results_c["总体评估指标"]["NSE"],
            }
        
        # 实验D: 基线模型 (无信号处理)
        logger.info(f"\n\n开始实验D: 基线模型 (无信号处理)")
        results_d = process_station_experiment_d(station, upstream_stations)
        if results_d is not None:
            experiment_results["实验D"][station] = {
                "RMSE": results_d["总体评估指标"]["RMSE"],
                "R2": results_d["总体评估指标"]["R2"],
                "NSE": results_d["总体评估指标"]["NSE"],
            }
            
        # 实验E: 仅NRBO优化+VMD分解 (移除小波降噪)
        logger.info(f"\n\n开始实验E: 仅NRBO优化+VMD分解 (移除小波降噪)")
        results_e, _ = process_station_experiment_e(station, upstream_stations, nrbo_population, nrbo_iterations)
        if results_e is not None:
            experiment_results["实验E"][station] = {
                "RMSE": results_e["总体评估指标"]["RMSE"],
                "R2": results_e["总体评估指标"]["R2"],
                "NSE": results_e["总体评估指标"]["NSE"],
            }
    
    # 保存所有实验结果
    with open("ablation_results/all_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2, ensure_ascii=False)
    
    # 生成结果汇总表
    results_summary = {}
    for experiment, stations_data in experiment_results.items():
        experiment_avg = {
            "RMSE": np.mean([data["RMSE"] for data in stations_data.values()]),
            "R2": np.mean([data["R2"] for data in stations_data.values()]),
            "NSE": np.mean([data["NSE"] for data in stations_data.values()]),
        }
        results_summary[experiment] = experiment_avg
    
    # 保存结果汇总
    with open("ablation_results/summary_results.json", "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2, ensure_ascii=False)
    
    # 输出结果对比
    logger.info("\n\n" + "="*80)
    logger.info("消融实验结果汇总")
    logger.info("="*80)
    
    for experiment, metrics in results_summary.items():
        logger.info(f"\n{experiment}:")
        logger.info(f"  平均RMSE: {metrics['RMSE']:.4f}")
        logger.info(f"  平均R2: {metrics['R2']:.4f}")
        logger.info(f"  平均NSE: {metrics['NSE']:.4f}")
    
    # 可视化结果
    plot_ablation_results(experiment_results, results_summary)
    
    logger.info("\n消融实验完成!")
    return experiment_results, results_summary

def plot_ablation_results(experiment_results, results_summary):
    """可视化消融实验结果"""
    try:
        logger.info("开始生成消融实验结果可视化...")
        
        # 创建保存目录
        plots_dir = "ablation_results/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 绘制平均指标对比图
        metrics = ["RMSE", "R2", "NSE"]
        experiments = list(results_summary.keys())
        
        for metric in metrics:
            plt.figure(figsize=(12, 7))
            values = [results_summary[exp][metric] for exp in experiments]
            
            # 设置颜色映射 - 更新以包含实验E的颜色
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            
            bars = plt.bar(experiments, values, color=colors)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}',
                        ha='center', va='bottom', rotation=0)
            
            plt.title(f'不同实验的平均{metric}指标对比')
            plt.ylabel(metric)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 根据指标调整y轴显示范围
            if metric == "RMSE":
                plt.gca().invert_yaxis()  # RMSE越低越好
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"avg_{metric}_comparison.png"), dpi=300)
            plt.close()
        
        # 绘制各站点指标对比图
        for metric in metrics:
            plt.figure(figsize=(14, 8))
            
            # 获取所有站点
            all_stations = set()
            for exp_data in experiment_results.values():
                all_stations.update(exp_data.keys())
            all_stations = sorted(list(all_stations))
            
            # 为每个实验准备数据
            x = np.arange(len(all_stations))
            width = 0.15  # 更新柱状图宽度以适应5个实验
            offsets = [-0.3, -0.15, 0, 0.15, 0.3]  # 调整偏移量
            
            for i, (exp, offset) in enumerate(zip(experiments, offsets)):
                values = []
                for station in all_stations:
                    if station in experiment_results[exp]:
                        values.append(experiment_results[exp][station][metric])
                    else:
                        values.append(np.nan)
                
                plt.bar(x + offset, values, width, label=exp, color=colors[i], alpha=0.8)
            
            plt.xlabel('站点')
            plt.ylabel(metric)
            plt.title(f'各站点{metric}指标对比')
            plt.xticks(x, all_stations, rotation=45)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 根据指标调整y轴显示范围
            if metric == "RMSE":
                plt.gca().invert_yaxis()  # RMSE越低越好
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"station_{metric}_comparison.png"), dpi=300)
            plt.close()
        
        # 绘制热力图
        for metric in metrics:
            plt.figure(figsize=(10, 8))
            
            # 创建数据矩阵
            data_matrix = []
            for station in all_stations:
                station_data = []
                for exp in experiments:
                    if station in experiment_results[exp]:
                        station_data.append(experiment_results[exp][station][metric])
                    else:
                        station_data.append(np.nan)
                data_matrix.append(station_data)
            
            data_array = np.array(data_matrix)
            
            # 创建热力图
            cmap = 'RdYlGn_r' if metric == 'RMSE' else 'RdYlGn'  # RMSE越低越好，其他越高越好
            ax = sns.heatmap(data_array, annot=True, fmt='.4f', 
                            xticklabels=experiments, yticklabels=all_stations,
                            cmap=cmap)
            
            plt.title(f'{metric}指标热力图')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{metric}_heatmap.png"), dpi=300)
            plt.close()
        
        # 新增：生成实验组成部分的对比图
        components = {
            "实验A": ["NRBO优化", "VMD分解", "小波降噪"],
            "实验B": ["固定参数", "VMD分解", "小波降噪"],
            "实验C": ["无VMD", "小波降噪", ""],
            "实验D": ["无处理", "", ""],
            "实验E": ["NRBO优化", "VMD分解", "无小波"]
        }
        
        plt.figure(figsize=(12, 8))
        
        # 使用表格展示每个实验的组成部分
        cell_text = []
        for exp in experiments:
            cell_text.append(components[exp])
        
        the_table = plt.table(cellText=cell_text,
                            rowLabels=experiments,
                            colLabels=["参数优化", "信号分解", "降噪方法"],
                            loc='center',
                            cellLoc='center')
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)
        the_table.scale(1, 2)
        
        plt.axis('off')
        plt.title('消融实验组成部分对比', fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "experiment_components.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("消融实验结果可视化完成!")
        
    except Exception as e:
        logger.error(f"生成可视化时出错: {str(e)}")
        logger.exception("详细错误信息:")

if __name__ == "__main__":
    import torch.nn as nn
    from torch.utils.data import DataLoader
    
    try:
        # 设置随机种子保证可重复性
        np.random.seed(42)
        torch.manual_seed(42)
        
        # 设置日志文件名包含时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"ablation_experiment_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        logger.info("开始消融实验...")
        logger.info("实验A: 完整模型 (NRBO优化+VMD分解+小波降噪)")
        logger.info("实验B: 固定VMD参数 (移除NRBO优化)")
        logger.info("实验C: 仅使用小波降噪 (移除VMD分解)")
        logger.info("实验D: 基线模型 (无信号处理)")
        logger.info("实验E: 仅NRBO优化+VMD分解 (移除小波降噪)")
        
        # 可以指定特定站点进行测试
        # target_stations = ["安康", "襄阳", "宜城"]
        target_stations = None  # None表示处理除最上游外的所有站点
        
        # 运行消融实验
        experiment_results, results_summary = run_ablation_study(
            target_stations=target_stations,
            nrbo_population=30,
            nrbo_iterations=50
        )
        
        logger.info("消融实验程序正常结束")
        
    except Exception as e:
        logger.error("程序执行出错")
        logger.exception("详细错误信息:")