# vmd_optimizer.py - VMD参数优化
import numpy as np
import matplotlib.pyplot as plt
from vmdpy import VMD
import pywt
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from NRBO import nrbo
import logging
from datetime import datetime
import os
import json

def weighted_vmd_wavelet_denoise(signal, params, forecast_function=None, target_signal=None):
    """
    使用权重的VMD-小波混合降噪方法，适用于NRBO优化
    
    参数:
        signal: 原始信号
        params: 包含VMD参数和模态权重的数组
        forecast_function: 可选的预测函数，用于评估降噪效果
        target_signal: 可选的目标信号，用于评估预测准确性
        
    返回:
        denoised_signal: 降噪后的信号
        metrics: 如果提供了forecast_function和target_signal，则返回评估指标
    """
    # 解析参数
    K = int(params[0])  # 模态数量
    alpha = params[1]   # 带宽约束
    tau = params[2]     # 噪声容忍度
    mode_weights = params[3:3+K]  # 模态权重
    
    # 处理NaN值
    if np.any(np.isnan(signal)):
        nan_indices = np.isnan(signal)
        not_nan_indices = ~nan_indices
        signal = np.interp(
            x=np.arange(len(signal)),
            xp=np.arange(len(signal))[not_nan_indices],
            fp=signal[not_nan_indices]
        )

    # 信号标准化
    signal_mean = np.mean(signal)
    signal_std = np.std(signal)
    if signal_std > 0:
        normalized_signal = (signal - signal_mean) / signal_std
    else:
        normalized_signal = signal - signal_mean

    # VMD分解参数
    DC = True      # 包含直流项
    init = 0       # 初始化方法
    tol = 1e-7     # 收敛容忍度

    try:
        # 执行VMD分解
        u, u_hat, omega = VMD(normalized_signal, alpha, tau, K, DC, init, tol)
        
        # 对每个模态应用小波降噪
        denoised_modes = []
        
        for i, mode_signal in enumerate(u):
            # 根据模态频率选择小波参数
            frequency_ratio = omega[-1, i] / np.max(omega[-1, :])
            
            # 选择小波类型
            if frequency_ratio < 0.3:
                mode_wavelet = 'sym8'  # 低频
                threshold_multiplier = 0.6
            elif frequency_ratio < 0.7:
                mode_wavelet = 'db4'   # 中频
                threshold_multiplier = 1.0
            else:
                mode_wavelet = 'db1'   # 高频
                threshold_multiplier = 1.5
            
            # 小波分解
            coeffs = pywt.wavedec(mode_signal, mode_wavelet)
            
            # 计算自适应阈值
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(mode_signal))) * threshold_multiplier
            
            # 阈值处理
            denoised_coeffs = [coeffs[0]]  # 保留近似系数
            for j in range(1, len(coeffs)):
                denoised_coeffs.append(pywt.threshold(coeffs[j], threshold, mode='soft'))
            
            # 小波重构
            denoised_mode = pywt.waverec(denoised_coeffs, mode_wavelet)
            
            # 确保长度匹配
            if len(denoised_mode) != len(mode_signal):
                denoised_mode = denoised_mode[:len(mode_signal)]
                
            denoised_modes.append(denoised_mode)
        
        # 使用权重重建信号
        denoised_signal = np.zeros_like(normalized_signal)
        for i in range(K):
            denoised_signal += denoised_modes[i] * mode_weights[i]
        
        # 反标准化
        denoised_signal = denoised_signal * signal_std + signal_mean
        
        # 确保长度匹配
        if len(denoised_signal) != len(signal):
            denoised_signal = denoised_signal[:len(signal)]
        
        # 如果提供了预测函数和目标信号，评估预测性能
        metrics = {}
        if forecast_function is not None and target_signal is not None:
            predictions = forecast_function(denoised_signal)
            rmse = np.sqrt(mean_squared_error(target_signal, predictions))
            r2 = r2_score(target_signal, predictions)
            metrics = {
                'rmse': rmse,
                'r2': r2
            }
            
        return denoised_signal, metrics
        
    except Exception as e:
        print(f"VMD分解失败: {str(e)}")
        # 失败时返回原始信号和空指标
        return signal, {}

def vmd_objective_function(params, signal, forecast_function=None, target_signal=None):
    """
    VMD参数优化的目标函数
    
    参数:
        params: 包含VMD参数和模态权重的数组
        signal: 输入信号
        forecast_function: 可选的预测函数
        target_signal: 可选的目标信号
        
    返回:
        score: 优化目标分数（越低越好）
    """
    # 应用带权重的VMD降噪
    denoised_signal, metrics = weighted_vmd_wavelet_denoise(
        signal, params, forecast_function, target_signal
    )
    
    if forecast_function is not None and target_signal is not None and metrics:
        # 如果有预测功能，使用预测性能作为主要目标
        rmse = metrics['rmse']
        r2 = metrics['r2']
        
        # 计算信号保真度 (SNR的简化版本)
        signal_power = np.mean(np.square(signal))
        noise_power = np.mean(np.square(signal - denoised_signal))
        if noise_power == 0:  # 避免除零
            snr = 100  # 大值表示高保真度
        else:
            snr = 10 * np.log10(signal_power / noise_power)
        
        # 组合目标: 较低的RMSE, 较高的R2, 合理的SNR
        # 这里我们给预测准确性更高的权重
        score = 0.6 * rmse - 0.3 * r2 + 0.1 * max(0, 15 - snr)
    else:
        # 如果没有预测功能，仅使用信号平滑度和保真度
        # 计算信号平滑度
        smoothness = np.mean(np.square(np.diff(denoised_signal)))
        
        # 计算与原始信号的相似度
        similarity = np.corrcoef(signal, denoised_signal)[0, 1]
        
        # 组合目标: 平滑但保持相似性
        score = smoothness - similarity
    
    return score

def optimize_vmd_parameters(signal, forecast_function=None, target_signal=None, 
                           population=30, max_iter=50):
    """
    使用NRBO优化VMD参数和模态权重
    
    参数:
        signal: 输入信号
        forecast_function: 可选的预测函数
        target_signal: 可选的目标信号
        population: NRBO种群大小
        max_iter: NRBO最大迭代次数
        
    返回:
        best_params: 最佳参数组合
        best_score: 最佳分数
        convergence: 收敛曲线
    """
    # 设置参数范围
    max_K = 8  # 最大模态数
    
    # 定义参数边界
    # [K, alpha, tau, weight_1, weight_2, ..., weight_max_K]
    lb = np.array([2, 500, 0] + [0.1] * max_K)
    ub = np.array([max_K, 5000, 0.5] + [2.0] * max_K)
    
    # 参数维度
    dim = 3 + max_K  # 3个VMD参数 + K个权重
    
    # 定义目标函数(闭包)
    def objective(params):
        # 确保K是整数且在合理范围内
        K = max(2, min(max_K, int(params[0])))
        
        # 复制参数并更正K
        p = params.copy()
        p[0] = K
        
        # 调用VMD目标函数
        return vmd_objective_function(p, signal, forecast_function, target_signal)
    
    # 运行NRBO优化
    best_params, best_score, convergence = nrbo(
        objective, dim, lb, ub, population, max_iter
    )
    
    # 确保K是整数
    best_params[0] = int(best_params[0])
    
    return best_params, best_score, convergence
