import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.preprocessing import MinMaxScaler

from models.src.predict_model import LatentOHLCVRNN, LatentOHLCVLSTM
from models.src.VAE_trainer import TransformerVAE_TDist
from models.src.dataloader_setup import LatentOHLCVPredictor


@st.cache_resource
def load_vae_model(vae_path, feature_dim, latent_dim, embed_dim, df, device):
    model = TransformerVAE_TDist(feature_dim=feature_dim, latent_dim=latent_dim, embed_dim=embed_dim, df=df).to(device)
    model.load_state_dict(torch.load(vae_path, map_location=device))
    model.eval()
    return model


# --- 【关键修改 A】: load_predictor_model 现在不再依赖于动态的 seq_length ---
# 我们把它从@st.cache_resource中移除，因为它现在依赖于固定的超参数，或者确保它的参数在一次运行中是恒定的
# 为了避免缓存冲突，最简单的方法是确保传入的seq_length是恒定的
@st.cache_resource
def load_predictor_model(model_name, predictor_path, device):
    # 使用固定的seq_length来初始化模型
    new_model = None
    if model_name == "rnn":
        new_model = LatentOHLCVRNN().to(device)
    elif model_name == "lstm":
        new_model = LatentOHLCVLSTM().to(device)
    else:
        new_model = LatentOHLCVPredictor().to(device)

    model = new_model
    model.load_state_dict(torch.load(predictor_path, map_location=device))
    model.eval()
    return model


# 添加数据处理工具函数
def compute_logdiff(data):
    """计算log差分"""
    return np.diff(np.log(data + 1e-8), axis=0)

# 【关键修改】新增数据处理函数，使用与训练时一致的方式
def process_inference_data(data_file, target_columns, feature_columns_list,y_scaler=None, x_scalers=None, start_row=0, end_row=None, log_diff=False):
    """
    使用与训练时相同的数据处理方式
    """
    # Load data
    data = pd.read_csv(data_file, parse_dates=['date'] if 'date' in pd.read_csv(data_file, nrows=1).columns else None)

    if end_row is None:
        end_row = len(data)

    # Select target columns
    y = data.iloc[start_row:end_row, target_columns].values
    target_column_names = data.columns[target_columns]
    print("Target columns:", target_column_names)

    # Process each set of feature columns
    x_list = []
    feature_column_names_list = []
    # x_scalers = []

    for feature_columns in feature_columns_list:
        # Select feature columns
        x = data.iloc[start_row:end_row, feature_columns].values
        feature_column_names = data.columns[feature_columns]
        print("Feature columns:", feature_column_names)
        print(f"Feature shape: {x.shape}")
        x_list.append(x)
        feature_column_names_list.append(feature_column_names)

    # Apply log differencing if needed
    if log_diff:
        x_list = [compute_logdiff(x) for x in x_list]
        y = compute_logdiff(y)

    # # Normalize each x set separately (using fit_transform for inference)
    # normalized_x_list = []
    # for x in x_list:
    #     x_scaler = MinMaxScaler(feature_range=(0, 1))
    #     normalized_x = x_scaler.fit_transform(x)
    #     normalized_x_list.append(normalized_x)
    #     x_scalers.append(x_scaler)
    #
    # # Normalize y
    # y_scaler = MinMaxScaler(feature_range=(0, 1))
    # normalized_y = y_scaler.fit_transform(y)
    # 【核心逻辑】处理 y 归一化
    if y_scaler is None:
        print("警告：未传入 y_scaler，将基于当前数据拟合新的归一化器。")
        y_scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_y = y_scaler.fit_transform(y)
    else:
        normalized_y = y_scaler.transform(y)

    # 【核心逻辑】处理 x 归一化
    normalized_x_list = []
    if x_scalers is None or len(x_scalers) != len(x_list):
        if x_scalers is not None:
            print(f"警告：x_scalers 数量({len(x_scalers)}个)与特征集数量({len(x_list)}个)不匹配，将创建新的归一化器。")
        else:
            print("警告：未传入 x_scalers，将基于当前数据拟合新的归一化器。")
        x_scalers = []
        for x in x_list:
            x_scaler = MinMaxScaler(feature_range=(0, 1))
            normalized_x = x_scaler.fit_transform(x)
            normalized_x_list.append(normalized_x)
            x_scalers.append(x_scaler)
    else:
        print("归一化器开始执行")
        for i, x in enumerate(x_list):
            normalized_x = x_scalers[i].transform(x)
            normalized_x_list.append(normalized_x)

    return {
        'x_list': normalized_x_list,
        'y': normalized_y,
        'x_scalers': x_scalers,
        'y_scaler': y_scaler,
        'raw_data': data.iloc[start_row:end_row],
        'feature_columns_list': feature_columns_list,
        'target_columns': target_columns
    }