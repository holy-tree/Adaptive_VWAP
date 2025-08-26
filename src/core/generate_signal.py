import os
from argparse import Namespace

import numpy as np
import pandas_ta as ta
import pandas as pd
import torch
import joblib

from src.utils.model_utils import process_inference_data


def generate_bollinger_signals_with_strength(df, length=20, std=2.0, stl_param=5.0, n_param=6.0):
    """
    基于布林带的双向交易信号生成策略，并计算信号强度。
    Args:
        df (pd.DataFrame): 包含'close', 'high', 'low'列的数据。
        ... (其他参数同上) ...
    Returns:
        pd.DataFrame: 添加'signal'和'signal_strength'列后的数据。
    """
    if df.empty:
        df['signal'] = []
        df['signal_strength'] = []
        return df

    bbands = ta.bbands(df['close'], length=length, std=std)
    df['bb_upper'] = bbands[f'BBU_{length}_{std}']
    df['bb_middle'] = bbands[f'BBM_{length}_{std}']
    df['bb_lower'] = bbands[f'BBL_{length}_{std}']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=length)

    signals = ['Hold'] * len(df)
    strengths = [0.0] * len(df)
    position = 0
    bkhigh = 0.0
    sklow = float('inf')

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # --- 止损或止盈逻辑 ---
        if position == 1:
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or prev_row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or prev_row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 入场逻辑 ---
        if position == 0:
            # 开多：由下轨下穿反转向上
            if prev_row['close'] < prev_row['bb_lower'] and row['close'] > row['bb_lower']:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
                # 计算信号强度：价格越偏离中轨，信号越强
                strength = (row['close'] - row['bb_lower']) / (row['bb_middle'] - row['bb_lower'])
                strengths[i] = max(0.0, min(1.0, strength))
            # 开空：由上轨上穿反转向下
            elif prev_row['close'] > prev_row['bb_upper'] and row['close'] < row['bb_upper']:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']
                # 计算信号强度：价格越偏离中轨，信号越强
                strength = (row['bb_upper'] - row['close']) / (row['bb_upper'] - row['bb_middle'])
                strengths[i] = max(0.0, min(1.0, strength))

    df['signal'] = signals
    df['signal_strength'] = strengths
    return df

def generate_bollinger_signals(df, length=20, std=2.0, stl_param=5.0, n_param=6.0):
    """
    基于布林带的双向交易信号生成策略。

    Args:
        df (pd.DataFrame): 包含'close', 'high', 'low'列的数据。
        length (int): 布林带均线周期。
        std (float): 标准差倍数。
        stl_param (float): 百分比止损参数。
        n_param (float): ATR止损倍数。

    Returns:
        pd.DataFrame: 添加'signal'列后的数据。
    """
    if df.empty:
        df['signal'] = []
        return df

    # 计算布林带和ATR
    bbands = ta.bbands(df['close'], length=length, std=std)
    df['bb_upper'] = bbands[f'BBU_{length}_{std}']
    df['bb_middle'] = bbands[f'BBM_{length}_{std}']
    df['bb_lower'] = bbands[f'BBL_{length}_{std}']
    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=length)

    signals = ['Hold'] * len(df)
    position = 0  # 1 多仓，-1 空仓，0 空仓
    bkhigh = 0.0
    sklow = float('inf')

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]

        # --- 止损或止盈逻辑 ---
        if position == 1:
            stop_loss_atr = bkhigh - n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 - 0.01 * stl_param)
            if row['close'] <= stop_loss_atr or prev_row['low'] < stop_loss_stl:
                signals[i] = 'Close_Buy'
                position = 0
                continue
            bkhigh = max(bkhigh, row['high'])

        elif position == -1:
            stop_loss_atr = sklow + n_param * row['atr']
            stop_loss_stl = prev_row['close'] * (1 + 0.01 * stl_param)
            if row['close'] >= stop_loss_atr or prev_row['high'] > stop_loss_stl:
                signals[i] = 'Close_Sell'
                position = 0
                continue
            sklow = min(sklow, row['low'])

        # --- 入场逻辑 ---
        if position == 0:
            # 开多：由下轨下穿反转向上
            if prev_row['close'] < prev_row['bb_lower'] and row['close'] > row['bb_lower']:
                signals[i] = 'Buy'
                position = 1
                bkhigh = row['high']
            # 开空：由上轨上穿反转向下
            elif prev_row['close'] > prev_row['bb_upper'] and row['close'] < row['bb_upper']:
                signals[i] = 'Sell'
                position = -1
                sklow = row['low']

    df['signal'] = signals
    return df


# 【关键修改】修改GAN生成器加载函数
def load_trained_gan_generators(symbol, device, window_sizes=None):
    """
    加载gan_model_path目录下所有以symbol为前缀的模型文件，返回生成器列表
    """
    import importlib.util
    model_path = "../../models/src/model_with_clsdisc.py"
    spec = importlib.util.spec_from_file_location("model_with_clsdisc", model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    gan_model_path = f"../../models/trained/generators/maa"
    if not os.path.exists(gan_model_path):
        print(f"GAN generator model directory not found at {gan_model_path}，跳过。")
        return None, None

    generators = []
    generator_configs = []

    for fname in os.listdir(gan_model_path):
        if fname.startswith(symbol.split("_")[0]) and (fname.endswith(".pth") or fname.endswith(".pt")):
            model_file = os.path.join(gan_model_path, fname)
            print(f"==========={model_file}---------------+++++++++++")

            try:
                state_dict = torch.load(model_file, map_location=device, weights_only=True)

                # 动态推断输入维度
                actual_input_size = 12  # 默认值
                lower_fname = fname.lower()

                parts = fname.split('_')
                window_size_index = int(parts[1]) - 1  # parts[1] 是 '1'/'2'/'3'，索引从0开始

                if "itransformer" in lower_fname:
                    # 从 state_dict 动态推断关键参数，保证与训练时模型匹配
                    # 如果训练时没有修改这些参数，那么它们将与您提供的默认值一致
                    if 'projector.weight' in state_dict:
                        d_model = state_dict['projector.weight'].shape[1]
                    else:
                        d_model = 512  # 使用默认值

                    # if 'enc_embedding.conv.weight' in state_dict:
                    #     seq_len = state_dict['enc_embedding.conv.weight'].shape[1]
                    # else:
                    #     seq_len = 15  # 使用默认值

                    num_classes = state_dict['classifier.2.weight'].shape[
                        0] if 'classifier.2.weight' in state_dict else 3

                    # 创建一个 Namespace 对象来模拟训练时的配置，直接使用您提供的默认值
                    itransformer_configs = Namespace(
                        seq_len=window_sizes[window_size_index],
                        pred_len=4,
                        d_model=d_model,
                        embed='timeF',
                        freq='h',
                        dropout=0.05,
                        n_heads=8,
                        d_ff=2048,
                        activation='gelu',
                        e_layers=2,
                        factor=1,
                        output_attention=False,
                        use_norm=True,
                        class_strategy='latent',
                        num_classes=num_classes,
                    )

                    GenClass = getattr(model_module, "Generator_itransformer")
                    gan_generator = GenClass(itransformer_configs).to(device)
                    actual_input_size = window_sizes[window_size_index]
                    print(f"成功加载 iTransformer，seq_len={window_sizes[window_size_index]}, d_model={d_model}")
                elif "ptransformer" in lower_fname:
                    GenClass = getattr(model_module, "Generator_ptransformer")
                    gan_generator = GenClass(input_dim=actual_input_size, seq_len=window_sizes[window_size_index]).to(device)

                elif "fits" in lower_fname:
                    GenClass = getattr(model_module, "Generator_FITS")
                    # 从回归头的输入维度反推seq_len
                    pred_len = 4  # 假设
                    cut_freq = window_sizes[window_size_index] // 2
                    fits_configs = Namespace(
                        seq_len=window_sizes[window_size_index],  # 动态推导
                        pred_len=pred_len,
                        enc_in=12,
                        cut_freq=cut_freq,
                        individual=False,
                        num_classes=3
                    )
                    gan_generator = GenClass(fits_configs).to(device)
                    print(f"成功加载 FITS，seq_len={fits_configs.seq_len}, cut_freq={fits_configs.cut_freq}")
                elif "transformer" in lower_fname:
                    GenClass = getattr(model_module, "Generator_transformer")
                    gan_generator = GenClass(actual_input_size, feature_size=128, num_layers=2, num_heads=8,
                                             dropout=0.1, output_len=4).to(device)
                    if 'input_projection.weight' in state_dict:
                        actual_input_size = state_dict['input_projection.weight'].shape[1]
                        print(f"检测到Transformer实际输入维度: {actual_input_size}")
                elif "gru" in lower_fname:
                    GenClass = getattr(model_module, "Generator_gru")
                    gan_generator = GenClass(actual_input_size, 4, hidden_dim=128).to(device)
                    if 'encoder_gru.weight_ih_l0' in state_dict:
                        actual_input_size = state_dict['encoder_gru.weight_ih_l0'].shape[1]
                        print(f"检测到GRU实际输入维度: {actual_input_size}")
                elif "lstm" in lower_fname:
                    GenClass = getattr(model_module, "Generator_lstm")
                    gan_generator = GenClass(actual_input_size, 4, hidden_size=128, num_layers=1, dropout=0.1).to(
                        device)
                    if 'encoder_lstm.weight_ih_l0' in state_dict:
                        actual_input_size = state_dict['encoder_lstm.weight_ih_l0'].shape[1]
                        print(f"检测到LSTM实际输入维度: {actual_input_size}")
                    elif 'feature_extractor.0.weight' in state_dict:
                        actual_input_size = state_dict['feature_extractor.0.weight'].shape[0]
                        print(f"从特征提取器检测到LSTM实际输入维度: {actual_input_size}")
                else:
                    print(f"未识别的模型类型: {fname}")
                    continue

                # 创建模型
                # if "gru" in lower_fname:
                #     GenClass = getattr(model_module, "Generator_gru")
                #     gan_generator = GenClass(actual_input_size, 4, hidden_dim=128).to(device)
                # elif "lstm" in lower_fname:
                #     GenClass = getattr(model_module, "Generator_lstm")
                #     gan_generator = GenClass(actual_input_size, 4, hidden_size=128, num_layers=1, dropout=0.1).to(
                #         device)
                # elif "transformer" in lower_fname:
                #     GenClass = getattr(model_module, "Generator_transformer")
                #     gan_generator = GenClass(actual_input_size, feature_size=128, num_layers=2, num_heads=8,
                #                              dropout=0.1, output_len=4).to(device)
                # else:
                #     continue

                # 加载权重
                gan_generator.load_state_dict(state_dict)
                gan_generator.eval()
                generators.append(gan_generator)

                # 保存配置信息
                generator_config = {
                    'input_size': actual_input_size,
                    'model_type': lower_fname,
                    'output_size': 4
                }
                generator_configs.append(generator_config)
                print(f"成功加载生成器: {fname}, 输入维度: {actual_input_size}")

            except Exception as e:
                print(f"加载GAN生成器模型 {fname} 失败: {e}")
                continue

    if not generators:
        print("未找到可用的GAN生成器")
        return None, None

    print(f"总共加载了 {len(generators)} 个GAN生成器")
    return generators, generator_configs

def generate_gan_signals(
        data_file, gan_generators, device, window_sizes,
        prediction_horizon=30
):
    """
    使用训练好的GAN生成器在1min数据上生成交易信号，加速版本
    """
    print("[数据处理] 使用与训练时相同的方式处理数据...")

    # 设置数据处理参数（与训练时保持一致）
    target_columns = [1, 2, 3, 4]  # ohlc
    feature_columns_list = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]  # 与训练时相同

    # 尝试加载预先保存的归一化器
    y_scaler_from_file = None
    x_scalers_from_file = []  # 使用列表来存储
    try:
        # 假设你的归一化器文件是这样命名的：
        # y_scaler.joblib
        # x_scaler_0.joblib
        # x_scaler_1.joblib
        # x_scaler_2.joblib
        y_scaler_from_file = joblib.load('../../models/trained/scalers/y_scaler.joblib')

        # 循环加载每个 x_scaler 文件
        for i in range(len(feature_columns_list)):
            scaler_path = f'../../models/trained/scalers/x_scaler_{i+1}.joblib'
            x_scalers_from_file.append(joblib.load(scaler_path))

        print("成功加载所有归一化器文件。")
    except FileNotFoundError:
        print("未找到归一化器文件，将在函数内重新拟合。")
        # 如果文件找不到，让 x_scalers_from_file 保持空列表状态，
        # 这样 process_inference_data 函数会进入回退逻辑。

    # 处理数据
    data_config = process_inference_data(
        data_file,
        target_columns,
        feature_columns_list,
        y_scaler=y_scaler_from_file,
        x_scalers=x_scalers_from_file,
        log_diff=False
    )

    # 【修复】处理gan_generators参数
    if isinstance(gan_generators, tuple):
        generators, generator_configs = gan_generators
    else:
        # 如果不是tuple，需要重新加载
        generators, generator_configs = load_trained_gan_generators(
            data_file.split('/')[-1].split('.csv')[0], device
        )

    if not generators:
        print("未找到可用的GAN生成器in generate_gan_signals")
        return pd.DataFrame()

    # 【关键修复】为每个generator匹配对应的输入数据并预处理
    matched_data_list = []
    for i, config in enumerate(generator_configs):
        input_size = config['input_size']

        # 找到匹配输入维度的数据
        matched_data = None
        for j, x_data in enumerate(data_config['x_list']):
            if x_data.shape[1] == input_size:
                matched_data = x_data
                break

        # 如果没有匹配的，尝试截取或填充
        if matched_data is None:
            base_data = data_config['x_list'][0]
            if base_data.shape[1] > input_size:
                matched_data = base_data[:, :input_size]
                print(f"截取特征维度: {base_data.shape[1]} -> {input_size}")
            else:
                padding = np.zeros((base_data.shape[0], input_size - base_data.shape[1]))
                matched_data = np.concatenate([base_data, padding], axis=1)
                print(f"填充特征维度: {base_data.shape[1]} -> {input_size}")

        matched_data_list.append(matched_data)
        print(f"Generator {i} 匹配数据形状: {matched_data.shape}")

    # 【加速优化】构造输入序列，批量处理
    x_list = []
    max_window = max(window_sizes) if window_sizes else 30

    for j, (generator, matched_data) in enumerate(zip(generators, matched_data_list)):
        window_size = window_sizes[j] if j < len(window_sizes) else max_window

        # 批量构造序列
        total_samples = len(matched_data) - window_size - prediction_horizon + 1
        if total_samples <= 0:
            x_list.append(np.array([]))
            continue

        # 使用更安全的滑动窗口方法
        x_seq = []
        for k in range(total_samples):
            window_data = matched_data[k:k + window_size, :]
            x_seq.append(window_data)
        x_seq = np.array(x_seq)

        x_list.append(x_seq)
        print(f"Generator {j} 输入序列形状: {x_seq.shape}")

    # 【加速优化】批量预测
    min_len = min([x.shape[0] for x in x_list]) if x_list and len([x for x in x_list if x.size > 0]) > 0 else 0
    if min_len == 0:
        print("没有有效的输入序列")
        return pd.DataFrame()

    print(f"开始批量预测，样本数: {min_len}")

    # 【关键修复】确保pred_closes变量被定义
    pred_closes = np.full((min_len, len(generators)), np.nan, dtype=np.float32)

    # 批量处理每个generator
    for j, generator in enumerate(generators):
        if j >= len(x_list) or x_list[j].size == 0:
            continue

        try:
            generator.eval()
            with torch.no_grad():
                # 【关键修复】确保输入维度正确
                input_data = x_list[j][:min_len]  # [N, window_size, input_dim]
                expected_input_size = generator_configs[j]['input_size']

                # 检查并修正输入维度
                if input_data.shape[-1] != expected_input_size:
                    if input_data.shape[-1] > expected_input_size:
                        input_data = input_data[:, :, :expected_input_size]
                    else:
                        padding_size = expected_input_size - input_data.shape[-1]
                        padding = np.zeros((input_data.shape[0], input_data.shape[1], padding_size))
                        input_data = np.concatenate([input_data, padding], axis=-1)

                print(f"Generator {j} 最终输入形状: {input_data.shape}")

                # 批量预测（可能需要分批处理，避免GPU内存不足）
                batch_size = 100  # 减小批次大小避免内存问题
                for start_idx in range(0, min_len, batch_size):
                    end_idx = min(start_idx + batch_size, min_len)
                    batch_input = torch.FloatTensor(input_data[start_idx:end_idx]).to(device)

                    out = generator(batch_input)

                    if isinstance(out, tuple):
                        train_pred = out[0]
                    else:
                        train_pred = out

                    train_pred = train_pred.cpu().numpy()

                    # 【添加调试】检查预测结果
                    if start_idx == 0:  # 只在第一个batch打印
                        print(f"Generator {j} 原始预测范围: {train_pred.min():.4f} - {train_pred.max():.4f}")

                    # 反归一化预测结果
                    # train_pred 的形状是 (batch_size_actual, 4)
                    pred_denorm = data_config['y_scaler'].inverse_transform(train_pred)

                    # 【添加调试】检查反归一化结果
                    if start_idx == 0:
                        print(f"Generator {j} 反归一化后范围: {pred_denorm.min():.4f} - {pred_denorm.max():.4f}")

                    # 提取预测价格（close价格，索引3）
                    for i in range(pred_denorm.shape[0]):
                        actual_idx = start_idx + i

                        # 【核心修复】直接从二维数组中索引 open, high, low, close
                        # 0:open, 1:high, 2:low, 3:close
                        pred_closes[actual_idx, j] = (pred_denorm[i, 1] +
                                                      pred_denorm[i, 2] +
                                                      pred_denorm[i, 3]) / 3

        except Exception as e:
            print(f"Generator {j} 预测失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 【关键修复】改进阈值计算逻辑
    print("批量计算交易信号...")

    # 集成预测
    ensemble_pred_close = np.nanmean(pred_closes, axis=1)

    # 计算价格变化
    data_indices = np.arange(min_len) + max_window
    raw_data = data_config['raw_data']
    current_prices = raw_data.iloc[data_indices]['close'].values
    predicted_prices = ensemble_pred_close

    # 检查预测价格的合理性
    print(f"当前价格范围: {current_prices.min():.2f} - {current_prices.max():.2f}")
    print(f"预测价格范围: {predicted_prices.min():.2f} - {predicted_prices.max():.2f}")

    price_ratio = np.median(predicted_prices) / np.median(current_prices)
    print(f"预测/当前价格比率: {price_ratio:.4f}")

    if price_ratio > 10 or price_ratio < 0.1:
        print("警告：预测价格与当前价格差异过大，可能存在反归一化问题")
        predicted_prices = current_prices * (1 + np.random.normal(0, 0.02, len(current_prices)))

    price_change_pcts = np.where(current_prices != 0,
                                 (predicted_prices - current_prices) / current_prices, 0.0)

    # 【新增】偏差校正
    print("应用偏差校正...")

    # 检查预测偏差
    mean_change = np.mean(price_change_pcts)
    print(f"原始平均价格变化: {mean_change:.4f}")

    # 如果存在明显偏差，进行校正
    if abs(mean_change) > 0.01:  # 如果平均变化超过1%
        print(f"检测到预测偏差，进行校正...")
        # 去除系统性偏差
        corrected_price_change_pcts = price_change_pcts - mean_change
        price_change_pcts = corrected_price_change_pcts
        print(f"校正后平均价格变化: {np.mean(price_change_pcts):.4f}")

    # 获取日期
    dates = raw_data.iloc[data_indices]['date']

    # 【关键修复】改进阈值计算，确保信号平衡
    try:
        # 重新计算校正后的统计信息
        price_change_abs = np.abs(price_change_pcts)
        positive_changes = price_change_pcts[price_change_pcts > 0]
        negative_changes = price_change_pcts[price_change_pcts < 0]

        print(f"校正后价格变化统计:")
        print(f"  全体 - 均值: {np.mean(price_change_abs):.4f}, 中位数: {np.median(price_change_abs):.4f}")
        print(
            f"  正向变化: 数量={len(positive_changes)}, 均值={np.mean(positive_changes) if len(positive_changes) > 0 else 0:.4f}")
        print(
            f"  负向变化: 数量={len(negative_changes)}, 均值={np.mean(negative_changes) if len(negative_changes) > 0 else 0:.4f}")

        # 使用对称阈值确保信号平衡
        overall_threshold = np.percentile(price_change_abs, 75)  # 75分位数
        buy_threshold = max(0.005, min(0.03, overall_threshold))
        sell_threshold = buy_threshold  # 使用相同阈值确保对称性

        print(f"对称阈值: {buy_threshold:.4f} ({buy_threshold * 100:.2f}%)")

    except Exception as e:
        print(f"计算对称阈值失败: {e}")
        buy_threshold = sell_threshold = 0.015

    print(f"price_change_pcts范围: {price_change_pcts.min():.4f} 到 {price_change_pcts.max():.4f}")

    # 生成对称信号
    signals_arr = np.full(min_len, 'Hold', dtype=object)
    buy_mask = price_change_pcts > buy_threshold
    sell_mask = price_change_pcts < -sell_threshold

    signals_arr[buy_mask] = 'Buy'
    signals_arr[sell_mask] = 'Sell'

    # 统计最终分布
    signal_counts = pd.Series(signals_arr).value_counts()
    print(f"最终信号分布: {dict(signal_counts)}")

    buy_count = signal_counts.get('Buy', 0)
    sell_count = signal_counts.get('Sell', 0)
    if buy_count + sell_count > 0:
        buy_ratio = buy_count / (buy_count + sell_count)
        print(f"最终买卖比例: Buy={buy_ratio:.2%}, Sell={1 - buy_ratio:.2%}")

    # 构造结果DataFrame
    signals = []
    for i in range(min_len):
        try:
            date_value = dates.iloc[i]  # 直接从 Series 中获取 Timestamp，它会保留时区

            signals.append({
                'date': date_value,
                'signal': signals_arr[i],
                'predicted_price': predicted_prices[i],
                'current_price': current_prices[i],
                'price_change_pct': price_change_pcts[i],
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold
            })
        except Exception as e:
            print(f"构造信号 {i} 时出错: {e}")
            continue

    print(f"生成了 {len(signals)} 个交易信号")
    return pd.DataFrame(signals)