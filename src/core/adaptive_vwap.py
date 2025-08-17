import os

import pandas as pd
import torch
import argparse

from generate_signal import generate_bollinger_signals_with_strength, generate_bollinger_signals
from src.core.strategy import save_directional_signals, run_adaptive_backtest
from src.utils.data_utils import resample_data, set_seed, load_financial_data, get_signal_index, \
    get_and_validate_signals
from src.utils.model_utils import load_predictor_model, load_vae_model

SEQ_LENGTH = 345

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义周期与可选execution_window的映射
interval_to_windows = {
    '1min': [1],
    '5min': [1, 5],
    '15min': [1, 5, 15],
    '30min': [1, 5, 15, 30],
    '60min': [1, 5, 15, 30, 60],
    'D': [1, 5, 15, 30, 60, SEQ_LENGTH]  # 假设日线SEQ_LENGTH分钟
}

time_interval_map = {
    '1分钟': '1min', '5分钟': '5min', '15分钟': '15min',
    '30分钟': '30min', '1小时': '60min', '日线': 'D'
}

if __name__ == "__main__":

    for trade_direction in ["buy", "sell"]:
        for model_name in ["our-transformer", "rnn", "lstm"]:
            for symbol in ["rb9999", "i9999", "cu9999", "ni9999", "sc9999", "pg9999", "y9999", "ag9999", "m9999",
                           "c9999", "TA9999", "UR9999", "OI9999", "au9999", "IH9999", "T9999", "CF9999", "AP9999"]:
                set_seed(42)
                # 参数设置
                parser = argparse.ArgumentParser(description="VWAP批量回测参数")
                parser.add_argument('--data_file', type=str, default=f'../../data/latent_features/{symbol}_1min_data.csv')
                parser.add_argument('--vae_model_path', type=str, default=f'../../models/trained/VAE_models/{symbol}_tdist_vae_model.pth')
                parser.add_argument('--predictor_model_path', type=str,
                                    default=f'../../models/trained/predictor_models/{symbol}_{model_name}_predictor_model.pth')
                parser.add_argument('--total_quantity', type=float, default=10000)
                parser.add_argument('--failsafe_ratio', type=float, default=0.75)
                parser.add_argument('--execution_window', type=int, default=30)
                parser.add_argument('--length_param', type=int, default=20)
                parser.add_argument('--std_param', type=float, default=2.0)
                parser.add_argument('--stl_param', type=float, default=5.0)
                parser.add_argument('--n_param', type=float, default=6.0)
                parser.add_argument('--model_input_seq_length', type=int, default=345)
                parser.add_argument('--latent_dim', type=int, default=16)
                parser.add_argument('--ohlcv_dim', type=int, default=5)
                parser.add_argument('--intervals', type=str, default='1min,5min,15min,30min,60min,D',
                                    help='逗号分隔的周期列表，如1min,5min,15min,30min,60min,D')

                args = parser.parse_args()

                # 解析周期映射
                intervals = args.intervals.split(',')
                time_interval_map = {k: v for k, v in time_interval_map.items() if v in intervals}

                # 解析参数
                data_file = args.data_file
                vae_model_path = args.vae_model_path
                predictor_model_path = None
                if model_name != "our-transformer":
                    predictor_model_path = args.predictor_model_path
                else:
                    predictor_model_path = f"../../models/trained/predictor_models/{symbol}_predictor_model.pth"
                total_quantity = args.total_quantity
                failsafe_ratio = args.failsafe_ratio
                execution_window = args.execution_window
                length_param = args.length_param
                std_param = args.std_param
                stl_param = args.stl_param
                n_param = args.n_param
                model_input_seq_length = args.model_input_seq_length
                latent_dim = args.latent_dim
                ohlcv_dim = args.ohlcv_dim

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                print(f"---\n**分析开始** | 设备: `{device}`")

                # 开始遍历执行窗口
                for selected_interval_label, time_interval in time_interval_map.items():
                    try:
                        # 对原始1min数据进行重采样
                        print(f"[{selected_interval_label}] 1. 加载并重采样数据...")
                        raw_df_1min = pd.read_csv(data_file, parse_dates=['date'])
                        resampled_df = resample_data(raw_df_1min.copy(), time_interval)

                        # STEP 1:
                        # STEP 1.1 根据信号生成器，生成buy/sell,只保留强度超过阈值的信号
                        bollinger_params = {"length": length_param, "std": std_param, "stl_param": stl_param,
                                            "n_param": n_param}
                        # 使用 resampled_df 作为输入，并确保它有时间索引
                        signals_df = resampled_df.copy()
                        bollinger_signal_df = generate_bollinger_signals(signals_df, **bollinger_params)

                        allowed_windows = interval_to_windows.get(time_interval, [1])
                        # 基于阈值和信号类型筛选数据
                        for execution_window in allowed_windows:
                            save_directional_signals(
                                df=bollinger_signal_df,
                                symbol=symbol,
                                time_interval=time_interval,
                                execution_window=execution_window,
                                signal_type="bollinger",
                                signal_threshold=0.6
                            )



                        # STEP 1.2 预测 OHCLV
                        # 1. 加载模型
                        predictor_model = load_predictor_model(model_name, predictor_model_path, device=device)
                        vae_model = load_vae_model(vae_model_path, ohlcv_dim, latent_dim, 64, 5.0, device)

                        # 2. 获取最新日期
                        unique_dates = sorted(resampled_df['date'].dt.date.unique(), reverse=True)
                        if not unique_dates:
                            print(f"[{selected_interval_label}] 无可用日期，跳过。")
                        else:
                            selected_date = unique_dates[0]

                            # 3. 获取并验证第一个交易信号
                            first_signal = get_and_validate_signals(
                                df=bollinger_signal_df,
                                direction_filter="Buy" if trade_direction.lower() == "buy" else "Sell",
                                selected_date=selected_date
                            )

                            if first_signal is not None:
                                trade_direction = first_signal['signal']
                                signal_time = first_signal['date']

                                # 4. 获取信号在1分钟数据中的索引
                                signal_idx = get_signal_index(raw_df_1min, signal_time)

                                # 5. 加载1分钟数据集
                                dataset_1min = load_financial_data(data_file, model_input_seq_length, latent_dim)

                                # 6. 执行所有窗口的回测
                                allowed_windows = interval_to_windows.get(time_interval, [1])
                                for execution_window in allowed_windows:
                                    # 检查数据是否充足
                                    if signal_idx + execution_window > len(raw_df_1min):
                                        print(
                                            f"[{selected_interval_label}] execution_window={execution_window} 数据不足，跳过。")
                                        continue

                                    print(
                                        f"[{selected_interval_label}] 选定信号时间: {first_signal['date']}，信号索引: {signal_idx}, 执行窗口: {execution_window}分钟"
                                    )
                                    print(
                                        f"[{selected_interval_label}] 交易方向: {trade_direction}, 总交易量: {total_quantity}, 失败安全系数: {failsafe_ratio}"
                                    )
                                    print(f"[{selected_interval_label}] boll参数: {bollinger_params}")
                                    print(
                                        f"[{selected_interval_label}] 4. 执行回测... execution_window={execution_window}")

                                    # 调用回测函数
                                    results_df, metrics = run_adaptive_backtest(
                                        predictor_model,
                                        dataset_1min,
                                        raw_df_1min,
                                        signal_idx,
                                        model_input_seq_length,
                                        execution_window,
                                        total_quantity,
                                        trade_direction,
                                        failsafe_ratio,
                                        device,
                                        vae_model
                                    )
                                    # --- 保存结果 ---
                                    os.makedirs(f"../../results/{trade_direction}_results", exist_ok=True)
                                    base_filename = os.path.basename(data_file).replace('.csv', '')
                                    param_str = f"{length_param}_{std_param}_{stl_param}_{n_param}_win{execution_window}"
                                    results_df_name = f"aanew_boll_results_df_{trade_direction}_{execution_window}MINexecution_window_{model_name}_{symbol}_{time_interval}_{base_filename}_{param_str}.csv"
                                    metrics_name = f"aanew_boll_metrics_{trade_direction}_{execution_window}MINexecution_window_{model_name}_{symbol}_{time_interval}_{base_filename}_{param_str}.csv"
                                    results_df.to_csv(os.path.join(f"../../results/{trade_direction}_results", results_df_name))
                                    pd.DataFrame([metrics]).to_csv(
                                        os.path.join(f"../../results/{trade_direction}_results", metrics_name),
                                        index=False)
                                    print(
                                        f"[{selected_interval_label}] 回测完成，结果已保存为 {results_df_name} 和 {metrics_name}")



                    except Exception as e:
                        print(f"[{selected_interval_label}] 发生错误: {e}")
                        import traceback
                        print(traceback.format_exc())





