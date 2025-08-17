import pandas as pd
import torch
import os
import argparse

from generate_signal import generate_bollinger_signals_with_strength
from src.core.strategy import save_directional_signals
from src.utils.data_utils import resample_data,set_seed


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
                parser.add_argument('--vae_model_path', type=str, default=f'./models/{symbol}_tdist_vae_model.pth')
                parser.add_argument('--predictor_model_path', type=str,
                                    default=f'./predictor_models/{symbol}_{model_name}_predictor_model.pth')
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
                    predictor_model_path = f"./predictor_models/{symbol}_predictor_model.pth"
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
                        # 1. 根据信号生成器，生成buy/sell,只保留强度超过阈值的信号
                        bollinger_params = {"length": length_param, "std": std_param, "stl_param": stl_param,
                                            "n_param": n_param}
                        # 使用 resampled_df 作为输入，并确保它有时间索引
                        signals_df = resampled_df.copy()
                        bollinger_signal_df = generate_bollinger_signals_with_strength(signals_df, **bollinger_params)

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



                    except Exception as e:
                        print(f"[{selected_interval_label}] 发生错误: {e}")
                        import traceback
                        print(traceback.format_exc())





