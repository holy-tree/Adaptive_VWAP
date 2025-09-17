import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- 中文显示 ---
plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False

# --- 模型映射 ---
name_map = {
    "macd": "MACD",
    "itransformer": "iTransformer",
    "fits": "FITS",
    "ptransformer": "PatchTST",
    "maa": "MAA"
}

color_map = {
    "macd": "#E7DAD2",
    "itransformer": "#82B0D2",
    "fits": "#8ECFC9",
    "ptransformer": "#BEB8DC",
    "maa": "#FA7F6F"
}

# ===== 全局配置 =====
base_path = '../results'
symbols = ["rb9999", "i9999", "ni9999", "OI9999", "AP9999"]
top_models = ["macd", "itransformer", "fits", "ptransformer", "maa"]
trades = ["buy"]
sub_models = ["our-transformer"]
intervals = [15,30,60]
execution_windows = [15,30,60]

start_date_str = '2025-05-01 00:00:00+08:00'
end_date_str = '2025-05-30 23:59:59+08:00'
start_date = pd.to_datetime(start_date_str)
end_date = pd.to_datetime(end_date_str)

def filter_by_time(df, start_date, end_date, time_col="timestamp"):
    if time_col not in df.columns:
        return df
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
    return df.loc[mask].copy()


def load_metric_data(symbol, interval, win, trade, sub_model):
    """
    读取所有模型的数据并返回回撤和收益字典
    """
    ret_dict = {m: [] for m in top_models}
    dd_dict = {m: [] for m in top_models}

    for model in top_models:
        path = os.path.join(
            base_path,
            model,
            trade,
            sub_model,
            symbol,
            f"{interval}min_win{win}",
            "metrics.csv"
        )
        try:
            df = pd.read_csv(path)
            df = filter_by_time(df, start_date, end_date, time_col="start_time")
            if "Total Return" in df.columns and "Max Adverse Excursion (MAE)" in df.columns:
                ret_dict[model] = df["Total Return"].dropna().tolist()
                dd_dict[model] = df["Max Adverse Excursion (MAE)"].dropna().tolist()
        except Exception as e:
            print(f"⚠️ 无法读取 {path} - {e}")
    print(f"MACD return data length: {len(ret_dict['macd'])}")
    print(f"MACD drawdown data length: {len(dd_dict['macd'])}")
    return ret_dict, dd_dict

def plot_distribution(data_dict, title_prefix, metric_col, metric_label):
    """
    data_dict: {模型: 数值列表}
    metric_col: 表头里的真实列名，比如 'Total Return' / 'Max Adverse Excursion (MAE)'
    metric_label: 用于图里显示的名字，比如 'Return' / 'Drawdown'
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    df_plot = []
    for model, values in data_dict.items():
        if len(values) == 0:
            continue
        for v in values:
            # 将数值转换为百分比（乘以100）
            df_plot.append([name_map[model], v * 100]) # 转换成百分比

    if len(df_plot) == 0:
        print(f"⚠️ {title_prefix} 没有数据，跳过绘图")
        plt.close(fig)
        return

    df_plot = pd.DataFrame(df_plot, columns=["Strategy", metric_label])

    palette = {name_map[k]: v for k, v in color_map.items()}

    sns.kdeplot(data=df_plot, x=metric_label, hue="Strategy",
                fill=False, common_norm=False, palette=palette, ax=ax)

    # 修改y轴标签，加上百分号
    ax.set_title(f"{title_prefix} Density")
    ax.set_ylabel("Density")

    # 如果你希望x轴也显示百分比，可以这样做：
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda y, _: '{:.2f}%'.format(y))
    ax.xaxis.set_major_formatter(formatter)


    plt.tight_layout()

    plt.text(0.5, 1.02, f"{title_prefix} 分布可视化",
             ha="center", va="bottom",
             transform=fig.transFigure, fontsize=16, weight="bold")

    plt.show()

def plot_signal_frequency(symbol, interval, win, trade, sub_model):
    """
    统计不同模型生成的信号数量，并画柱状图
    """
    freq_dict = {}
    for model in top_models:
        path = os.path.join(
            base_path, model, "gan_signals", f"gan_signals_{interval}min_{symbol}.csv"
        )
        try:
            df = pd.read_csv(path)
            df = filter_by_time(df, start_date, end_date, time_col="date")
            freq_dict[name_map[model]] = len(df)
        except Exception as e:
            print(f"⚠️ 无法读取 {path} - {e}")
            freq_dict[name_map[model]] = 0

    # 转换为 DataFrame
    df_plot = pd.DataFrame(list(freq_dict.items()), columns=["Strategy", "Signal Count"])
    palette = {name_map[k]: v for k, v in color_map.items()}

    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_plot, x="Strategy", y="Signal Count", palette=palette)
    plt.title(f"{symbol} {interval}min/{win}min {trade} Signal Frequency")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_slippage_distribution(symbol, interval, win, trade, sub_model):
    """
    绘制不同模型的滑点分布 (Slippage Reduction (BPS))，只用小提琴图
    """
    df_plot = []
    for model in top_models:
        path = os.path.join(
            base_path, model, trade, sub_model, symbol,
            f"{interval}min_win{win}", "metrics.csv"
        )
        try:
            df = pd.read_csv(path)
            df = filter_by_time(df, start_date, end_date, time_col="start_time")
            if "Slippage Reduction (BPS)" in df.columns:
                for v in df["Slippage Reduction (BPS)"].dropna():
                    df_plot.append([name_map[model], v])
        except Exception as e:
            print(f"⚠️ 无法读取 {path} - {e}")

    if len(df_plot) == 0:
        print(f"⚠️ {symbol} {interval}min/{win}min {trade} 没有滑点数据")
        return

    df_plot = pd.DataFrame(df_plot, columns=["Strategy", "Slippage (BPS)"])
    palette = {name_map[k]: v for k, v in color_map.items()}

    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df_plot, x="Strategy", y="Slippage (BPS)",
                   palette=palette, cut=0, scale="width")
    plt.title(f"{symbol} {interval}min/{win}min {trade} Slippage Distribution (Violin)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()





# ===== 主逻辑：生成图表 =====
for symbol in symbols:
    for sub_model in sub_models:
        for interval in intervals:
            for win in execution_windows:
                for trade in trades:
                    ret_dict, dd_dict = load_metric_data(symbol, interval, win, trade, sub_model)

                    # 图 1 回撤
                    # plot_distribution(
                    #     dd_dict,
                    #     f"{symbol} {interval}min {trade} Drawdown",
                    #     "Max Adverse Excursion (MAE)", # 真实列名
                    #     "Drawdown"                     # 图里显示的标签
                    # )

                    # 图 2 收益
                    plot_distribution(
                        ret_dict,
                        f"{symbol} {interval}min/{win}min {trade} Return",
                        "Total Return", # 真实列名
                        "Return"        # 图里显示的标签
                    )

                    # 图 3 信号频率
                    # plot_signal_frequency(symbol, interval, win, trade, sub_model)

                    # 图 4 滑点分布
                    # plot_slippage_distribution(symbol, interval, win, trade, sub_model)
