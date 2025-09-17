import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- 科研风全局样式 ---
sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['SimHei', 'Microsoft YaHei', 'Heiti TC', 'WenQuanYi Micro Hei']

# --- 模型映射 ---
name_map = {
    "macd": "MACD",
    "itransformer": "iTransformer",
    "fits": "FITS",
    "ptransformer": "PatchTST",
    "maa": "MAA"
}

# 突出 MAA，其余灰色
color_map = {
    "macd": "#AAAAAA",
    "itransformer": "#BBBBBB",
    "fits": "#CCCCCC",
    "ptransformer": "#DDDDDD",
    "maa": "#1f77b4"   # 深蓝，醒目
}

def filter_by_time(df, start_date, end_date, time_col="timestamp"):
    if time_col not in df.columns:
        return df
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
    return df.loc[mask].copy()

def plot_bar(ax, data_dict, metric_name, sub_title):
    """
    绘制单个子图的科研风柱状图
    """
    avg_values = {model: np.nanmean(values) for model, values in data_dict.items()}
    models = list(avg_values.keys())
    values = [avg_values[m] for m in models]

    colors = [color_map.get(m, "#333333") for m in models]
    x = np.arange(len(models))

    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9, width=0.6)

    # 数值标注（小号，避免杂乱）
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=9, color="black")

    mapped_labels = [name_map.get(m, m) for m in models]
    ax.set_xticks(x)
    ax.set_xticklabels(mapped_labels, fontsize=12,rotation=0)

    ax.set_title(sub_title, fontsize=16, pad=8)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    sns.despine(ax=ax)
    return bars, mapped_labels


# ===== 主循环 =====
base_path = '../results'
symbols = ["rb9999", "i9999", "ni9999", "OI9999", "AP9999"]
top_models = ["macd", "itransformer", "fits", "ptransformer", "maa"]

trades = ["buy"]
sub_models = ["our-transformer"]
intervals = [15, 30, 60]
execution_windows = [15, 30, 60]

start_date_str = '2025-04-01 00:00:00+08:00'
end_date_str = '2025-05-30 23:59:59+08:00'

start_date = pd.to_datetime(start_date_str)
end_date = pd.to_datetime(end_date_str)
period_days = (end_date - start_date).days

# 有效的频率组合 (6个)
freq_combinations = [(15, 15), (30, 15), (30, 30), (60, 15), (60, 30), (60, 60)]

for sub_model in sub_models:
    for symbol in symbols:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        legend_handles = None
        legend_labels = None

        for i, (interval, execution_window) in enumerate(freq_combinations):
            ax = axes[i]
            return_per_day_data = {m: [] for m in top_models}

            for top_model in top_models:
                all_per_day = []
                for trade in trades:
                    path = os.path.join(
                        base_path,
                        top_model,
                        trade,
                        sub_model,
                        symbol,
                        f"{interval}min_win{execution_window}",
                        "metrics.csv"
                    )
                    try:
                        df = pd.read_csv(path)
                        df = filter_by_time(df, start_date, end_date, time_col="start_time")
                        total_ret = df['Total Return'].sum()
                        all_per_day.append(total_ret / period_days)
                    except Exception:
                        all_per_day.append(np.nan)

                return_per_day_data[top_model] = [v for v in all_per_day if not np.isnan(v)] or [np.nan]

            sub_title = f"{interval}min / win{execution_window}"
            bars, labels = plot_bar(ax, return_per_day_data, "日均收益", sub_title)

            if i == 0:
                legend_handles = bars
                legend_labels = labels

        # 总标题（科研风，粗体，图外）
        fig.suptitle(f"{symbol} | 预测器: {sub_model}", fontsize=20, fontweight="bold", y=0.97 )

        # 图例放下方，横排
        fig.legend(
            legend_handles,
            legend_labels,
            title="信号生成器",
            loc="lower center",
            ncol=len(legend_labels),
            fontsize=18,
            frameon=False
        )

        plt.subplots_adjust(top=0.90, bottom=0.12, hspace=0.35, wspace=0.25)
        plt.show()
