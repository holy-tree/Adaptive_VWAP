import pandas as pd
import matplotlib.font_manager as fm
import os

# 参数
base_path = '../all_results'
symbols = [
    "rb9999", "i9999", "cu9999", "ni9999", "sc9999", "pg9999", "y9999", "ag9999", "m9999",
    "c9999", "TA9999", "UR9999", "OI9999", "au9999", "IH9999", "T9999", "CF9999", "AP9999"
]

top_models = ["maa", "itransformer", "fits", "ptransformer","macd"]
trades = ["buy", "sell"]
sub_models = ["lstm", "rnn", "our-transformer"]
intervals = [15, 30, 60]
execution_windows = [15, 30, 60]

name_map = {
    "macd": "MACD",
    "itransformer": "iTransformer",
    "fits": "FITS",
    "ptransformer": "PatchTST",
    "maa": "MAA"
}

start_date_str = '2025-04-01 00:00:00+08:00'
end_date_str = '2025-06-30 23:59:59+08:00'
start_date = pd.to_datetime(start_date_str)
end_date = pd.to_datetime(end_date_str)

def get_chinese_font():
    """
    在 Windows 下自动选择可用的中文字体。
    """
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",   # 黑体
        "C:/Windows/Fonts/msyh.ttc",     # 微软雅黑
        "C:/Windows/Fonts/simsun.ttc"    # 宋体
    ]
    for path in font_paths:
        if os.path.exists(path):
            return fm.FontProperties(fname=path)
    raise RuntimeError("未找到可用的中文字体，请检查 C:/Windows/Fonts 目录")

my_font = get_chinese_font()

def filter_by_time(df, start_date, end_date, time_col="timestamp"):
    if time_col not in df.columns:
        return df
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    mask = (df[time_col] >= start_date) & (df[time_col] <= end_date)
    return df.loc[mask].copy()

# ===== 主循环 =====
for sub_model in sub_models:
    for interval in intervals:
        for execution_window in execution_windows:
            if execution_window > interval:
                continue

            # 存放结果 {model: [slippage across symbols]}
            model_results = {m: [] for m in top_models}

            for symbol in symbols:
                slippage_values = {}
                for top_model in top_models:
                    all_slip = []

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
                            all_slip.append(df["Slippage Reduction (BPS)"].mean())
                        except Exception:
                            all_slip.append(None)

                    slippage_values[top_model] = pd.Series(all_slip).mean(skipna=True)

                for top_model in top_models:
                    model_results[top_model].append(slippage_values[top_model])

            import matplotlib.pyplot as plt
            import numpy as np
            import seaborn as sns

            sns.set_style("whitegrid")
            sns.set_context("talk")  # 字体适中，论文感

            plt.figure(figsize=(12, 6))
            x = np.arange(len(symbols))
            width = 0.15

            # 定义颜色方案：MAA 深蓝，其余灰色
            colors = {
                "maa": "#1f77b4",  # 深蓝，醒目
                "itransformer": "#BBBBBB",
                "fits": "#DDDDDD",
                "ptransformer": "#CCCCCC",
                "macd": "#AAAAAA"
            }

            for i, top_model in enumerate(top_models):
                bars = plt.bar(
                    x + i * width,
                    model_results[top_model],
                    width,
                    label=name_map.get(top_model, top_model),
                    color=colors[top_model],
                    edgecolor="black",
                    linewidth=1.0,
                    alpha=0.9 if top_model == "maa" else 0.7
                )

            # 横轴标签
            plt.xticks(
                x + width * (len(top_models) - 1) / 2,
                symbols,
                rotation=30,
                ha="right",
                fontproperties=my_font
            )

            # 基准线
            plt.axhline(0, color="black", linestyle="--", linewidth=1)

            plt.xlabel("合约品种", fontsize=13, fontproperties=my_font)
            plt.ylabel("滑点改善 (BPS)", fontsize=13, fontproperties=my_font)

            # 用 suptitle 避免标题压图
            plt.suptitle(
                f"滑点对比 | 预测器: {sub_model} | interval={interval}",
                fontsize=15,
                fontproperties=my_font,
                y=0.95  # 往上抬
            )

            plt.legend(
                title="信号生成器",
                bbox_to_anchor=(0.5, 1.12),
                loc="center",
                ncol=len(top_models),
                frameon=False,
                prop=my_font,
                title_fontproperties=my_font
            )

            sns.despine()
            plt.tight_layout()
            plt.show()



