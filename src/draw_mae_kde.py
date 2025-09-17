import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

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

# --- 你已有的映射（如已定义可删除这里） ---
name_map = {"macd":"MACD","itransformer":"iTransformer","fits":"FITS","ptransformer":"PatchTST","maa":"MAA"}
color_map = {"macd":"#E7DAD2","itransformer":"#82B0D2","fits":"#8ECFC9","ptransformer":"#BEB8DC","maa":"#FA7F6F"}
top_models = ["macd","itransformer","fits","ptransformer","maa"]

sns.set_style("whitegrid")
sns.set_context("talk")  # 论文里常用 'paper' 或 'talk'，这里用 talk 字体稍大

def plot_distribution(data_dict, title_prefix, metric_label, ax, top_models, name_map, color_map,
                      show_legend=False, xlim=None):
    """
    data_dict: {model_key: list_of_values}   <- 值为原始比例（例如 0.001 表示 0.1%）
    ax: matplotlib Axes
    metric_label: string e.g. "Drawdown" or "Return"
    top_models/name_map/color_map: 来自脚本的全局映射，保证顺序和颜色一致
    show_legend: 是否在此子图显示图例（通常只在第一个子图显示）
    xlim: tuple (xmin, xmax) 或 None —— 若提供则统一 x 轴范围（单位为百分比）
    """
    # 收集数据并转换成百分比表示
    rows = []
    for m in top_models:
        vals = data_dict.get(m, [])
        if not vals:
            continue
        for v in vals:
            # 将原始比例放大 100 -> 百分比
            rows.append([name_map.get(m, m), v * 100.0, m])

    if len(rows) == 0:
        ax.text(0.5, 0.5, "无数据", ha="center", va="center", fontsize=11, color="gray",
                transform=ax.transAxes)
        ax.set_title(title_prefix, fontsize=11)
        ax.set_xlabel(f"{metric_label}")
        ax.set_ylabel("Density")
        return

    df_plot = pd.DataFrame(rows, columns=["Strategy", metric_label, "model_key"])

    # palette: map visible Strategy 名称 -> color (保证 MAA 醒目)
    palette = {}
    for k in top_models:
        label = name_map.get(k, k)
        # 若 color_map 中无定义，则使用灰色作为回退
        palette[label] = color_map.get(k, "#B0B0B0")

    # 为了保持图例项的次序（与 top_models 一致）构造 hue_order，仅包含当前存在的
    present = set(df_plot["Strategy"].unique().tolist())
    hue_order = [name_map[m] for m in top_models if name_map[m] in present]

    # KDE 绘制（不填充，科研风格，线条清晰）
    for strat in hue_order:
        sub = df_plot.loc[df_plot["Strategy"] == strat, metric_label]
        if sub.empty:
            continue
        # seaborn kdeplot 单独画以便控制每条线的样式与顺序
        sns.kdeplot(sub,
                    ax=ax,
                    label=strat if show_legend else None,
                    linewidth=1.6,
                    fill=False,
                    bw_method="scott",
                    common_norm=False,
                    clip_on=True,
                    color=palette.get(strat),
                    alpha=0.95 if strat == name_map.get("maa") else 0.7,
                    )

    # 美化坐标与标题
    ax.set_title(title_prefix, fontsize=20)
    ax.set_xlabel(f"{metric_label}", fontsize=16)
    ax.set_ylabel("Density", fontsize=18)

    # x 轴显示百分号（数值已为百分比）
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))

    # 统一 x 轴范围（若传入）
    if xlim is not None:
        ax.set_xlim(xlim)

    # 仅在请求时显示图例，且放在子图外侧顶部（避免遮挡）
    if show_legend:
        leg = ax.legend(title="Strategy", fontsize=16, title_fontsize=17,
                        loc="upper right", frameon=False)
        # 适当微调图例位置防止遮挡
        # leg.set_bbox_to_anchor((1.02, 1.0))  # 需要时可以取消注释，或在 fig.legend 全局显示

    # 细节：去掉顶部与右侧脊
    sns.despine(ax=ax)

# ----------------------------
# 主流程：为每个 symbol 生成一张 2x3 的图（回撤分布示例）
# ----------------------------
# 这里选定 6 个 (interval, window) 的组合，注意必须满足 win <= interval
combos = [(15, 15), (30, 15), (30, 30), (60, 15), (60, 30), (60, 60)]

# 假设 variables 已在脚本中定义：symbols, top_models, name_map, color_map, sub_models, trades
for symbol in symbols:
    for sub_model in sub_models:
        for trade in trades:
            # 先批量加载 6 个组合的数据（避免重复 IO）
            combo_dd_list = []
            for interval, win in combos:
                ret_dict, dd_dict = load_metric_data(symbol, interval, win, trade, sub_model)
                # dd_dict: {model: [values]}  —— 我们将直接传给绘图函数
                combo_dd_list.append(((interval, win), dd_dict))

            # 计算全局 xlim（百分比单位），以便子图间可比
            all_vals = []
            for (_, dd) in combo_dd_list:
                for m in top_models:
                    vs = dd.get(m, [])
                    if vs:
                        all_vals.extend([v * 100.0 for v in vs])
            if len(all_vals) > 0:
                xmin = min(all_vals)
                xmax = max(all_vals)
                # 加一点边距
                span = xmax - xmin if (xmax - xmin) != 0 else max(abs(xmin), 1.0)
                margin = span * 0.08
                global_xlim = (xmin - margin, xmax + margin)
            else:
                global_xlim = None

            # 创建 2x3 画布
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()
            for idx, ((interval, win), dd_dict) in enumerate(combo_dd_list):
                ax = axes[idx]
                title = f"{symbol}  {interval}min / win{win}  {trade}"
                # 仅在第一个子图显示图例
                show_legend = (idx == 0)
                plot_distribution(dd_dict, title, "Drawdown", ax,
                                  top_models=top_models, name_map=name_map, color_map=color_map,
                                  show_legend=show_legend, xlim=global_xlim)

            # 总标题（避免压到子图）
            fig.suptitle(f"{symbol} — 不同周期/执行窗口下的回撤分布 ({sub_model}, {trade})",
                         fontsize=20, y=1.12)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
