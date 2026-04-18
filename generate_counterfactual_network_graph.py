from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "counterfactual_outputs"

mpl.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False

PROVINCE_ABBR = {
    "北京": "BJ", "天津": "TJ", "河北": "HEB", "山西": "SX", "内蒙古": "NM",
    "辽宁": "LN", "吉林": "JL", "黑龙江": "HL", "上海": "SH", "江苏": "JS",
    "浙江": "ZJ", "安徽": "AH", "福建": "FJ", "江西": "JX", "山东": "SD",
    "河南": "HEN", "湖北": "HUB", "湖南": "HUN", "广东": "GD", "广西": "GX",
    "海南": "HAN", "重庆": "CQ", "四川": "SC", "贵州": "GZ", "云南": "YN",
    "陕西": "SAX", "甘肃": "GS", "青海": "QH", "宁夏": "NX", "新疆": "XJ",
}

PROVINCE_EN = {
    "北京": "Beijing", "天津": "Tianjin", "河北": "Hebei", "山西": "Shanxi", "内蒙古": "Inner Mongolia",
    "辽宁": "Liaoning", "吉林": "Jilin", "黑龙江": "Heilongjiang", "上海": "Shanghai", "江苏": "Jiangsu",
    "浙江": "Zhejiang", "安徽": "Anhui", "福建": "Fujian", "江西": "Jiangxi", "山东": "Shandong",
    "河南": "Henan", "湖北": "Hubei", "湖南": "Hunan", "广东": "Guangdong", "广西": "Guangxi",
    "海南": "Hainan", "重庆": "Chongqing", "四川": "Sichuan", "贵州": "Guizhou", "云南": "Yunnan",
    "陕西": "Shaanxi", "甘肃": "Gansu", "青海": "Qinghai", "宁夏": "Ningxia", "新疆": "Xinjiang",
}


def load_geo_matrix() -> pd.DataFrame:
    df = pd.read_excel(BASE_DIR / "1.xlsx", sheet_name="2023")
    df = df.rename(columns={df.columns[0]: "province"})
    return df.set_index("province")


def circular_positions(nodes: list[str], target: str) -> dict[str, tuple[float, float]]:
    n = len(nodes)
    # 目标省份固定在底部中央附近，其他点顺时针排布成环
    target_angle = -np.pi / 2
    angles = np.linspace(target_angle, target_angle + 2 * np.pi, n, endpoint=False)

    order = nodes[:]
    if target in order:
        idx = order.index(target)
        order = order[idx:] + order[:idx]

    pos: dict[str, tuple[float, float]] = {}
    for node, angle in zip(order, angles):
        radius = 1.0
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        pos[node] = (x, y)

    return pos


def draw_network(target: str, shock_tag: str = "05") -> Path:
    detail_path = OUTPUT_DIR / f"counterfactual_{target}_plus_{shock_tag}sd_details.csv"
    detail_df = pd.read_csv(detail_path)
    geo = load_geo_matrix()

    nodes = detail_df["Province"].tolist()
    pos = circular_positions(nodes, target)
    detail = detail_df.set_index("Province")

    G = nx.Graph()
    for _, row in detail_df.iterrows():
        province = row["Province"]
        G.add_node(
            province,
            province_type=row["Province_Type"],
            ee_change=float(row["EE_Change"]),
        )

    geo_edges = []
    for i, src in enumerate(nodes):
        for dst in nodes[i + 1:]:
            if int(geo.loc[src, dst]) == 1:
                geo_edges.append((src, dst))
                G.add_edge(src, dst, edge_kind="geo")

    spill_edges = []
    for _, row in detail_df.iterrows():
        province = row["Province"]
        if province == target:
            continue
        spill_edges.append((target, province, float(row["EE_Change"])))

    fig, ax = plt.subplots(figsize=(12.5, 9.5))
    ax.set_facecolor("white")

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=geo_edges,
        width=1.2,
        edge_color="#f2cdd3",
        alpha=0.82,
        ax=ax,
    )

    max_spill = max((s for _, _, s in spill_edges), default=1.0)
    for src, dst, spill in spill_edges:
        norm = spill / max_spill if max_spill > 0 else 0.0
        width = 0.8 + 6.5 * norm
        color = (1.0, 1.0 - 0.72 * norm, 1.0 - 0.72 * norm)
        alpha = 0.18 + 0.55 * norm
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(src, dst)],
            width=width,
            edge_color=[color],
            alpha=alpha,
            ax=ax,
        )

    changes = [G.nodes[n]["ee_change"] for n in nodes]
    norm = mpl.colors.Normalize(vmin=min(changes), vmax=max(changes))
    cmap = plt.cm.coolwarm

    node_sizes = []
    node_colors = []
    edgecolors = []
    linewidths = []
    for node in nodes:
        change = G.nodes[node]["ee_change"]
        ptype = G.nodes[node]["province_type"]
        if ptype == "Target":
            node_sizes.append(360)
            edgecolors.append("#8b0000")
            linewidths.append(1.2)
        elif ptype == "Neighbor":
            node_sizes.append(180)
            edgecolors.append("#e29a00")
            linewidths.append(1.8)
        else:
            node_sizes.append(110)
            edgecolors.append("#444444")
            linewidths.append(0.5)
        node_colors.append(change)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=cmap,
        vmin=norm.vmin,
        vmax=norm.vmax,
        edgecolors=edgecolors,
        linewidths=linewidths,
        ax=ax,
    )

    labels = {n: PROVINCE_ABBR.get(n, n) for n in nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight="bold", ax=ax)

    ax.set_title(
        f"Spatial Network: Spillover Effects from {PROVINCE_EN.get(target, target)}\n"
        f"(Node color indicates EE change; edge thickness/color indicate spillover intensity, +0.50 SD shock)",
        fontsize=15,
        pad=12,
    )
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.04, pad=0.04)
    cbar.set_label("EE Change", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    legend_lines = [
        plt.Line2D([0], [0], color=(1.0, 0.78, 0.78), lw=1.3, label="Weak spillover"),
        plt.Line2D([0], [0], color=(1.0, 0.42, 0.42), lw=2.6, label="Moderate spillover"),
        plt.Line2D([0], [0], color=(0.90, 0.08, 0.08), lw=4.0, label="Strong spillover"),
        plt.Line2D([0], [0], color="#f2cdd3", lw=1.2, label="Geographic connection"),
    ]
    legend_nodes = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#d73027", markeredgecolor="#8b0000", markersize=8, label="Target province"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#FDB347", markeredgecolor="#e29a00", markersize=7, label="Neighbor province"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#f0f0f0", markeredgecolor="#555555", markersize=6, label="Other province"),
    ]
    leg1 = ax.legend(handles=legend_lines, loc="upper left", bbox_to_anchor=(-0.02, 0.92), frameon=True, fontsize=8, title="Spillover Intensity", title_fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=legend_nodes, loc="upper right", bbox_to_anchor=(0.88, 0.92), frameon=True, fontsize=8, title="Nodes", title_fontsize=9)

    out_path = OUTPUT_DIR / f"counterfactual_{target}_plus_{shock_tag}sd_network.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    for target in ["江苏", "广东"]:
        print(draw_network(target, "05"))
