"""
eda.py
──────
탐색적 데이터 분석(EDA) 시각화.
각 함수는 Figure를 저장하고 저장 경로를 반환한다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # headless 환경 대응
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns

# 한글 폰트 설정
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

OUT_DIR = Path(__file__).parents[1] / "outputs" / "eda"


def _save(fig: plt.Figure, name: str, out_dir: Path = OUT_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
#  1. 결측치 현황
# ─────────────────────────────────────────────────────────────
def plot_missing(df: pd.DataFrame) -> Path:
    """컬럼별 결측치 비율 막대그래프."""
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0]

    fig, ax = plt.subplots(figsize=(6, 3))
    missing.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("결측치 비율 (%)", fontsize=13, fontweight="bold")
    ax.set_ylabel("결측 비율 (%)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return _save(fig, "01_missing.png")


# ─────────────────────────────────────────────────────────────
#  2. paid_amount 분포
# ─────────────────────────────────────────────────────────────
def plot_paid_amount_dist(df: pd.DataFrame) -> Path:
    """paid_amount 히스토그램 + KDE."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 히스토그램
    axes[0].hist(df["paid_amount"], bins=60, color="steelblue",
                 edgecolor="white", alpha=0.8)
    axes[0].axvline(df["paid_amount"].mean(), color="red",
                    linestyle="--", label=f"평균 {df['paid_amount'].mean():,.0f}")
    axes[0].axvline(df["paid_amount"].median(), color="orange",
                    linestyle="--", label=f"중앙값 {df['paid_amount'].median():,.0f}")
    axes[0].set_title("paid_amount 분포", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("결제 금액 (원)")
    axes[0].set_ylabel("빈도")
    axes[0].legend()
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))

    # 박스플롯 (결제 수단별)
    df.boxplot(column="paid_amount", by="payment_method", ax=axes[1],
               patch_artist=True,
               boxprops=dict(facecolor="steelblue", alpha=0.5))
    axes[1].set_title("결제 수단별 paid_amount 분포", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("결제 수단")
    axes[1].set_ylabel("결제 금액 (원)")
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}K"))
    plt.suptitle("")

    fig.tight_layout()
    return _save(fig, "02_paid_amount_dist.png")


# ─────────────────────────────────────────────────────────────
#  3. 일별 매출 추세
# ─────────────────────────────────────────────────────────────
def plot_daily_sales(daily: pd.DataFrame) -> Path:
    """일별 매출 시계열 + 30일 이동 평균."""
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(daily["ds"], daily["y"], color="steelblue",
            alpha=0.5, linewidth=0.8, label="일별 매출")
    rolling = daily["y"].rolling(30, center=True).mean()
    ax.plot(daily["ds"], rolling, color="red",
            linewidth=2, label="30일 이동평균")

    ax.axvline(pd.Timestamp("2024-04-01"), color="gray",
               linestyle="--", linewidth=1.2, label="Test 시작(2024-04-01)")
    ax.set_title("일별 총 매출 추세", fontsize=13, fontweight="bold")
    ax.set_xlabel("날짜")
    ax.set_ylabel("매출 (원)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))
    ax.legend()
    fig.tight_layout()
    return _save(fig, "03_daily_sales_trend.png")


# ─────────────────────────────────────────────────────────────
#  4. 요일·월별 매출 패턴
# ─────────────────────────────────────────────────────────────
def plot_seasonality(daily: pd.DataFrame) -> Path:
    """요일별 / 월별 평균 매출 히트맵."""
    df = daily.copy()
    df["weekday"] = df["ds"].dt.day_name()
    df["month"]   = df["ds"].dt.month

    weekday_order = ["Monday", "Tuesday", "Wednesday",
                     "Thursday", "Friday", "Saturday", "Sunday"]
    weekday_avg = (
        df.groupby("weekday")["y"].mean()
        .reindex(weekday_order)
    )
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"]

    month_avg = df.groupby("month")["y"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # 요일별
    bars = axes[0].bar(weekday_kr, weekday_avg.values,
                       color=sns.color_palette("Blues_d", 7))
    axes[0].set_title("요일별 평균 매출", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("평균 매출 (원)")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.2f}M"))
    for bar in bars:
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.01,
                     f"{bar.get_height()/1_000_000:.2f}M",
                     ha="center", va="bottom", fontsize=9)

    # 월별
    bars2 = axes[1].bar(month_avg.index, month_avg.values,
                        color=sns.color_palette("Oranges_d", 12))
    axes[1].set_title("월별 평균 매출", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("월")
    axes[1].set_ylabel("평균 매출 (원)")
    axes[1].set_xticks(range(1, 13))
    axes[1].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.2f}M"))
    for bar in bars2:
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.01,
                     f"{bar.get_height()/1_000_000:.2f}M",
                     ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return _save(fig, "04_seasonality.png")


# ─────────────────────────────────────────────────────────────
#  5. 할인율 vs 결제 금액
# ─────────────────────────────────────────────────────────────
def plot_discount_effect(df: pd.DataFrame) -> Path:
    """할인율별 평균 paid_amount 및 거래 건수."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    discount_stats = df.groupby("discount_rate").agg(
        avg_amount=("paid_amount", "mean"),
        count=("paid_amount", "size")
    ).reset_index()

    discount_labels = [f"{int(r*100)}%" for r in discount_stats["discount_rate"]]

    axes[0].bar(discount_labels, discount_stats["avg_amount"],
                color=sns.color_palette("Greens_d", len(discount_labels)))
    axes[0].set_title("할인율별 평균 결제 금액", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("할인율")
    axes[0].set_ylabel("평균 결제 금액 (원)")
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for i, v in enumerate(discount_stats["avg_amount"]):
        axes[0].text(i, v * 1.01, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)

    axes[1].bar(discount_labels, discount_stats["count"],
                color=sns.color_palette("Purples_d", len(discount_labels)))
    axes[1].set_title("할인율별 거래 건수", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("할인율")
    axes[1].set_ylabel("거래 건수")
    for i, v in enumerate(discount_stats["count"]):
        axes[1].text(i, v * 1.005, f"{v:,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    return _save(fig, "05_discount_effect.png")


# ─────────────────────────────────────────────────────────────
#  6. 유저별 구매 빈도 분포
# ─────────────────────────────────────────────────────────────
def plot_user_frequency(df: pd.DataFrame) -> Path:
    """유저별 총 구매 횟수 분포."""
    freq = df.groupby("user_id").size()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(freq, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(freq.mean(), color="red", linestyle="--",
               label=f"평균 {freq.mean():.1f}회")
    ax.axvline(freq.median(), color="orange", linestyle="--",
               label=f"중앙값 {freq.median():.0f}회")
    ax.set_title("유저별 구매 빈도 분포", fontsize=13, fontweight="bold")
    ax.set_xlabel("구매 횟수")
    ax.set_ylabel("유저 수")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "06_user_frequency.png")


# ─────────────────────────────────────────────────────────────
#  7. 앱 사용 시간 분포
# ─────────────────────────────────────────────────────────────
def plot_app_time(df: pd.DataFrame) -> Path:
    """app_time_min 분포 (범주형)."""
    counts = df["app_time_min"].value_counts().sort_index()
    labels = [f"{v}분" for v in counts.index]

    fig, ax = plt.subplots(figsize=(6, 4))
    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(counts)),
        startangle=90
    )
    ax.set_title("앱 세션 시간 분포", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "07_app_time_dist.png")


# ─────────────────────────────────────────────────────────────
#  전체 실행
# ─────────────────────────────────────────────────────────────
def run(data: dict) -> list[Path]:
    """EDA 시각화 전체를 실행하고 저장 경로 목록을 반환한다."""
    print("[EDA] 시각화 생성 중...")

    df    = data["clean"]
    daily = data["daily_sales"]

    paths = [
        plot_missing(data["raw"]),
        plot_paid_amount_dist(df),
        plot_daily_sales(daily),
        plot_seasonality(daily),
        plot_discount_effect(df),
        plot_user_frequency(df),
        plot_app_time(df),
    ]

    print(f"[EDA] 완료 — {len(paths)}개 그래프 저장됨")
    for p in paths:
        print(f"  → {p}")
    return paths
