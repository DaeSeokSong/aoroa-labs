"""
forecasting.py
──────────────
일별 총 매출 예측 (Time-Series Forecasting).

파이프라인:
  1. Prophet        — 계절성·추세 모델 (baseline)
  2. XGBoost        — lag feature 기반 ML 모델
  3. 앙상블          — Prophet + XGBoost 가중 평균 (weight는 Train MAE 역수 기반)
  4. 평가            — MAPE, RMSE, MAE
  5. 시각화          — 예측 vs 실제, 잔차 분포
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns

from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

OUT_DIR = Path(__file__).parents[1] / "outputs" / "forecasting"

# XGBoost lag feature 설정
LAG_DAYS     = [1, 2, 3, 7, 14, 21, 28]   # 이전 N일 매출
ROLLING_WINS = [7, 14, 30]                  # 이동 평균 윈도우


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
#  평가 지표
# ─────────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    """MAPE, RMSE, MAE를 계산하고 출력한다."""
    # MAPE: y_true == 0인 경우 제외
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    if label:
        print(f"  [{label}] MAPE={mape:.2f}%  RMSE={rmse:,.0f}  MAE={mae:,.0f}")
    return {"mape": mape, "rmse": rmse, "mae": mae}


# ─────────────────────────────────────────────────────────────
#  1. Prophet 모델
# ─────────────────────────────────────────────────────────────
def run_prophet(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Prophet으로 학습 후 Test 기간 예측값을 반환한다.

    Prophet의 주요 설정:
    - yearly_seasonality=True  : 연간 계절성 (이커머스 연간 패턴)
    - weekly_seasonality=True  : 주간 계절성 (요일별 패턴)
    - daily_seasonality=False  : 일내 시계열 아님
    - changepoint_prior_scale=0.05 : 추세 변화 민감도 (과적합 방지)
    - 한국 공휴일 추가 (holidays 패키지)
    """
    try:
        import holidays as hd
        kr_holidays = hd.country_holidays("KR", years=[2023, 2024])
        holiday_df = pd.DataFrame([
            {"ds": pd.Timestamp(d), "holiday": name}
            for d, name in kr_holidays.items()
        ])
    except Exception:
        holiday_df = None

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays=holiday_df,
    )
    model.fit(train[["ds", "y"]])

    # Train 내 예측 (앙상블 가중치 산출용)
    train_forecast = model.predict(train[["ds"]])
    train_pred = train_forecast["yhat"].values

    # Test 예측
    future      = model.predict(test[["ds"]])
    test_pred   = future["yhat"].clip(lower=0)

    train_metrics = evaluate(train["y"].values, train_pred, label="Prophet(Train)")
    test_metrics  = evaluate(test["y"].values, test_pred.values, label="Prophet(Test)")

    return (
        pd.Series(train_pred, index=train.index, name="prophet"),
        pd.Series(test_pred.values, index=test.index, name="prophet"),
        {"train": train_metrics, "test": test_metrics},
    )


# ─────────────────────────────────────────────────────────────
#  2. XGBoost 모델
# ─────────────────────────────────────────────────────────────
def _build_xgb_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost용 시계열 피처를 생성한다.

    - lag_N       : N일 전 매출 (단기·중기 의존성)
    - roll_mean_N : N일 이동 평균 (추세 스무딩)
    - dayofweek   : 요일 (0=월 ~ 6=일)
    - month       : 월
    - is_weekend  : 주말 여부
    - dayofyear   : 연중 일자 (연간 계절성)
    """
    df = daily[["ds", "y"]].copy()

    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = df["y"].shift(lag)

    for win in ROLLING_WINS:
        df[f"roll_mean_{win}"] = df["y"].shift(1).rolling(win).mean()

    df["dayofweek"] = df["ds"].dt.dayofweek
    df["month"]     = df["ds"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["dayofyear"]  = df["ds"].dt.dayofyear
    df["week"]       = df["ds"].dt.isocalendar().week.astype(int)

    return df


def run_xgboost(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.Series, pd.Series, dict]:
    """XGBoost로 학습 후 Test 기간 예측값을 반환한다."""
    daily = pd.concat([train, test], ignore_index=True)
    df_feat = _build_xgb_features(daily)

    feature_cols = [c for c in df_feat.columns if c not in ["ds", "y"]]

    # lag 최대값(28일)만큼 앞부분 제거
    df_feat = df_feat.dropna()

    n_train = (df_feat["ds"] <= "2024-03-31").sum()
    X_train = df_feat.iloc[:n_train][feature_cols]
    y_train = df_feat.iloc[:n_train]["y"]
    X_test  = df_feat.iloc[n_train:][feature_cols]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False,
    )

    train_pred = np.clip(model.predict(X_train), 0, None)
    test_pred  = np.clip(model.predict(X_test),  0, None)

    # test index를 원래 test DataFrame index에 맞춤
    test_ds = df_feat.iloc[n_train:]["ds"].values
    test_index = test[test["ds"].isin(test_ds)].index

    train_metrics = evaluate(y_train.values, train_pred, label="XGBoost(Train)")
    test_metrics  = evaluate(
        test.loc[test_index, "y"].values, test_pred, label="XGBoost(Test)"
    )

    return (
        pd.Series(train_pred, index=df_feat.iloc[:n_train].index, name="xgboost"),
        pd.Series(test_pred,  index=test_index,                    name="xgboost"),
        {"train": train_metrics, "test": test_metrics},
    )


# ─────────────────────────────────────────────────────────────
#  3. 앙상블: Train MAE 역수 기반 가중 평균
# ─────────────────────────────────────────────────────────────
def ensemble(
    prophet_test: pd.Series,
    xgb_test: pd.Series,
    prophet_metrics: dict,
    xgb_metrics: dict,
    test: pd.DataFrame,
) -> tuple[pd.Series, dict]:
    """
    Prophet과 XGBoost를 Train MAE 역수 가중으로 앙상블한다.
    Train MAE가 낮을수록 해당 모델에 더 높은 가중치를 부여한다.
    """
    w_p = 1.0 / (prophet_metrics["train"]["mae"] + 1e-9)
    w_x = 1.0 / (xgb_metrics["train"]["mae"] + 1e-9)
    w_total = w_p + w_x

    w_p /= w_total
    w_x /= w_total
    print(f"\n  앙상블 가중치 — Prophet: {w_p:.3f} | XGBoost: {w_x:.3f}")

    # 인덱스 정렬 후 가중 평균
    common_idx = prophet_test.index.intersection(xgb_test.index)
    ensemble_pred = (
        w_p * prophet_test.loc[common_idx].values +
        w_x * xgb_test.loc[common_idx].values
    )
    ensemble_series = pd.Series(
        np.clip(ensemble_pred, 0, None),
        index=common_idx,
        name="ensemble"
    )

    metrics = evaluate(
        test.loc[common_idx, "y"].values,
        ensemble_series.values,
        label="Ensemble(Test)"
    )
    return ensemble_series, metrics


# ─────────────────────────────────────────────────────────────
#  4. 시각화
# ─────────────────────────────────────────────────────────────
def plot_forecast(
    train: pd.DataFrame,
    test: pd.DataFrame,
    prophet_pred: pd.Series,
    xgb_pred: pd.Series,
    ensemble_pred: pd.Series,
) -> Path:
    """Test 기간 실제 vs 각 모델 예측 비교 그래프."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9),
                             gridspec_kw={"height_ratios": [2, 1]})

    # 상단: 전체 추세 + Test 구간 확대
    ax = axes[0]
    # Train 마지막 60일만 표시 (맥락용)
    train_tail = train.tail(60)
    ax.plot(train_tail["ds"], train_tail["y"],
            color="gray", alpha=0.5, label="실제(Train 후반)", linewidth=1)
    ax.plot(test["ds"], test["y"],
            color="black", linewidth=2, marker="o", markersize=3, label="실제(Test)")
    ax.plot(test["ds"], prophet_pred.values,
            color="steelblue", linewidth=1.5, linestyle="--", label="Prophet 예측")
    ax.plot(test.loc[xgb_pred.index, "ds"], xgb_pred.values,
            color="darkorange", linewidth=1.5, linestyle="-.", label="XGBoost 예측")
    ax.plot(test.loc[ensemble_pred.index, "ds"], ensemble_pred.values,
            color="crimson", linewidth=2.5, label="앙상블 예측")
    ax.axvline(pd.Timestamp("2024-04-01"), color="gray",
               linestyle=":", linewidth=1)
    ax.set_title("일별 매출 예측 vs 실제 (2024년 4월)", fontsize=13, fontweight="bold")
    ax.set_ylabel("매출 (원)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # 하단: 앙상블 잔차
    ax2 = axes[1]
    common_idx = ensemble_pred.index
    residuals = test.loc[common_idx, "y"].values - ensemble_pred.values
    ax2.bar(test.loc[common_idx, "ds"], residuals,
            color=["crimson" if r < 0 else "steelblue" for r in residuals],
            alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("앙상블 잔차 (실제 - 예측)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("잔차 (원)")
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return _save(fig, "01_forecast_comparison.png")


def plot_metrics_comparison(metrics: dict) -> Path:
    """모델별 MAPE / RMSE 비교 막대그래프."""
    models = list(metrics.keys())
    mapes  = [metrics[m]["mape"] for m in models]
    rmses  = [metrics[m]["rmse"] / 1_000_000 for m in models]   # 단위: 백만원

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    palette = sns.color_palette("Set2", len(models))

    for ax, vals, title, unit in zip(
        axes,
        [mapes, rmses],
        ["MAPE (%)", "RMSE (백만원)"],
        ["%", "M"],
    ):
        bars = ax.bar(models, vals, color=palette)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(title)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{v:.2f}{unit}", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.3)

    fig.suptitle("모델별 예측 성능 비교 (Test: 2024년 4월)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_metrics_comparison.png")


def plot_residual_dist(
    test: pd.DataFrame, ensemble_pred: pd.Series
) -> Path:
    """앙상블 잔차 분포 히스토그램."""
    common_idx = ensemble_pred.index
    residuals = test.loc[common_idx, "y"].values - ensemble_pred.values

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(residuals, bins=15, color="steelblue",
            edgecolor="white", alpha=0.8, density=True)
    sns.kdeplot(residuals, ax=ax, color="crimson", linewidth=2)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("앙상블 잔차 분포", fontsize=13, fontweight="bold")
    ax.set_xlabel("잔차 (원)")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1_000_000:.1f}M"))
    fig.tight_layout()
    return _save(fig, "03_residual_dist.png")


def print_metrics_table(all_metrics: dict) -> None:
    """전체 모델 성능 지표를 표 형식으로 출력한다."""
    print("\n" + "=" * 55)
    print("  매출 예측 성능 지표 (Test: 2024년 4월)")
    print("=" * 55)
    print(f"  {'모델':<12} {'MAPE':>8} {'RMSE':>14} {'MAE':>14}")
    print("-" * 55)
    for model, m in all_metrics.items():
        print(f"  {model:<12} {m['mape']:>7.2f}%  "
              f"{m['rmse']:>12,.0f}  {m['mae']:>12,.0f}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────
#  전체 실행
# ─────────────────────────────────────────────────────────────
def run(data: dict) -> dict:
    """
    시계열 예측 전체 파이프라인을 실행한다.

    반환 딕셔너리:
    - ensemble_pred : 앙상블 예측 Series (Test 기간)
    - prophet_pred  : Prophet 예측 Series
    - xgb_pred      : XGBoost 예측 Series
    - metrics       : 모델별 평가 지표
    """
    train = data["train"]
    test  = data["test"]

    print("\n[Forecasting] Prophet 학습 중...")
    prophet_train, prophet_test, prophet_metrics = run_prophet(train, test)

    print("[Forecasting] XGBoost 학습 중...")
    xgb_train, xgb_test, xgb_metrics = run_xgboost(train, test)

    print("[Forecasting] 앙상블 구성...")
    ensemble_pred, ensemble_metrics = ensemble(
        prophet_test, xgb_test, prophet_metrics, xgb_metrics, test
    )

    all_test_metrics = {
        "Prophet":  prophet_metrics["test"],
        "XGBoost":  xgb_metrics["test"],
        "Ensemble": ensemble_metrics,
    }
    print_metrics_table(all_test_metrics)

    print("\n[Forecasting] 시각화 저장 중...")
    plot_forecast(train, test, prophet_test, xgb_test, ensemble_pred)
    plot_metrics_comparison(all_test_metrics)
    plot_residual_dist(test, ensemble_pred)

    print("[Forecasting] 완료")
    return {
        "ensemble_pred":    ensemble_pred,
        "prophet_pred":     prophet_test,
        "xgb_pred":         xgb_test,
        "metrics":          all_test_metrics,
        "ensemble_metrics": ensemble_metrics,
    }
