"""
preprocessing.py
────────────────
데이터 로딩, 결측치 처리, Feature Engineering.
유저별 집계 피처(RFM + 할인 민감도 + 앱 사용 시간)와
일별 매출 시계열을 반환한다.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
#  상수
# ─────────────────────────────────────────────────────────────
DATA_PATH = Path(__file__).parents[2] / "data" / "aiml_test_data.csv"

REFERENCE_DATE = pd.Timestamp("2024-04-30")   # 데이터 기준 최종일


# ─────────────────────────────────────────────────────────────
#  1. 원시 데이터 로딩
# ─────────────────────────────────────────────────────────────
def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """CSV를 로드하고 date 컬럼을 datetime으로 변환한다."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


# ─────────────────────────────────────────────────────────────
#  2. 결측치 처리
# ─────────────────────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    결측치 처리 전략:
    - payment_method (결측 2,417건 ≈ 5%):
        유저별로 최빈 결제 수단으로 대체.
        유저 전체가 결측인 경우 전체 최빈값('Card')으로 대체.
        → 결제 수단은 개인 습관에 따라 결정되므로
          유저 내 최빈값이 가장 합리적 추정이다.
    """
    df = df.copy()

    user_mode = (
        df.groupby("user_id")["payment_method"]
        .agg(lambda s: s.mode().iloc[0] if s.notna().any() else None)
    )
    global_mode = df["payment_method"].mode().iloc[0]

    def fill_payment(row):
        if pd.isna(row["payment_method"]):
            mode = user_mode.get(row["user_id"])
            return mode if mode is not None else global_mode
        return row["payment_method"]

    df["payment_method"] = df.apply(fill_payment, axis=1)
    return df


# ─────────────────────────────────────────────────────────────
#  3. Feature Engineering — 유저별 집계
# ─────────────────────────────────────────────────────────────
def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    유저별 행동 피처를 생성한다.

    반환 컬럼:
    - recency_days       : 마지막 구매로부터 기준일까지 경과 일수 (낮을수록 최근)
    - frequency          : 총 구매 횟수
    - total_amount       : 총 결제 금액
    - avg_amount         : 평균 결제 금액
    - discount_usage_rate: 할인 적용 구매 비율 (할인 민감도)
    - avg_discount_rate  : 평균 할인율
    - avg_app_time       : 평균 앱 사용 시간 (분) — 충성도 프록시
    - purchase_span_days : 첫 구매 ~ 마지막 구매 기간 (활동 기간)
    - avg_interval_days  : 평균 구매 간격 (낮을수록 자주 구매)
    - pref_card_ratio    : Card 결제 비율
    - pref_pay_ratio     : Pay 결제 비율
    - pref_transfer_ratio: Transfer 결제 비율
    """
    g = df.groupby("user_id")

    last_purchase  = g["date"].max()
    first_purchase = g["date"].min()
    frequency      = g.size().rename("frequency")
    total_amount   = g["paid_amount"].sum().rename("total_amount")
    avg_amount     = g["paid_amount"].mean().rename("avg_amount")

    discount_usage_rate = (
        g["discount_rate"].apply(lambda s: (s > 0).mean())
        .rename("discount_usage_rate")
    )
    avg_discount_rate = g["discount_rate"].mean().rename("avg_discount_rate")
    avg_app_time      = g["app_time_min"].mean().rename("avg_app_time")

    purchase_span = (last_purchase - first_purchase).dt.days.rename("purchase_span_days")
    recency       = (REFERENCE_DATE - last_purchase).dt.days.rename("recency_days")

    # 평균 구매 간격: 구매 2회 이상인 경우만 계산, 나머지는 중앙값으로 대체
    def mean_interval(dates: pd.Series) -> float:
        sorted_dates = dates.sort_values()
        diffs = sorted_dates.diff().dt.days.dropna()
        return diffs.mean() if len(diffs) > 0 else np.nan

    avg_interval = g["date"].apply(mean_interval).rename("avg_interval_days")
    median_interval = avg_interval.median()
    avg_interval = avg_interval.fillna(median_interval)

    # 결제 수단 비율
    payment_dummies = pd.get_dummies(df["payment_method"], prefix="pmt")
    df_with_dummies = df.copy()
    for col in payment_dummies.columns:
        df_with_dummies[col] = payment_dummies[col]
    pmt_cols = {
        "pmt_Card": "pref_card_ratio",
        "pmt_Pay": "pref_pay_ratio",
        "pmt_Transfer": "pref_transfer_ratio",
    }
    pmt_ratios = {}
    for src, dst in pmt_cols.items():
        if src in df_with_dummies.columns:
            pmt_ratios[dst] = df_with_dummies.groupby("user_id")[src].mean()

    features = pd.concat(
        [recency, frequency, total_amount, avg_amount,
         discount_usage_rate, avg_discount_rate, avg_app_time,
         purchase_span, avg_interval,
         *pmt_ratios.values()],
        axis=1,
    )
    features.columns = (
        ["recency_days", "frequency", "total_amount", "avg_amount",
         "discount_usage_rate", "avg_discount_rate", "avg_app_time",
         "purchase_span_days", "avg_interval_days"]
        + list(pmt_ratios.keys())
    )
    return features.reset_index()


# ─────────────────────────────────────────────────────────────
#  4. 일별 매출 시계열 생성
# ─────────────────────────────────────────────────────────────
def build_daily_sales(df: pd.DataFrame) -> pd.DataFrame:
    """
    일별 총 paid_amount를 집계하여 시계열 DataFrame을 반환한다.

    반환 컬럼:
    - ds           : 날짜 (Prophet 규약에 맞춰 'ds'로 명명)
    - y            : 일별 총 매출 (paid_amount 합계)
    - daily_orders : 일별 거래 건수
    """
    daily = (
        df.groupby(df["date"].dt.normalize())
        .agg(y=("paid_amount", "sum"), daily_orders=("paid_amount", "count"))
        .reset_index()
        .rename(columns={"date": "ds"})
    )
    # 날짜 연속성 보장 — 누락된 날짜에 0 채우기
    full_range = pd.date_range(daily["ds"].min(), daily["ds"].max(), freq="D")
    daily = daily.set_index("ds").reindex(full_range).fillna(0).reset_index()
    daily.columns = ["ds", "y", "daily_orders"]
    return daily


# ─────────────────────────────────────────────────────────────
#  5. Train / Test 분리
# ─────────────────────────────────────────────────────────────
def split_train_test(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train: 2023-01-01 ~ 2024-03-31
    Test : 2024-04-01 ~ 2024-04-30
    """
    train = daily[daily["ds"] <= "2024-03-31"].copy()
    test  = daily[daily["ds"] >= "2024-04-01"].copy()
    return train, test


# ─────────────────────────────────────────────────────────────
#  편의 함수: 전체 파이프라인 한 번에 실행
# ─────────────────────────────────────────────────────────────
def run(path: Path = DATA_PATH) -> dict:
    """
    전처리 전체 파이프라인을 실행하고 결과 딕셔너리를 반환한다.

    반환 키:
    - raw          : 원시 DataFrame
    - clean        : 결측치 처리된 DataFrame
    - user_features: 유저별 피처 DataFrame
    - daily_sales  : 일별 매출 DataFrame
    - train        : 학습용 시계열
    - test         : 평가용 시계열
    """
    print("[Preprocessing] 데이터 로딩 중...")
    raw   = load_raw(path)
    clean = handle_missing(raw)

    print("[Preprocessing] 유저 피처 생성 중...")
    user_features = build_user_features(clean)

    print("[Preprocessing] 일별 매출 집계 중...")
    daily_sales = build_daily_sales(clean)
    train, test = split_train_test(daily_sales)

    print(f"[Preprocessing] 완료 — 유저 {len(user_features)}명 | "
          f"Train {len(train)}일 | Test {len(test)}일")
    return {
        "raw": raw,
        "clean": clean,
        "user_features": user_features,
        "daily_sales": daily_sales,
        "train": train,
        "test": test,
    }
