"""
segmentation.py
───────────────
고객 세분화 (Customer Segmentation).

파이프라인:
  1. 피처 스케일링 (StandardScaler)
  2. K-Means — 최적 K 탐색 (Elbow + Silhouette)
  3. GMM     — 최적 K 탐색 (BIC)
  4. 두 모델 비교 및 최종 군집 레이블 선택 (Silhouette 기준)
  5. t-SNE 시각화
  6. 군집별 특성 분석 + 비즈니스 페르소나 명명
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

OUT_DIR = Path(__file__).parents[1] / "outputs" / "segmentation"

# 피처 엔지니어링에서 생성한 수치형 컬럼만 사용
FEATURE_COLS = [
    "recency_days",
    "frequency",
    "total_amount",
    "avg_amount",
    "discount_usage_rate",
    "avg_discount_rate",
    "avg_app_time",
    "purchase_span_days",
    "avg_interval_days",
]

# 군집 수 탐색 범위
KM_K_RANGE  = range(2, 9)    # K-Means: 해석 가능한 소규모 군집 탐색
GMM_K_RANGE = range(2, 13)   # GMM: BIC 탐색 범위 확장 (boundary 방지)

# 군집별 비즈니스 페르소나 — run() 내부에서 데이터 기반으로 할당
PERSONA_MAP: dict[int, str] = {}


def _save(fig: plt.Figure, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────
#  1. 스케일링
# ─────────────────────────────────────────────────────────────
def scale_features(user_features: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X = scaler.fit_transform(user_features[FEATURE_COLS])
    return X, scaler


# ─────────────────────────────────────────────────────────────
#  2. K-Means: 최적 K 탐색
# ─────────────────────────────────────────────────────────────
def search_kmeans(X: np.ndarray) -> tuple[pd.DataFrame, int]:
    """
    Elbow (WCSS) + Silhouette Score로 최적 K를 탐색한다.
    두 지표를 종합하여 최적 K를 반환한다.
    """
    results = []
    for k in KM_K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results.append({"k": k, "wcss": km.inertia_, "silhouette": sil})

    df_res = pd.DataFrame(results)

    # Elbow: WCSS 감소율이 꺾이는 지점
    wcss = df_res["wcss"].values
    diffs = np.diff(wcss)
    diffs2 = np.diff(diffs)
    elbow_k = int(df_res["k"].iloc[np.argmax(diffs2) + 1])

    # Silhouette: 최대값
    sil_k = int(df_res.loc[df_res["silhouette"].idxmax(), "k"])

    # 두 기준이 다르면 Silhouette 우선 (해석력 중심), 단 최소 3개 군집
    best_k = sil_k if sil_k >= 3 else max(elbow_k, 3)

    return df_res, best_k


def plot_kmeans_search(df_res: pd.DataFrame, best_k: int) -> Path:
    """Elbow + Silhouette 탐색 결과 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df_res["k"], df_res["wcss"], marker="o", color="steelblue")
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"선택 K={best_k}")
    axes[0].set_title("K-Means Elbow (WCSS)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("K (군집 수)"); axes[0].set_ylabel("WCSS")
    axes[0].legend()

    axes[1].plot(df_res["k"], df_res["silhouette"], marker="s", color="darkorange")
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"선택 K={best_k}")
    axes[1].set_title("K-Means Silhouette Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("K (군집 수)"); axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    fig.suptitle("K-Means 최적 K 탐색", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "01_kmeans_search.png")


# ─────────────────────────────────────────────────────────────
#  3. GMM: 최적 K 탐색 (BIC)
# ─────────────────────────────────────────────────────────────
def search_gmm(X: np.ndarray) -> tuple[pd.DataFrame, int]:
    """BIC(Bayesian Information Criterion)으로 최적 K를 탐색한다."""
    results = []
    for k in GMM_K_RANGE:
        gm = GaussianMixture(n_components=k, random_state=42,
                             covariance_type="full", n_init=3)
        gm.fit(X)
        labels = gm.predict(X)
        sil = silhouette_score(X, labels)
        results.append({"k": k, "bic": gm.bic(X), "silhouette": sil})

    df_res = pd.DataFrame(results)
    best_k = int(df_res.loc[df_res["bic"].idxmin(), "k"])
    best_k = max(best_k, 3)   # 최소 3개 군집 보장
    return df_res, best_k


def plot_gmm_search(df_res: pd.DataFrame, best_k: int) -> Path:
    """GMM BIC 탐색 결과 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df_res["k"], df_res["bic"], marker="o", color="purple")
    axes[0].axvline(best_k, color="red", linestyle="--", label=f"선택 K={best_k}")
    axes[0].set_title("GMM BIC Score", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("K (군집 수)"); axes[0].set_ylabel("BIC")
    axes[0].legend()

    axes[1].plot(df_res["k"], df_res["silhouette"], marker="s", color="darkorange")
    axes[1].axvline(best_k, color="red", linestyle="--", label=f"선택 K={best_k}")
    axes[1].set_title("GMM Silhouette Score", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("K (군집 수)"); axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    fig.suptitle("GMM 최적 K 탐색 (BIC)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_gmm_search.png")


# ─────────────────────────────────────────────────────────────
#  4. 최종 모델 학습 및 비교
# ─────────────────────────────────────────────────────────────
def fit_models(
    X: np.ndarray, km_k: int, gmm_k: int
) -> tuple[KMeans, GaussianMixture, np.ndarray, np.ndarray, str]:
    """
    K-Means와 GMM을 최적 K로 학습하고,
    Silhouette Score가 높은 모델의 레이블을 최종 레이블로 반환한다.
    """
    km = KMeans(n_clusters=km_k, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)
    km_sil = silhouette_score(X, km_labels)

    gm = GaussianMixture(n_components=gmm_k, random_state=42,
                         covariance_type="full", n_init=3)
    gm.fit(X)
    gm_labels = gm.predict(X)
    gm_sil = silhouette_score(X, gm_labels)

    print(f"  K-Means  (K={km_k}): Silhouette={km_sil:.4f}")
    print(f"  GMM      (K={gmm_k}): Silhouette={gm_sil:.4f}")

    if km_sil >= gm_sil:
        winner, final_labels = "K-Means", km_labels
        print(f"  → 최종 선택: K-Means (K={km_k})")
    else:
        winner, final_labels = "GMM", gm_labels
        print(f"  → 최종 선택: GMM (K={gmm_k})")

    return km, gm, km_labels, gm_labels, winner


def plot_model_comparison(
    X: np.ndarray, km_labels: np.ndarray, gm_labels: np.ndarray,
    km_k: int, gmm_k: int
) -> Path:
    """K-Means vs GMM Silhouette 비교 막대그래프."""
    km_sil = silhouette_score(X, km_labels)
    gm_sil = silhouette_score(X, gm_labels)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [f"K-Means\n(K={km_k})", f"GMM\n(K={gmm_k})"],
        [km_sil, gm_sil],
        color=["steelblue", "darkorchid"],
        width=0.5
    )
    for bar, val in zip(bars, [km_sil, gm_sil]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("K-Means vs GMM — Silhouette Score 비교",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Silhouette Score")
    ax.set_ylim(0, max(km_sil, gm_sil) * 1.2)
    fig.tight_layout()
    return _save(fig, "03_model_comparison.png")


# ─────────────────────────────────────────────────────────────
#  5. t-SNE 시각화
# ─────────────────────────────────────────────────────────────
def plot_tsne(
    X: np.ndarray,
    km_labels: np.ndarray,
    gm_labels: np.ndarray,
    persona_map: dict[int, str],
    winner: str,
) -> Path:
    """t-SNE로 군집을 2D 시각화한다."""
    print("  [t-SNE] 차원 축소 중 (시간 소요)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_2d = tsne.fit_transform(X)

    final_labels = km_labels if winner == "K-Means" else gm_labels
    n_clusters = len(np.unique(final_labels))
    palette = sns.color_palette("tab10", n_clusters)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, labels, title in zip(
        axes,
        [km_labels, gm_labels],
        ["K-Means", "GMM"]
    ):
        n_k = len(np.unique(labels))
        colors = sns.color_palette("tab10", n_k)
        for k in range(n_k):
            mask = labels == k
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                       c=[colors[k]], s=15, alpha=0.6, label=f"Cluster {k}")
        ax.set_title(f"t-SNE — {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
        ax.legend(markerscale=2, fontsize=9)

    # 최종 선택 모델에 페르소나 이름 오버레이
    final_ax = axes[0] if winner == "K-Means" else axes[1]
    for k, persona in persona_map.items():
        mask = final_labels == k
        cx = X_2d[mask, 0].mean()
        cy = X_2d[mask, 1].mean()
        final_ax.annotate(persona, (cx, cy),
                          fontsize=9, fontweight="bold",
                          ha="center",
                          bbox=dict(boxstyle="round,pad=0.3",
                                    fc="white", alpha=0.7))

    fig.suptitle(f"t-SNE 군집 시각화 (최종 선택: {winner})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "04_tsne.png")


# ─────────────────────────────────────────────────────────────
#  6. 군집별 특성 분석 및 페르소나 명명
# ─────────────────────────────────────────────────────────────
def analyze_clusters(
    user_features: pd.DataFrame, labels: np.ndarray
) -> tuple[pd.DataFrame, dict[int, str]]:
    """
    군집별 피처 평균을 분석하고 비즈니스 페르소나를 자동으로 명명한다.

    명명 기준:
    - recency_days 낮음 + frequency 높음 + total_amount 높음      → VIP 고객
    - discount_usage_rate 높음 + avg_discount_rate 높음 + avg_amount 낮음 → 체리피커
    - avg_app_time 높음 + purchase_span_days 높음 + avg_amount 낮음 → 일반 고객
    - recency_days 높음 + frequency 낮음                           → 이탈 위험
    - 나머지                                                        → 기타 고객

    각 페르소나 점수는 사용하는 피처 수로 정규화하여 공정한 비교를 보장한다.
    군집 배정은 점수 내림차순 greedy 방식으로 중복을 방지한다.
    """
    df = user_features.copy()
    df["cluster"] = labels
    profile = df.groupby("cluster")[FEATURE_COLS].mean()

    # 각 피처의 순위 (1=최소, n_clusters=최대)
    ranks = profile.rank(ascending=True)
    n = len(profile)

    # 피처 수로 정규화된 페르소나 점수 계산
    all_scores: dict[int, dict[str, float]] = {}
    for cluster_id in profile.index:
        r = ranks.loc[cluster_id]
        all_scores[cluster_id] = {
            "VIP 고객":    ((n - r["recency_days"]) + r["frequency"] + r["total_amount"]) / 3,
            "체리피커":    (r["discount_usage_rate"] + r["avg_discount_rate"] + (n - r["avg_amount"])) / 3,
            "일반 고객":   (r["avg_app_time"] + r["purchase_span_days"] + (n - r["avg_amount"])) / 3,
            "이탈 위험":   (r["recency_days"] + (n - r["frequency"])) / 2,
        }

    # 점수 내림차순 정렬 → greedy 배정 (최고 점수 군집이 페르소나 선점)
    candidates = sorted(
        [
            (max(s.values()), cid, max(s, key=s.get))
            for cid, s in all_scores.items()
        ],
        reverse=True,
    )
    persona_map: dict[int, str] = {}
    used_personas: set[str] = set()
    for _, cluster_id, best_persona in candidates:
        if best_persona not in used_personas:
            persona_map[cluster_id] = best_persona
            used_personas.add(best_persona)
        else:
            persona_map[cluster_id] = "기타 고객"

    return profile, persona_map


def plot_cluster_profile(
    profile: pd.DataFrame, persona_map: dict[int, str]
) -> Path:
    """군집별 피처 레이더 차트 (정규화 후 시각화)."""
    # 0~1 정규화
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    n_clusters = len(profile)
    cols = list(norm.columns)
    n_cols = len(cols)
    angles = np.linspace(0, 2 * np.pi, n_cols, endpoint=False).tolist()
    angles += angles[:1]

    palette = sns.color_palette("tab10", n_clusters)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, (cluster_id, row) in enumerate(norm.iterrows()):
        values = row.tolist() + [row.iloc[0]]
        persona = persona_map.get(cluster_id, f"Cluster {cluster_id}")
        ax.plot(angles, values, "o-", linewidth=2,
                color=palette[i], label=f"C{cluster_id}: {persona}")
        ax.fill(angles, values, alpha=0.1, color=palette[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cols, fontsize=9)
    ax.set_title("군집별 피처 프로파일 (정규화)", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    fig.tight_layout()
    return _save(fig, "05_cluster_profile.png")


def print_cluster_summary(
    user_features: pd.DataFrame, labels: np.ndarray,
    profile: pd.DataFrame, persona_map: dict[int, str]
) -> None:
    """군집별 요약 정보를 출력한다."""
    df = user_features.copy()
    df["cluster"] = labels
    counts = df["cluster"].value_counts().sort_index()

    print("\n" + "=" * 60)
    print("  군집 세분화 결과 요약")
    print("=" * 60)
    for cid in profile.index:
        persona = persona_map.get(cid, f"Cluster {cid}")
        n = counts.get(cid, 0)
        pct = n / len(df) * 100
        row = profile.loc[cid]
        print(f"\n  [ Cluster {cid} ] {persona}  ({n}명, {pct:.1f}%)")
        print(f"    최근성(days)    : {row['recency_days']:.1f}")
        print(f"    구매 빈도       : {row['frequency']:.1f}회")
        print(f"    총 결제 금액    : {row['total_amount']:,.0f}원")
        print(f"    평균 결제 금액  : {row['avg_amount']:,.0f}원")
        print(f"    할인 사용 비율  : {row['discount_usage_rate']*100:.1f}%")
        print(f"    평균 앱 시간    : {row['avg_app_time']:.1f}분")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
#  전체 실행
# ─────────────────────────────────────────────────────────────
def run(data: dict) -> dict:
    """
    고객 세분화 전체 파이프라인을 실행한다.

    반환 딕셔너리:
    - user_features_labeled : 군집 레이블이 추가된 유저 피처 DataFrame
    - final_labels          : 최종 군집 레이블 배열
    - persona_map           : {cluster_id: 페르소나명}
    - profile               : 군집별 피처 평균 DataFrame
    - winner                : 선택된 모델명 ('K-Means' or 'GMM')
    """
    print("\n[Segmentation] 고객 세분화 시작...")

    user_features = data["user_features"]
    X, scaler = scale_features(user_features)

    print("  K-Means 최적 K 탐색...")
    km_search, km_k = search_kmeans(X)
    plot_kmeans_search(km_search, km_k)

    print(f"  GMM 최적 K 탐색...")
    gm_search, gm_k = search_gmm(X)
    plot_gmm_search(gm_search, gm_k)

    print(f"  모델 학습 (K-Means K={km_k}, GMM K={gm_k})...")
    km, gm, km_labels, gm_labels, winner = fit_models(X, km_k, gm_k)
    plot_model_comparison(X, km_labels, gm_labels, km_k, gm_k)

    final_labels = km_labels if winner == "K-Means" else gm_labels

    print("  군집 특성 분석...")
    profile, persona_map = analyze_clusters(user_features, final_labels)
    print_cluster_summary(user_features, final_labels, profile, persona_map)

    plot_cluster_profile(profile, persona_map)
    plot_tsne(X, km_labels, gm_labels, persona_map, winner)

    user_features_labeled = user_features.copy()
    user_features_labeled["cluster"] = final_labels
    user_features_labeled["persona"] = [
        persona_map.get(l, f"Cluster {l}") for l in final_labels
    ]

    print(f"\n[Segmentation] 완료 — {len(profile)}개 군집, 최종 모델: {winner}")
    return {
        "user_features_labeled": user_features_labeled,
        "final_labels": final_labels,
        "persona_map": persona_map,
        "profile": profile,
        "winner": winner,
        "km_k": km_k,
        "gmm_k": gm_k,
    }
