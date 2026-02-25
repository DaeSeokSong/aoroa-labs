"""
main.py
───────
AOROA LABS 채용 과제 문제 2 — 전체 파이프라인 진입점.

실행 방법:
    cd aoroa_labs
    python problem2/main.py

실행 순서:
    1. 전처리  (preprocessing)
    2. EDA     (eda)
    3. 고객 세분화 (segmentation)
    4. 매출 예측   (forecasting)

산출물:
    problem2/outputs/eda/          — EDA 시각화 7장
    problem2/outputs/segmentation/ — 세분화 시각화 5장
    problem2/outputs/forecasting/  — 예측 시각화 3장
"""

import sys
from pathlib import Path

# 패키지 루트를 sys.path에 추가 (aoroa_labs/ 하위 어디서 실행해도 동작)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from problem2.src import preprocessing, eda, segmentation, forecasting


def main() -> None:
    print("=" * 60)
    print("  AOROA LABS — 문제 2 파이프라인 시작")
    print("=" * 60)

    # ── 1. 전처리 ────────────────────────────────────────────
    data = preprocessing.run()

    # ── 2. EDA ───────────────────────────────────────────────
    eda.run(data)

    # ── 3. 고객 세분화 ────────────────────────────────────────
    seg_result = segmentation.run(data)

    # ── 4. 매출 예측 ──────────────────────────────────────────
    fc_result = forecasting.run(data)

    # ── 최종 요약 출력 ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  파이프라인 완료 요약")
    print("=" * 60)

    print("\n▶ 고객 세분화")
    print(f"  채택 모델   : {seg_result['winner']}"
          f" (K={seg_result['km_k'] if seg_result['winner'] == 'K-Means' else seg_result['gmm_k']})")
    for cid, persona in seg_result["persona_map"].items():
        n = (seg_result["user_features_labeled"]["cluster"] == cid).sum()
        print(f"  Cluster {cid}   : {persona} ({n}명)")

    print("\n▶ 매출 예측 (Test: 2024년 4월)")
    for model, m in fc_result["metrics"].items():
        print(f"  {model:<10} MAPE={m['mape']:.2f}%  "
              f"RMSE={m['rmse']:,.0f}  MAE={m['mae']:,.0f}")

    print("\n▶ 산출물 위치")
    outputs = ROOT / "problem2" / "outputs"
    for sub in ["eda", "segmentation", "forecasting"]:
        files = list((outputs / sub).glob("*.png"))
        print(f"  {sub:>14}/ — {len(files)}개 이미지")

    print("\n" + "=" * 60)
    print("  완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
