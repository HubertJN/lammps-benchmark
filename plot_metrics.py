from pathlib import Path

from plot_utils import generate_performance_review

# -------------------- Paths / settings --------------------

RUNS_DIR = Path("runs")
JSON_PATH = RUNS_DIR / "metrics_summary.json"
OUT_PDF = RUNS_DIR / "performance_review.pdf"

TOP_N_RUNS = 3          # top N by timesteps/s
MERGE_LT_PCT = 5.0      # merge any pie slice < this % into "Remainder"

MANUAL_TAG = "manual"
if __name__ == "__main__":
    generate_performance_review(
        runs_dir=RUNS_DIR,
        json_path=JSON_PATH,
        out_pdf=OUT_PDF,
        top_n_runs=TOP_N_RUNS,
        merge_lt_pct=MERGE_LT_PCT,
        manual_tag=MANUAL_TAG,
    )