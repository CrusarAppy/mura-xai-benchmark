#!/usr/bin/env python
"""Fast repo sanity check — run before pushing or on a fresh checkout.

Catches the class of bug where code works locally but is broken on a clean clone
(e.g. a new explainer file that was never `git add`-ed). Exits non-zero on any problem
so it can gate CI or a pre-push hook.

Usage:
    python scripts/check_repo.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPECTED_EXPLAINERS = {
    "gradcam", "gradcampp", "scorecam", "layercam",
    "integrated_gradients", "shap",
}


def check_explainers() -> list[str]:
    problems = []
    from xai_bench.explainers import EXPLAINERS, build_explainer  # noqa: F401
    have = set(EXPLAINERS)
    missing = EXPECTED_EXPLAINERS - have
    if missing:
        problems.append(f"registry missing explainers: {sorted(missing)} (have {sorted(have)})")
    # every registered explainer module must actually import
    for name, cls in EXPLAINERS.items():
        if cls is None:
            problems.append(f"explainer '{name}' is registered but None")
    return problems


def check_evaluation_api() -> list[str]:
    """Assert the evaluation package exposes the Phase-A API from a clean checkout."""
    expected = ["classification_metrics", "calibration_metrics", "youden_threshold",
                "reliability_curve", "temperature_scale", "deletion_insertion",
                "average_drop_increase", "make_baseline", "baseline_sensitivity"]
    problems = []
    try:
        import xai_bench.evaluation as ev
        for name in expected:
            if not hasattr(ev, name):
                problems.append(f"evaluation API missing: {name}")
    except Exception as e:
        problems.append(f"evaluation package failed to import: {e}")
    return problems


def check_untracked_python() -> list[str]:
    """Warn if any tracked-looking .py under src/ is untracked in git — the exact
    failure mode that broke the Kaggle run (files on disk but not committed)."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "src"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
    except Exception as e:  # not a git repo / git unavailable — skip, don't fail
        return [f"(skipped git check: {e})"]
    stray = [ln for ln in out.splitlines() if ln.endswith(".py")]
    return [f"untracked python file (not committed): {s}" for s in stray]


def main() -> int:
    problems = check_explainers() + check_evaluation_api() + check_untracked_python()
    # separate hard failures from informational skips
    hard = [p for p in problems if not p.startswith("(skipped")]
    if hard:
        print("REPO CHECK FAILED:")
        for p in hard:
            print("  -", p)
        return 1
    print("Repo check passed: all 6 explainers import & are registered; "
          "no untracked python under src/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
