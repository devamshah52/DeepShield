"""
DeepShield V3.2 Class-Weighted Evaluation

Uses the existing V3.2 evaluation logic, but loads the
class-weighted V3.2 BEST checkpoint.

Evaluates:
1. Original untouched V3 test set
2. Original 20-file gTTS blind test

No training.
No production changes.
"""

import sys
from pathlib import Path


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(
    r"C:\DeepShield"
)

BACKEND_DIR = (
    PROJECT_ROOT
    / "backend"
)

V31_ROOT = (
    BACKEND_DIR
    / "ml"
    / "experiments"
    / "v3_1"
)

WEIGHTED_CHECKPOINT = (
    V31_ROOT
    / "checkpoints"
    / "deepshield_v3_2_weighted_best.pth"
)


# ============================================================
# Python path
# ============================================================

if str(BACKEND_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(BACKEND_DIR)
    )


# ============================================================
# Import existing evaluation module
# ============================================================

from ml.experiments.v3_1.training import (
    evaluate_v3_2 as evaluator,
)


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 70)

    print(
        "DeepShield V3.2 "
        "Class-Weighted Evaluation"
    )

    print("=" * 70)

    print()
    print(
        "Checkpoint:"
    )

    print(
        WEIGHTED_CHECKPOINT
    )

    if not WEIGHTED_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Weighted V3.2 checkpoint not found:\n"
            f"{WEIGHTED_CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Make the existing evaluator load the weighted checkpoint
    # --------------------------------------------------------

    evaluator.CHECKPOINT_PATH = (
        WEIGHTED_CHECKPOINT
    )

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    import torch

    if not torch.cuda.is_available():

        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda"
    )

    print()
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: "
        f"{torch.version.cuda}"
    )

    # --------------------------------------------------------
    # Load weighted model
    # --------------------------------------------------------

    model = evaluator.load_model(
        device
    )

    # --------------------------------------------------------
    # Original V3 test
    # --------------------------------------------------------

    test_metrics = (
        evaluator.evaluate_v3_test(
            model,
            device,
        )
    )

    # --------------------------------------------------------
    # gTTS blind test
    # --------------------------------------------------------

    gtts_results = (
        evaluator.evaluate_gtts_blind(
            model,
            device,
        )
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "WEIGHTED V3.2 EVALUATION COMPLETE"
    )

    print("=" * 70)

    print()
    print(
        "Original V3 test:"
    )

    print(
        f"  Accuracy: "
        f"{test_metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Balanced Accuracy: "
        f"{test_metrics['balanced_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Macro F1: "
        f"{test_metrics['macro_f1'] * 100:.2f}%"
    )

    print()
    print(
        "gTTS blind test:"
    )

    print(
        f"  Total: "
        f"{gtts_results['total']}"
    )

    print(
        f"  TTS detected: "
        f"{gtts_results['counts']['TTS']}"
    )

    print(
        f"  Genuine: "
        f"{gtts_results['counts']['Genuine']}"
    )

    print(
        f"  Voice Conversion: "
        f"{gtts_results['counts']['Voice_Conversion']}"
    )

    print(
        f"  TTS detection rate: "
        f"{gtts_results['detection_rate']:.2f}%"
    )

    print()
    print(
        "No training was performed."
    )

    print(
        "No production files were modified."
    )

    print(
        "V3.2 unweighted checkpoint was not modified."
    )

    print(
        "V3.1 checkpoint was not modified."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()