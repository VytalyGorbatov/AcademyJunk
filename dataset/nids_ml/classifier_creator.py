import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger(__name__)

try:
    from nids_ml.pipelines import ClassifierPipeline, TwoWayPipeline
    from nids_ml.utils import DataUtils, set_global_seed
except ModuleNotFoundError:
    # Running as a plain script: add the parent of this package to sys.path
    # so that `nids_ml` becomes importable as a package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nids_ml.pipelines import ClassifierPipeline, TwoWayPipeline
    from nids_ml.utils import DataUtils, set_global_seed


def _load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train classifier")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--device", help="Override device (cpu, cuda, mps)")
    parser.add_argument("--epochs", type=int, help="Override epochs")
    parser.add_argument("--dry_run", action="store_true", help="Run single batch only")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrain_epochN.pt checkpoint (skips Stage 1)")
    parser.add_argument("--calibrate", type=str, default=None,
                        help="Path to model_best.pt — run calibration only (no training)")
    args = parser.parse_args()

    config_path = Path(args.config)
    try:
        config = _load_config(config_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs

    set_global_seed(int(config.get("seed", 42)))
    device = DataUtils.resolve_device(args.device)

    stop_requested = {"value": False}

    def _handle_signal(signum, _frame) -> None:
        if not stop_requested["value"]:
            logger.warning("Signal %s received. Stopping after current step...", signum)
        stop_requested["value"] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    model_type = config.get("model", {}).get("type", "cnn").lower()
    logger.info("Model type: %s | Device: %s", model_type, device)
    stop_fn = lambda: stop_requested["value"]

    if model_type == "tcn_2way":
        pipeline = TwoWayPipeline(config=config, device=device)
        if args.calibrate:
            pipeline.calibrate(checkpoint_path=args.calibrate)
        else:
            pipeline.run(stop_flag=stop_fn, pretrained_path=args.pretrained)
    else:
        pipeline = ClassifierPipeline(config=config, device=device)
        pipeline.run(
            epochs_override=args.epochs,
            dry_run=args.dry_run,
            stop_flag=stop_fn,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
