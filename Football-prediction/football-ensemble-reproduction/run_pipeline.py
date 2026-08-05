#!/usr/bin/env python3
"""
Full reproduction pipeline.

Runs, in order:
  1. EDA figures                     (Figs. 1, 2, 4)
  2. Three-layer ensemble training   (Algorithm 1)
  3. Test-set evaluation             (Table VI, Fig. 5)
  4. Ablation study                  (Table V)
  5. 5-fold cross-validation         (Table III)
  6. Statistical significance tests  (Table IV)

Usage
-----
    python run_pipeline.py                 # everything
    python run_pipeline.py --skip-cv       # skip steps 5-6 (the slow ones)
    python run_pipeline.py --quick         # 3 epochs, smoke test only
    python run_pipeline.py --seed 7
"""

from __future__ import annotations

import argparse
import time

from src import ablation, config, cross_validation, data_loader, evaluate
from src import statistical_tests, train
from src.utils import plots
from src.utils.seed import set_global_seed


def banner(text: str) -> None:
    print("\n" + "#" * 78)
    print(f"# {text}")
    print("#" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-layer football ensemble")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip cross-validation and significance tests")
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip figure generation")
    parser.add_argument("--quick", action="store_true",
                        help="3 epochs instead of 30 -- smoke test, not a reproduction")
    parser.add_argument("--no-save", action="store_true",
                        help="Do not persist models to outputs/models/")
    args = parser.parse_args()

    if args.quick:
        config.BILSTM_PARAMS["epochs"] = 3
        config.CNN_PARAMS["epochs"] = 3
        print("QUICK MODE: 3 epochs. Results are NOT a reproduction of the paper.")

    set_global_seed(args.seed)
    t0 = time.time()

    # ---------------------------------------------------------------- Step 1
    banner("STEP 1  Loading data and exploratory analysis")
    df, feature_columns = data_loader.load_dataset()

    if not args.skip_eda:
        labels = df[config.TARGET_COLUMN]
        plots.plot_class_distribution_bar(labels)
        plots.plot_class_distribution_pie(labels)
        plots.plot_correlation_heatmap(df, feature_columns)
        print(f"EDA figures written to {config.FIGURES_DIR}")

    # ---------------------------------------------------------------- Step 2
    banner("STEP 2  Training the three-layer ensemble (Algorithm 1)")
    data = train.prepare_data(df, feature_columns)
    artifacts = train.train_ensemble(data, seed=args.seed)

    if not args.no_save:
        train.save_models(artifacts)

    # ---------------------------------------------------------------- Step 3
    banner("STEP 3  Test-set evaluation (Table VI, Fig. 5)")
    evaluation = evaluate.evaluate_ensemble(artifacts)
    evaluate.error_analysis(evaluation)
    evaluate.save_evaluation(evaluation)

    # ---------------------------------------------------------------- Step 4
    banner("STEP 4  Ablation study (Table V)")
    ablation_df = ablation.run_ablation(artifacts)
    ablation.save_ablation(ablation_df)

    # ------------------------------------------------------------- Steps 5-6
    if not args.skip_cv:
        banner(f"STEP 5  {config.CV_FOLDS}-fold stratified cross-validation (Table III)")
        summary, per_fold = cross_validation.run_cross_validation(df, feature_columns,
                                                                  seed=args.seed)
        cross_validation.save_cross_validation(summary, per_fold)

        banner("STEP 6  Statistical significance testing (Table IV)")
        tests = statistical_tests.paired_tests(per_fold)
        statistical_tests.save_statistical_tests(tests)
    else:
        print("\nSkipping cross-validation and significance tests (--skip-cv).")

    banner(f"PIPELINE COMPLETE in {(time.time() - t0) / 60:.1f} min")
    print(f"Metrics : {config.METRICS_DIR}")
    print(f"Figures : {config.FIGURES_DIR}")
    print(f"Models  : {config.MODELS_DIR}")


if __name__ == "__main__":
    main()
