"""CyG-Flow entry: VMamba (vssm_small) + normalizing flow on MVTec AD.

    python main_vmamba.py -cfg configs/vssm_small_vmamba.yaml \\
        --data MVTEC_ROOT -cat all --gpu 0 \\
        --seed 42 --deterministic \\
        --results-csv vmamba_mvtec_summary.csv
"""

import csv
import os

import constants as const
import flow
import vmamba_flow
import main as _main

flow.CyGFlow = vmamba_flow.CyGFlow


def _append_last_row(src_csv, dst_csv):
    if not src_csv or not os.path.isfile(src_csv):
        return
    with open(src_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    _main.append_results_csv(dst_csv, rows[-1])


if __name__ == "__main__":
    args = _main.parse_args()
    _main.set_reproducibility(args.seed, args.deterministic)
    master_results_csv = args.results_csv
    per_cat_dir = os.path.join(_main.PROJECT_ROOT, "_bg_runs", "vmamba_per_category")
    os.makedirs(per_cat_dir, exist_ok=True)

    if args.eval:
        if args.category == "all":
            raise ValueError("eval mode does not support -cat all; run eval per category.")
        _main.evaluate(args)
    else:
        if args.category == "all":
            summary = []
            cfg_tag = os.path.splitext(os.path.basename(args.config))[0]
            for c in const.MVTEC_CATEGORIES:
                print("\n========== Training category: {} ==========".format(c))
                args.category = c
                per_cat_csv = os.path.join(per_cat_dir, "{}_{}.csv".format(cfg_tag, c))
                args.results_csv = per_cat_csv
                i_auroc, p_auroc = _main.train(args)
                _append_last_row(per_cat_csv, master_results_csv)
                summary.append((c, p_auroc, i_auroc))
            print("\nAll categories finished.")
            print("\n==== Summary (best.pt) ====")
            for c, p, i in summary:
                print("{:<12} P-AUROC: {:>8} | I-AUROC: {:>8}".format(c, p, i))
        else:
            cfg_tag = os.path.splitext(os.path.basename(args.config))[0]
            per_cat_csv = os.path.join(per_cat_dir, "{}_{}.csv".format(cfg_tag, args.category))
            args.results_csv = per_cat_csv
            _main.train(args)
            _append_last_row(per_cat_csv, master_results_csv)
