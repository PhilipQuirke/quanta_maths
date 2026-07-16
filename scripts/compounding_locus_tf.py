"""Battery TF — temporal finalization (eager vs lazy) of the carry, token-time.

Thin CLI wrapper over the reusable library module
``quanta_maths.maths_temporal_finalization`` (the canonical, cross-model
implementation). Closes the token-time question left open by the layer-localization
result (CE24): at which TOKEN POSITION does the carry finalize — eagerly in-place at
its make-carry token, or lazily at the answer region?

Run:
    PYTHONPATH=. python3 scripts/compounding_locus_tf.py all [--fast]
    PYTHONPATH=. python3 scripts/compounding_locus_tf.py add_d10_l2_h3_t40K_s572091
"""
from __future__ import annotations
import json, os, sys

from quanta_maths.maths_temporal_finalization import run_temporal_finalization

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-compounding-locus-tf")
os.makedirs(RESULT_DIR, exist_ok=True)
DEFAULT_MODELS = ["add_d6_l2_h3_t20K_s173289", "add_d5_l2_h3_t15K_s372001"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    fast = "--fast" in sys.argv
    if not args or args == ["all"]:
        models = DEFAULT_MODELS
    else:
        models = args
    n_q = 300 if fast else 600
    results = {}
    for mn in models:
        print(f"=== {mn} (n_q={n_q}) ===", flush=True)
        r = run_temporal_finalization(mn, n_q=n_q)
        results[mn] = r
        v = r["verdict"]
        print(f"  propagated onset per layer: {r['propagated_onset']}", flush=True)
        print(f"  full-input pos {v['full_input_pos']} | = {v['equals_pos']} | onset pos "
              f"{v['propagated_onset_pos']} layer {v['propagated_onset_layer']} | deferral "
              f"{v['genuine_deferral_tokens']} token(s)", flush=True)
        print(f"  local make-carry @ own token, layer0: "
              f"{ {k: round(x, 2) for k, x in v['local_makecarry_decode_at_token_layer0'].items()} }", flush=True)
        print(f"  VERDICT: {v['read']}", flush=True)
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR, flush=True)


if __name__ == "__main__":
    main()
