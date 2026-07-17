"""Generate + (optionally) upload the d8 mixed model's verified map via the
standard headless-QMAnalyse pipeline (quanta_maths.maths_analysis) + the
registered techniques, round-trip verified, uploaded to the analysis repo.

Mirrors maths_hf_update.update_model's DISCOVERY path but with PCA tri-case
tagging DISABLED (``include_pca=False``): on the d8 model the PCA `.SP/.MP` step
dominated runtime (>30 min) and is the least-critical, SV-irrelevant part of the
map. The core map (Fail%/Impact/Math.Add|Sub|Neg/Attn behaviours + Algo role tags)
and all technique tags (STC/MTC/NTC combiners, delivery route, LINXFER, carry TF)
are still produced.

Modes:
  (default)        generate + save + round-trip verify locally (NO upload)
  --upload-only    upload the already-generated local files (no recompute)
  --execute        generate + verify + upload in one pass

Run: PYTHONPATH=. python scripts/run_d8_map.py [--execute | --upload-only]
"""
from __future__ import annotations

import argparse
import json
import os
import time

from quanta_maths.maths_hf_update import (
    techniques_for, _run_techniques, _verify_roundtrip,
    BEHAVIORS_FILE, FEATURES_FILE, _SAVE_MAJOR)
from quanta_maths.maths_model_loader import load_maths_model_from_analysis_repo, analysis_repo_id

MODEL = "ins1_mix_d8_l3_h4_t70K_s572091"
OUT = f"results/hf-update/{MODEL}/updated"
MANIFEST = "results/hf-update/d8_run_manifest.json"


def _upload(paths):
    from huggingface_hub import HfApi
    api = HfApi()
    for fname, path in paths.items():
        api.upload_file(path_or_fileobj=path, path_in_repo=fname,
                        repo_id=analysis_repo_id(MODEL),
                        commit_message=f"Add {fname} (d8 mixed map: QMAnalyse discovery "
                                       f"[no-PCA] + quanta_maths techniques)")
    return True


def generate():
    model, cfg = load_maths_model_from_analysis_repo(MODEL, device="cpu")
    from quanta_maths.maths_analysis import discover_behaviors, discover_features
    os.makedirs(OUT, exist_ok=True)
    applicable = techniques_for(cfg)
    tech_results = {}

    # behaviours (no PCA) -> behaviour techniques -> save behaviors.json (Algo-free)
    discover_behaviors(cfg, model, include_pca=False)
    _run_techniques(model, cfg, cfg.useful_nodes, BEHAVIORS_FILE, applicable, True, tech_results)
    bpath = os.path.join(OUT, BEHAVIORS_FILE)
    cfg.useful_nodes.save_nodes(bpath, _SAVE_MAJOR[BEHAVIORS_FILE])

    # features (Algo) -> feature techniques (combiners) -> save features.json
    discover_features(cfg)
    _run_techniques(model, cfg, cfg.useful_nodes, FEATURES_FILE, applicable, True, tech_results)
    fpath = os.path.join(OUT, FEATURES_FILE)
    cfg.useful_nodes.save_nodes(fpath, _SAVE_MAJOR[FEATURES_FILE])

    paths = {BEHAVIORS_FILE: bpath, FEATURES_FILE: fpath}
    ok = all(_verify_roundtrip(p, _SAVE_MAJOR[f]) for f, p in paths.items())
    n_nodes = len(cfg.useful_nodes.nodes)
    return paths, ok, tech_results, n_nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="generate + upload")
    ap.add_argument("--upload-only", action="store_true", help="upload existing local files")
    args = ap.parse_args()
    t0 = time.time()

    if args.upload_only:
        paths = {f: os.path.join(OUT, f) for f in (BEHAVIORS_FILE, FEATURES_FILE)}
        assert all(os.path.exists(p) for p in paths.values()), "generate first"
        ok = all(_verify_roundtrip(p, _SAVE_MAJOR[f]) for f, p in paths.items())
        uploaded = _upload(paths) if ok else False
        manifest = {"mode": "upload-only", "roundtrip_ok": ok, "uploaded": uploaded, "paths": paths}
    else:
        paths, ok, tech_results, n_nodes = generate()
        uploaded = _upload(paths) if (args.execute and ok) else False
        manifest = {"mode": "execute" if args.execute else "dry", "roundtrip_ok": ok,
                    "uploaded": uploaded, "n_nodes": n_nodes,
                    "techniques": {k: v.get("tags_added") for k, v in tech_results.items()},
                    "paths": paths}
    manifest["_seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(manifest, open(MANIFEST, "w"), indent=2, default=str)
    print("MANIFEST:", json.dumps({k: manifest[k] for k in manifest if k != "paths"}))


if __name__ == "__main__":
    main()
