"""Framework to refresh the per-model analysis JSONs on HuggingFace.

For each ``PhilipQuirke/QuantaMaths_<name>`` repo this framework:
  1. loads the model (``model.pth`` + ``training_loss.json``),
  2. downloads the current ``behaviors.json`` + ``features.json`` node lists,
  3. runs every registered *technique* that APPLIES to the model, adding inline
     tags (idempotently),
  4. writes the updated JSON locally (with a backup of the originals), and
  5. uploads the revised JSON back to the repo -- UNLESS in dry-run mode.

Techniques are library functions wrapped as :class:`Technique` descriptors in the
``TECHNIQUES`` registry below. The "addition" and "mixed" investigation threads add
their reusable techniques to ``quanta_maths`` and register them here; the runner
then executes "all sensible techniques" per model automatically.

Two JSON files are kept SEPARATE on purpose:
  * ``behaviors.json`` -- quantitative behavior tags (``Fail%``, ``Impact``,
    ``Math.Add``, ``Attn``, and new ``Probe:`` etc.). Saved with ALL tags.
  * ``features.json``  -- algorithmic role tags (``Algo:...``). Saved ALGO-only.
Merging them would inject ``Algo`` tags into ``behaviors.json`` (which historically
has none), so each technique declares which file it targets.

Dry-run is the DEFAULT. Nothing is uploaded unless ``dry_run=False``.
"""
from __future__ import annotations

import filecmp
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from quanta_maths.maths_model_loader import (
    analysis_repo_id, load_maths_model_from_analysis_repo, ANALYSIS_REPO_PREFIX)

BEHAVIORS_FILE = "behaviors.json"
FEATURES_FILE = "features.json"
MECHANISM_FILE = "mechanism.md"  # auto-generated per-model diagram doc (HF, not git)

# Where features.json (ALGO-only) vs behaviors.json (all tags) are filtered on save.
_SAVE_MAJOR = {FEATURES_FILE: "Algo", BEHAVIORS_FILE: ""}


# ===========================================================================
# Technique descriptor + registry
# ===========================================================================

@dataclass
class Technique:
    """A reusable analysis technique that adds inline tags to a model's nodes.

    Fields:
      name        : short identifier (used in the run manifest).
      target      : which file it writes -- BEHAVIORS_FILE or FEATURES_FILE.
      applies_to  : predicate ``cfg -> bool`` (e.g. only addition-capable models).
      owns_tag    : predicate ``tag_str -> bool`` marking tags this technique owns;
                    they are cleared before each run so re-runs are IDEMPOTENT
                    (no duplicates, stale values replaced).
      run         : ``(model, cfg, nodes) -> int`` adds tags to ``nodes`` (a
                    UsefulNodeList) and returns the number of tags added.
      description : human-readable note.
    """
    name: str
    target: str
    applies_to: Callable
    owns_tag: Callable
    run: Callable
    description: str = ""

    def clear_owned(self, nodes) -> int:
        removed = 0
        for node in nodes.nodes:
            keep = [t for t in node.tags if not self.owns_tag(t)]
            removed += len(node.tags) - len(keep)
            node.tags = keep
        return removed

    def is_present(self, nodes) -> bool:
        """True if this technique's owned tags already exist in ``nodes``."""
        return any(self.owns_tag(t) for node in nodes.nodes for t in node.tags)


# --- built-in techniques (wrapping the existing library taggers) -----------

def _run_combiner(operation):
    def _fn(model, cfg, nodes):
        from quanta_maths.maths_batch import tag_stc_nodes
        return tag_stc_nodes(model, cfg, nodes, operation=operation)
    return _fn


def _run_combiner_cls(cls):
    def _fn(model, cfg, nodes):
        from quanta_maths.maths_batch import tag_stc_nodes
        return tag_stc_nodes(model, cfg, nodes, cls=cls)
    return _fn


def _applies_add(cfg):
    return getattr(cfg, "perc_add", 0) > 0


def _applies_sub(cfg):
    return getattr(cfg, "perc_sub", 0) > 0


def _run_linxfer(model, cfg, nodes):
    from quanta_maths.maths_batch import tag_linxfer_nodes
    return tag_linxfer_nodes(model, cfg, nodes)


def _run_carry_finalization(model, cfg, nodes):
    from quanta_maths.maths_temporal_finalization import tag_carry_finalization_nodes
    return tag_carry_finalization_nodes(model, cfg, nodes)


def _run_delivery_route(model, cfg, nodes):
    from quanta_maths.maths_cascade import tag_delivery_route_nodes
    return tag_delivery_route_nodes(model, cfg, nodes)


def _applies_cascade(cfg):
    return getattr(cfg, "perc_add", 0) > 0 or getattr(cfg, "perc_sub", 0) > 0


def _minus():
    from quanta_maths.maths_constants import MathsToken
    return MathsToken.MINUS


# The registry. New addition/mixed techniques append Technique(...) entries here.
TECHNIQUES: List[Technique] = [
    Technique(
        name="add_combiner_STC",
        target=FEATURES_FILE,
        applies_to=_applies_add,
        owns_tag=lambda t: t.startswith("Algo:") and t.endswith(".STC"),
        run=_run_combiner(operation=None),  # None -> PLUS unless pure-sub cfg
        description="Answer-position last-layer MLP that combines the resolved carry into An (CE5).",
    ),
    Technique(
        name="sub_combiner_MTC",
        target=FEATURES_FILE,
        applies_to=_applies_sub,
        owns_tag=lambda t: t.startswith("Algo:") and t.endswith(".MTC"),
        run=_run_combiner_cls("SUB"),
        description="Answer-position last-layer MLP that combines the resolved borrow into An (positive-answer sub; CE22 mixed).",
    ),
    Technique(
        name="neg_combiner_NTC",
        target=FEATURES_FILE,
        applies_to=_applies_sub,
        owns_tag=lambda t: t.startswith("Algo:") and t.endswith(".NTC"),
        run=_run_combiner_cls("NEG"),
        description="Answer-position last-layer MLP that combines the resolved neg-borrow into An (negative-answer sub; CE22 mixed).",
    ),
    Technique(
        name="operand_linear_transfer_LINXFER",
        target=BEHAVIORS_FILE,
        applies_to=_applies_add,
        owns_tag=lambda t: t.startswith("Probe:") and ".LINXFER" in t,
        run=_run_linxfer,
        description="Operand digit is linearly decodable at its first-layer fetch site (CE2).",
    ),
    Technique(
        name="carry_temporal_finalization_CARRY",
        target=BEHAVIORS_FILE,
        applies_to=_applies_add,
        owns_tag=lambda t: t.startswith("Probe:") and ".CARRY" in t,
        run=_run_carry_finalization,
        description=("Token-time finalization of the propagated carry (CE24 TF): "
                     "Probe:A{top}.CARRYLAYER (read layer) + .CARRYDEFER (tokens past "
                     "full input availability; 0=eager, >0=lazy). ADDITION carry only "
                     "-- a borrow/neg-borrow variant is still TODO (see maths-code-"
                     "migration-plan)."),
    ),
    Technique(
        name="combiner_delivery_route",
        target=BEHAVIORS_FILE,
        applies_to=_applies_cascade,
        owns_tag=lambda t: t.startswith("Probe:DELIVERY"),
        run=_run_delivery_route,
        description=("How the resolved carry/borrow reaches the combiner per class "
                     "(Probe:DELIVERY.{ADD|SUB|NEG}=res|resatt|att|none; CE25 mixed). "
                     "ADD tends residual-only, SUB/NEG residual+attention. Run across "
                     "the zoo -- the route may differ by model."),
    ),
]


def register_technique(technique: Technique) -> None:
    """Append a technique to the registry (used by the addition/mixed threads)."""
    TECHNIQUES.append(technique)


def techniques_for(cfg, techniques: Optional[List[Technique]] = None) -> List[Technique]:
    """The subset of ``techniques`` (default: all registered) that apply to ``cfg``."""
    techniques = TECHNIQUES if techniques is None else techniques
    return [t for t in techniques if t.applies_to(cfg)]


# ===========================================================================
# Model discovery
# ===========================================================================

def list_analysis_models(require_analysis: bool = True) -> List[str]:
    """List model names that have a ``QuantaMaths_<name>`` repo.

    With ``require_analysis`` (default) only models whose repo already contains both
    ``behaviors.json`` and ``features.json`` are returned (the ~33 analysable set).
    """
    from huggingface_hub import HfApi
    api = HfApi()
    names = []
    for m in api.list_models(author="PhilipQuirke"):
        rid = m.id
        if "/QuantaMaths_" not in rid:
            continue
        name = rid.split("/QuantaMaths_", 1)[1]
        if require_analysis:
            try:
                files = set(api.list_repo_files(rid))
            except Exception:
                continue
            if not ({BEHAVIORS_FILE, FEATURES_FILE} <= files):
                continue
        names.append(name)
    return sorted(names)


def list_trainonly_models() -> List[str]:
    """List QuantaMaths_<name> repos that have model.pth + training_loss.json but
    LACK behaviors.json/features.json -- the repos needing discovery."""
    from huggingface_hub import HfApi
    api = HfApi()
    names = []
    for m in api.list_models(author="PhilipQuirke"):
        rid = m.id
        if "/QuantaMaths_" not in rid:
            continue
        try:
            files = set(api.list_repo_files(rid))
        except Exception:
            continue
        if "model.pth" in files and not ({BEHAVIORS_FILE, FEATURES_FILE} <= files):
            names.append(rid.split("/QuantaMaths_", 1)[1])
    return sorted(names)


def _op_group(name: str) -> str:
    """Operation group for ordering: 'add' | 'sub' | 'mix' (mix covers mix_/ins*_mix_/mas_)."""
    if name.startswith("add_"):
        return "add"
    if name.startswith("sub_"):
        return "sub"
    return "mix"


def _n_digits(name: str) -> int:
    m = re.search(r"_d(\d+)_", name)
    return int(m.group(1)) if m else 0


def ordered_analysis_models(models: Optional[List[str]] = None) -> List[str]:
    """Return processable models ordered addition -> subtraction -> mixed, each
    small -> large by digit count (then name). Default: all ~33 analysable models."""
    if models is None:
        models = list_analysis_models(require_analysis=True)
    order = {"add": 0, "sub": 1, "mix": 2}
    return sorted(models, key=lambda n: (order[_op_group(n)], _n_digits(n), n))


# ===========================================================================
# Per-model update
# ===========================================================================

def _load_nodes(model_name: str, filename: str, local_path: str):
    """Download a node-list JSON from the analysis repo into a UsefulNodeList."""
    from QuantaMechInterp import UsefulNodeList
    from huggingface_hub import hf_hub_download
    src = hf_hub_download(repo_id=analysis_repo_id(model_name), filename=filename)
    shutil.copyfile(src, local_path)
    nodes = UsefulNodeList()
    nodes.load_nodes(local_path)
    return nodes


def _verify_roundtrip(path: str, major: str) -> bool:
    """Save->reload stability check: reload the written JSON and re-serialize it;
    the bytes must match. Confirms the file parses and round-trips losslessly."""
    from QuantaMechInterp import UsefulNodeList
    nl = UsefulNodeList()
    nl.load_nodes(path)          # must parse
    tmp = path + ".roundtrip"
    nl.save_nodes(tmp, major)
    ok = filecmp.cmp(path, tmp, shallow=False)
    os.remove(tmp)
    return ok


def _repo_has_analysis(model_name: str) -> bool:
    """True if the repo already contains both behaviors.json and features.json."""
    from huggingface_hub import HfApi
    try:
        files = set(HfApi().list_repo_files(analysis_repo_id(model_name)))
    except Exception:
        return False
    return {BEHAVIORS_FILE, FEATURES_FILE} <= files


def _current_hf_text(model_name: str, filename: str):
    """Return the current text of ``filename`` in the model's repo, or None if absent."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo_id=analysis_repo_id(model_name), filename=filename)
        with open(path) as f:
            return f.read()
    except Exception:
        return None


def _generate_mechanism_md(model_name, cfg, features_path, behaviors_path) -> str:
    """Build the auto-generated per-model mechanism Markdown from the just-written
    features.json (Algo roles) + behaviors.json (behaviour tags). Pure/offline."""
    from quanta_maths.maths_diagram import build_model_map, build_mechanism_markdown
    with open(features_path) as f:
        feats = json.load(f)
    with open(behaviors_path) as f:
        behav = json.load(f)
    model_map = build_model_map(model_name, cfg, feats, behav)
    return build_mechanism_markdown(model_map, cfg=cfg)


def _run_techniques(model, cfg, nodes, target, applicable, skip_if_present, results):
    """Run the applicable techniques whose target == ``target`` against ``nodes``.
    Records per-technique outcome in ``results``; returns True if any tag added."""
    changed = False
    for t in applicable:
        if t.target != target:
            continue
        if skip_if_present and t.is_present(nodes):
            results[t.name] = {"target": t.target, "tags_added": 0, "skipped": "already_present"}
            continue
        t.clear_owned(nodes)
        added = t.run(model, cfg, nodes)
        results[t.name] = {"target": t.target, "tags_added": added}
        if added:
            changed = True
    return changed


def update_model(
    model_name: str,
    techniques: Optional[List[Technique]] = None,
    dry_run: bool = True,
    work_dir: str = "results/hf-update",
    device: str = "cpu",
    upload: bool = True,
    skip_if_present: bool = True,
    allow_discovery: bool = True,
) -> dict:
    """Run applicable techniques on one model and (optionally) upload the results.

    If the repo already has behaviors.json + features.json, those node lists are
    downloaded and the techniques extend them. If they are MISSING and
    ``allow_discovery`` is set, the QMAnalyse discovery pipeline
    (``maths_analysis``) first CREATES them from the model, then the techniques run.

    ``skip_if_present`` (default): a technique whose owned tags are already present
    is skipped (resumable re-runs). Discovery never overwrites existing analysis
    files -- a repo that already has them takes the extend path.

    Every written file is round-trip verified (save->reload) before upload.
    Per-model errors are captured, not raised, so a batch continues.
    """
    result = {"model": model_name, "dry_run": dry_run, "techniques": {},
              "discovered": False, "uploaded": False, "roundtrip_ok": None, "error": None}
    try:
        has_analysis = _repo_has_analysis(model_name)
        model, cfg = load_maths_model_from_analysis_repo(model_name, device=device)
        applicable = techniques_for(cfg, techniques)
        result["applicable"] = [t.name for t in applicable]

        model_dir = os.path.join(work_dir, model_name)
        backup_dir = os.path.join(model_dir, "original")
        out_dir = os.path.join(model_dir, "updated")
        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        written = {}
        changed_files = set()

        if has_analysis:
            # EXTEND path: download both lists, run techniques (skip-if-present).
            nodes_by_file = {
                f: _load_nodes(model_name, f, os.path.join(backup_dir, f))
                for f in (BEHAVIORS_FILE, FEATURES_FILE)}
            for fname in (BEHAVIORS_FILE, FEATURES_FILE):
                if _run_techniques(model, cfg, nodes_by_file[fname], fname,
                                   applicable, skip_if_present, result["techniques"]):
                    changed_files.add(fname)
            for fname, nodes in nodes_by_file.items():
                path = os.path.join(out_dir, fname)
                nodes.save_nodes(path, _SAVE_MAJOR[fname])
                written[fname] = path
        else:
            # DISCOVERY path: create behaviors.json + features.json from scratch.
            if not allow_discovery:
                result["error"] = "no analysis JSON and discovery disabled"
                return result
            from quanta_maths.maths_analysis import discover_behaviors, discover_features
            result["discovered"] = True

            # behaviours first; then behaviour-targeted techniques; SAVE behaviors.json
            # BEFORE any Algo tags exist (keeps behaviors.json Algo-free).
            discover_behaviors(cfg, model)
            _run_techniques(model, cfg, cfg.useful_nodes, BEHAVIORS_FILE,
                            applicable, skip_if_present, result["techniques"])
            bpath = os.path.join(out_dir, BEHAVIORS_FILE)
            cfg.useful_nodes.save_nodes(bpath, _SAVE_MAJOR[BEHAVIORS_FILE])
            written[BEHAVIORS_FILE] = bpath
            changed_files.add(BEHAVIORS_FILE)

            # features (Algo) next; then feature-targeted techniques; SAVE features.json
            discover_features(cfg)
            _run_techniques(model, cfg, cfg.useful_nodes, FEATURES_FILE,
                            applicable, skip_if_present, result["techniques"])
            fpath = os.path.join(out_dir, FEATURES_FILE)
            cfg.useful_nodes.save_nodes(fpath, _SAVE_MAJOR[FEATURES_FILE])
            written[FEATURES_FILE] = fpath
            changed_files.add(FEATURES_FILE)

        # Auto-generated mechanism.md (per-model diagram doc). Belongs on HF next to
        # behaviors/features -- NOT in git. Derived from the JSONs just written;
        # uploaded only if it differs from the copy on HF (keeps re-runs idempotent).
        md = _generate_mechanism_md(model_name, cfg, written[FEATURES_FILE],
                                    written[BEHAVIORS_FILE])
        mdpath = os.path.join(out_dir, MECHANISM_FILE)
        with open(mdpath, "w") as f:
            f.write(md)
        written[MECHANISM_FILE] = mdpath
        if md != _current_hf_text(model_name, MECHANISM_FILE):
            changed_files.add(MECHANISM_FILE)

        result["local_updated"] = written
        result["changed_files"] = sorted(changed_files)

        # Save->reload round-trip verification of the node-list JSONs (gates upload).
        result["roundtrip_ok"] = all(
            _verify_roundtrip(written[f], _SAVE_MAJOR[f]) for f in written if f in _SAVE_MAJOR)
        if not result["roundtrip_ok"]:
            result["error"] = "roundtrip verification failed; not uploaded"
            return result
        # Sanity-check the generated MD is non-empty and re-readable.
        if not (os.path.getsize(written[MECHANISM_FILE]) > 0):
            result["error"] = "empty mechanism.md; not uploaded"
            return result

        # Upload changed files, only when not a dry run.
        if upload and not dry_run and changed_files:
            from huggingface_hub import HfApi
            api = HfApi()
            note = "discovery + techniques" if result["discovered"] else "techniques"
            for fname in sorted(changed_files):
                api.upload_file(
                    path_or_fileobj=written[fname],
                    path_in_repo=fname,
                    repo_id=analysis_repo_id(model_name),
                    commit_message=f"Refresh {fname} via quanta_maths {note}",
                )
            result["uploaded"] = True
    except Exception as exc:  # per-model isolation
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ===========================================================================
# Batch runner
# ===========================================================================

def upload_mechanism_docs(models: Optional[List[str]] = None, dry_run: bool = True,
                          work_dir: str = "results/hf-update") -> dict:
    """Generate + upload the per-model auto ``mechanism.md`` to each QuantaMaths repo.

    Lightweight: builds the doc from the model's HF ``features.json`` +
    ``behaviors.json`` (map-only, no model load, no forward pass), and uploads it
    only when it differs from the copy already on HF (idempotent). Default scope is
    all analysable models. DRY-RUN by default.
    """
    from quanta_maths.maths_diagram import build_mechanism_markdown_for_model
    from huggingface_hub import HfApi

    models = ordered_analysis_models(models)
    os.makedirs(work_dir, exist_ok=True)
    api = HfApi()
    per_model = []
    for name in models:
        rec = {"model": name, "uploaded": False, "changed": False, "error": None}
        try:
            md = build_mechanism_markdown_for_model(name)
            path = os.path.join(work_dir, name, "updated", MECHANISM_FILE)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(md)
            rec["changed"] = md != _current_hf_text(name, MECHANISM_FILE)
            if rec["changed"] and not dry_run:
                api.upload_file(path_or_fileobj=path, path_in_repo=MECHANISM_FILE,
                                repo_id=analysis_repo_id(name),
                                commit_message="Add/refresh auto mechanism.md (quanta_maths.maths_diagram)")
                rec["uploaded"] = True
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        per_model.append(rec)
    return {"dry_run": dry_run, "n_models": len(per_model),
            "n_changed": sum(1 for r in per_model if r["changed"]),
            "n_uploaded": sum(1 for r in per_model if r["uploaded"]),
            "n_errors": sum(1 for r in per_model if r["error"]), "models": per_model}


def update_models(
    models: Optional[List[str]] = None,
    techniques: Optional[List[Technique]] = None,
    dry_run: bool = True,
    work_dir: str = "results/hf-update",
    device: str = "cpu",
    upload: bool = True,
    skip_if_present: bool = True,
    manifest_path: Optional[str] = None,
) -> dict:
    """Run the refresh over ``models`` (default: all ~33 analysable models, ordered
    addition -> subtraction -> mixed, small -> large).

    DRY-RUN IS THE DEFAULT. Pass ``dry_run=False`` to actually upload. Returns an
    overall manifest and writes it to ``manifest_path`` (default: under work_dir).
    """
    models = ordered_analysis_models(models)

    os.makedirs(work_dir, exist_ok=True)
    per_model = []
    for name in models:
        per_model.append(update_model(
            name, techniques=techniques, dry_run=dry_run,
            work_dir=work_dir, device=device, upload=upload,
            skip_if_present=skip_if_present))

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "uploaded_any": any(r["uploaded"] for r in per_model),
        "n_models": len(per_model),
        "n_errors": sum(1 for r in per_model if r["error"]),
        "n_roundtrip_fail": sum(1 for r in per_model if r["roundtrip_ok"] is False),
        "registered_techniques": [t.name for t in (techniques or TECHNIQUES)],
        "models": per_model,
    }
    if manifest_path is None:
        manifest_path = os.path.join(work_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    manifest["manifest_path"] = manifest_path
    return manifest


# ===========================================================================
# CLI
# ===========================================================================

def _main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Refresh QuantaMaths analysis JSONs on HF.")
    p.add_argument("--models", nargs="*", default=None,
                   help="Model names (default: all analysable). Omit for the full set.")
    p.add_argument("--execute", action="store_true",
                   help="Actually upload. Without this flag the run is a DRY RUN.")
    p.add_argument("--no-upload", action="store_true",
                   help="Never upload even with --execute (compute + write locally only).")
    p.add_argument("--no-skip", action="store_true",
                   help="Recompute every technique even if its tags are already present.")
    p.add_argument("--work-dir", default="results/hf-update")
    args = p.parse_args(argv)

    manifest = update_models(
        models=args.models,
        dry_run=not args.execute,
        upload=not args.no_upload,
        skip_if_present=not args.no_skip,
        work_dir=args.work_dir,
    )
    mode = "DRY RUN" if manifest["dry_run"] else "EXECUTE"
    print(f"[{mode}] models={manifest['n_models']} errors={manifest['n_errors']} "
          f"roundtrip_fail={manifest['n_roundtrip_fail']} uploaded_any={manifest['uploaded_any']}")
    for r in manifest["models"]:
        tags = {k: v["tags_added"] for k, v in r["techniques"].items()}
        flag = "ERR " + r["error"] if r["error"] else ("uploaded" if r["uploaded"] else "local-only")
        print(f"  {r['model']}: {tags} [{flag}]")
    print(f"manifest -> {manifest['manifest_path']}")
    return manifest


if __name__ == "__main__":
    _main()
