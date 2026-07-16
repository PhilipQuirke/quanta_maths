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

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from quanta_maths.maths_model_loader import (
    analysis_repo_id, load_maths_model_from_analysis_repo, ANALYSIS_REPO_PREFIX)

BEHAVIORS_FILE = "behaviors.json"
FEATURES_FILE = "features.json"

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


# --- built-in techniques (wrapping the existing library taggers) -----------

def _run_combiner(operation):
    def _fn(model, cfg, nodes):
        from quanta_maths.maths_batch import tag_stc_nodes
        return tag_stc_nodes(model, cfg, nodes, operation=operation)
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
        run=lambda model, cfg, nodes: _run_combiner(_minus())(model, cfg, nodes),
        description="Answer-position last-layer MLP that combines the resolved borrow into An (sub parallel of STC).",
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
                     "full input availability; 0=eager, >0=lazy). Run across models."),
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


def update_model(
    model_name: str,
    techniques: Optional[List[Technique]] = None,
    dry_run: bool = True,
    work_dir: str = "results/hf-update",
    device: str = "cpu",
    upload: bool = True,
) -> dict:
    """Run applicable techniques on one model and (optionally) upload the results.

    Returns a per-model manifest dict. Never raises for per-model failures -- the
    error is captured in the manifest so a batch can continue.
    """
    result = {"model": model_name, "dry_run": dry_run, "techniques": {},
              "uploaded": False, "error": None}
    try:
        model, cfg = load_maths_model_from_analysis_repo(model_name, device=device)

        applicable = techniques_for(cfg, techniques)
        result["applicable"] = [t.name for t in applicable]

        model_dir = os.path.join(work_dir, model_name)
        backup_dir = os.path.join(model_dir, "original")
        out_dir = os.path.join(model_dir, "updated")
        os.makedirs(backup_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)

        # Load both node lists (kept separate; see module docstring).
        nodes_by_file = {}
        for fname in (BEHAVIORS_FILE, FEATURES_FILE):
            nodes_by_file[fname] = _load_nodes(
                model_name, fname, os.path.join(backup_dir, fname))

        # Run each applicable technique against its target file (idempotent).
        changed_files = set()
        for t in applicable:
            nodes = nodes_by_file[t.target]
            t.clear_owned(nodes)
            added = t.run(model, cfg, nodes)
            result["techniques"][t.name] = {"target": t.target, "tags_added": added}
            if added:
                changed_files.add(t.target)

        # Write updated JSON locally (always -- lets you inspect a dry run).
        written = {}
        for fname, nodes in nodes_by_file.items():
            path = os.path.join(out_dir, fname)
            nodes.save_nodes(path, _SAVE_MAJOR[fname])
            written[fname] = path
        result["local_updated"] = written
        result["changed_files"] = sorted(changed_files)

        # Upload only the changed files, and only when not a dry run.
        if upload and not dry_run and changed_files:
            from huggingface_hub import HfApi
            api = HfApi()
            for fname in sorted(changed_files):
                api.upload_file(
                    path_or_fileobj=written[fname],
                    path_in_repo=fname,
                    repo_id=analysis_repo_id(model_name),
                    commit_message=f"Refresh {fname} via quanta_maths techniques "
                                   f"({', '.join(t.name for t in applicable)})",
                )
            result["uploaded"] = True
    except Exception as exc:  # per-model isolation
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


# ===========================================================================
# Batch runner
# ===========================================================================

def update_models(
    models: Optional[List[str]] = None,
    techniques: Optional[List[Technique]] = None,
    dry_run: bool = True,
    work_dir: str = "results/hf-update",
    device: str = "cpu",
    upload: bool = True,
    manifest_path: Optional[str] = None,
) -> dict:
    """Run the refresh over ``models`` (default: all ~33 analysable models).

    DRY-RUN IS THE DEFAULT. Pass ``dry_run=False`` to actually upload. Returns an
    overall manifest and writes it to ``manifest_path`` (default: under work_dir).
    """
    if models is None:
        models = list_analysis_models(require_analysis=True)

    os.makedirs(work_dir, exist_ok=True)
    per_model = []
    for name in models:
        per_model.append(update_model(
            name, techniques=techniques, dry_run=dry_run,
            work_dir=work_dir, device=device, upload=upload))

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "uploaded_any": any(r["uploaded"] for r in per_model),
        "n_models": len(per_model),
        "n_errors": sum(1 for r in per_model if r["error"]),
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
    p.add_argument("--work-dir", default="results/hf-update")
    args = p.parse_args(argv)

    manifest = update_models(
        models=args.models,
        dry_run=not args.execute,
        upload=not args.no_upload,
        work_dir=args.work_dir,
    )
    mode = "DRY RUN" if manifest["dry_run"] else "EXECUTE"
    print(f"[{mode}] models={manifest['n_models']} errors={manifest['n_errors']} "
          f"uploaded_any={manifest['uploaded_any']}")
    for r in manifest["models"]:
        tags = {k: v["tags_added"] for k, v in r["techniques"].items()}
        flag = "ERR " + r["error"] if r["error"] else ("uploaded" if r["uploaded"] else "local-only")
        print(f"  {r['model']}: {tags} [{flag}]")
    print(f"manifest -> {manifest['manifest_path']}")
    return manifest


if __name__ == "__main__":
    _main()
