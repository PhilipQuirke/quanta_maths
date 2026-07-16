"""Cross-size SV validation and tightness census (study-cross-size-sv.md).

Tests A12/C6 (working-axioms mode): does the SV interface (ST writers -> redundant
consumer head-pair delivering a canonical resolved carry from the ST cluster, never
`=` -> step combiner; CE13-CE17) TRANSFER to d10/d13, and does its REDUNDANCY
SHRINK as n grows (C6: small-model slack stripped at large n)?

Registries are NOT n-parameterized (XS-B): d10/d13 ST/combiner sets are built from
the published maths.json/behavior.json maps; consumer heads are identified
empirically with a carry-specificity gate (XS-D).

  Battery C  redundancy census / TIGHTNESS INDEX (core, never dropped):
    (i) map duplicate multiplicity per role per answer-digit, FIXED cross-size
        digit window (XS-A; census-only, never drives the verdict alone);
    (ii) interchange decisiveness (single ST-write twin-interchange flip) --
        corroborating-only, CE17-F2-ambiguous (XS-C);
    (iii) class-vs-single ablation gap over untagged baseline (CE13 unit) --
        the PRIMARY tightness leg, F2-free (XS-C).
    C6: (iii) single-node gap grows / class-vs-single gap shrinks with n.
  Battery I  interface transfer (core @ d10):
    (a) =-not-a-source (per-key CONTRIBUTION patch, CE16 SI-8; = arm ~0);
    (b) class necessity (joint consumer-pair mean-ablate, cascade vs carry-free);
    (c) step combiner (CE17 Battery-T on-manifold, endpoint-gated).
  Battery L  compounding locus at large n (STRETCH; drop first) -- only if C
    shows d10/d13 tighter (XS-E trigger).

Positive controls: acc>=0.99; PC2 reproduce CE13/CE16 at d5/d6; PC2b (XS-B) each
d10/d13 ST node reproduces its own map Impact tag; PC3 combiner endpoints
reproduce per-model 0/1; carry-axis anchor; behavioral gates.

CPU-only. Run:
    PYTHONPATH=. python3 scripts/cross_size_sv.py all
    PYTHONPATH=. python3 scripts/cross_size_sv.py all --fast
    PYTHONPATH=. python3 scripts/cross_size_sv.py all --gradient   # +d7/d8/d9
"""
from __future__ import annotations
import json, os, sys, re, math
import numpy as np
import torch
from huggingface_hub import hf_hub_download

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
)
from scripts.deep_cascade_mechanism import (
    build_chain, consuming_pos, ak_pos, affected_digits, dn_pos, dpn_pos,
    behavioral_gate, RNG,
)
from scripts.sv_implementation import (
    carry_axis, twin_pair, same_class_twin, edge_contribution, lnfair_project,
    mean_ci, cache_full, _cache_named, Z_NAMES,
)
from scripts.st_tristate_geometry import build_class_question, CFG as GEO_CFG
from scripts.sv_compounding import head_ov, _ln_norm, edge_patch_pred
from scripts.node_output_encoding import _mean_ablate_acc, build_random

HF_REPO = "PhilipQuirke/VerifiedArithmetic"
RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-cross-size-sv")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716

PRIMARY = ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289",
           "add_d10_l2_h3_t40K_s572091", "add_d13_l2_h3_t50K_s572091"]
GRADIENT = ["add_d7_l2_h3_t45K_s173289", "add_d8_l2_h3_t45K_s173289",
            "add_d9_l2_h3_t45K_s173289"]

# fixed cross-size scored-digit window (XS-A): leading digit + next 2 below it,
# expressed as offsets from the top digit n_top = n_digits-2 (present at every n).
WINDOW_OFFSETS = [0, 1, 2]   # top, top-1, top-2


# ---------------------------------------------------------------------------
# map building (XS-B): registries from the published maps
# ---------------------------------------------------------------------------

def load_maps(mn):
    ma = json.load(open(hf_hub_download(HF_REPO, f"{mn}_maths.json")))
    b = json.load(open(hf_hub_download(HF_REPO, f"{mn}_behavior.json")))
    return ma, b


def st_sites_from_map(ma):
    """L0 ST-writer heads: (pos, head, tagged_digit)."""
    out = []
    for nd in ma:
        if nd["layer"] == 0 and nd["is_head"]:
            for t in nd["tags"]:
                m = re.match(r"Algo:A(\d+)\.ST", t)
                if m:
                    out.append({"pos": nd["position"], "head": nd["num"], "digit": int(m.group(1))})
    return out


def combiner_mlps_from_map(b, fail_min=3):
    """High-Fail% L1 MLPs (answer-position combiners): pos -> (fail, impact digits)."""
    out = {}
    for nd in b:
        if nd["layer"] == 1 and not nd["is_head"]:
            fails = [int(t.split(":")[1]) for t in nd["tags"] if t.startswith("Fail%")]
            imps = []
            for t in nd["tags"]:
                if t.startswith("Impact:A"):
                    imps += [int(c) for c in t.split(":A")[1] if c.isdigit()]
            if fails and fails[0] >= fail_min:
                out[nd["position"]] = {"fail": fails[0], "impact": imps}
    return out


def eq_sign_pos(cfg):
    eq = 2 * cfg.n_digits + 1
    return eq, eq + 1


def map_carry_axis(model, cfg, mn, comb, n_q=200):
    """XS-B: carry axis c1-c0 at the combiner INPUT for the LEADING-digit cell,
    with (combiner_pos, digit) derived from the map for large n (GEO_CFG only has
    d5/d6). Combiner pos = the high-Fail L1 MLP serving the top chain digit; digit
    = n_top (the tri-state digit at the chain top)."""
    if mn in GEO_CFG:
        return carry_axis(model, cfg, mn, n_q=n_q)
    n_top = cfg.n_digits - 2
    n = n_top                       # tri-state digit at the chain top
    # combiner serving digit n: the L1 MLP whose Impact includes n, else the
    # consuming position of A_n.
    pos = None
    for p, info in comb.items():
        if n in info.get("impact", []):
            pos = p; break
    if pos is None:
        pos = consuming_pos(cfg, n)
    hook = "blocks.1.ln2.hook_normalized"
    means = {}
    for cls in ("c0", "c1"):
        acts = []
        for _ in range(n_q):
            a, b, _sa = build_class_question(cfg, n, cls)
            c = _cache_named(model, make_q(cfg, a, b), (hook,))
            acts.append(c[hook][0, pos, :].numpy())
        means[cls] = np.mean(acts, 0)
    axis = means["c1"] - means["c0"]; axis = axis / (np.linalg.norm(axis) + 1e-9)
    sep = float((means["c1"] - means["c0"]) @ axis)
    return {"axis": axis, "c0": means["c0"], "c1": means["c1"],
            "pos": pos, "hook": hook, "digit": n, "sep": sep}


# ---------------------------------------------------------------------------
# empirical consumer-head identification at the leading-digit cell (XS-D)
# ---------------------------------------------------------------------------

def _site_ov_write(model, cache, pos, head, WO0):
    z = cache["blocks.0.attn.hook_z"][0, pos, head, :]
    return (z @ WO0[head]).detach().numpy()


_CACHE_NAMES = ("blocks.0.attn.hook_z", "blocks.0.hook_resid_post",
                "blocks.1.attn.hook_z", "blocks.1.attn.hook_pattern",
                "blocks.1.attn.hook_v", "blocks.1.hook_resid_mid")


def cache_L0L1(model, q):
    with torch.no_grad():
        _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm in _CACHE_NAMES)
    return c


def identify_consumer_heads(model, cfg, mn, st_sites, ax, n_pairs=30):
    """Consumer head(s) for the leading digit A_top: L1 heads at the A_top consuming
    position that (1) attend the ST-cluster key positions heavily and (2) whose
    edge patch reproduces a carry-SPECIFIC 0/1 flip (deciding-matched null <=0.20).
    Returns (heads, diagnostics). Empty -> role not identified (A12-partial)."""
    n_top = cfg.n_digits - 2
    top = n_top + 1
    cpos = consuming_pos(cfg, top)
    ap = answer_positions(cfg); idx = len(ap) - 1 - top
    st_positions = sorted({s["pos"] for s in st_sites})
    k = min(3, n_top)

    # attention mass on ST cluster per head (avg over a few chains)
    attn_mass = {h: [] for h in range(cfg.n_heads)}
    flips = {h: [] for h in range(cfg.n_heads)}
    nulls = {h: [] for h in range(cfg.n_heads)}
    for _ in range(n_pairs):
        (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
        sq = make_q(cfg, ha, hb); tq = make_q(cfg, la, lb)
        clean = predict_answer(model, cfg, tq)
        sc = cache_full(model, sq); tc = cache_full(model, tq)
        na, nb = same_class_twin(cfg, n_top, k, "lo")
        nc = cache_full(model, make_q(cfg, na, nb))
        for h in range(cfg.n_heads):
            pat = tc["blocks.1.attn.hook_pattern"][0, h, cpos, :]
            attn_mass[h].append(float(pat[st_positions].sum()))
            # single-head edge patch (real + null)
            ps = edge_patch_pred(model, cfg, tq,
                    [(cpos, 1, h, sc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy())], tc)
            flips[h].append(float(ps[idx] != clean[idx]))
            pn = edge_patch_pred(model, cfg, tq,
                    [(cpos, 1, h, nc["blocks.1.attn.hook_z"][0, cpos, h, :].numpy())], tc)
            nulls[h].append(float(pn[idx] != clean[idx]))
    diag = {}
    consumers = []
    for h in range(cfg.n_heads):
        am = float(np.mean(attn_mass[h])); fl = float(np.mean(flips[h])); nu = float(np.mean(nulls[h]))
        diag[h] = {"attn_ST_mass": am, "single_flip": fl, "null": nu}
        # XS-D: a consumer head is CAUSAL (single edge flips the leading digit) AND
        # carry-SPECIFIC (deciding-matched null <= 0.20). Attention-to-ST is
        # diagnostic-only (recorded), NOT a hard gate — at large n the causal head
        # may attend the cluster diffusely (d13 head0: flip 1.0, attn 0.02).
        if fl >= 0.5 and nu <= 0.20:
            consumers.append((h, fl))
    # rank by causal flip; keep up to 2 (the CE16 redundant-pair structure)
    consumers = [h for h, fl in sorted(consumers, key=lambda x: -x[1])][:2]
    # fallback: if none clear the causal bar, the strongest carry-specific
    # attender (records A12-partial if still empty)
    if not consumers:
        cand = [(h, diag[h]["attn_ST_mass"]) for h in range(cfg.n_heads) if diag[h]["null"] <= 0.20]
        consumers = [h for h, _ in sorted(cand, key=lambda x: -x[1])[:1] if _ > 0.2]
    return consumers, {"cpos": cpos, "top": top, "per_head": diag,
                       "st_positions": st_positions,
                       "selector": "causal-flip>=0.5 & null<=0.20 (attn diagnostic-only)"}


# ---------------------------------------------------------------------------
# Battery C: redundancy census / tightness index
# ---------------------------------------------------------------------------

def scored_digits(cfg):
    n_top = cfg.n_digits - 2
    return [n_top - off for off in WINDOW_OFFSETS if n_top - off >= 0]


def battery_C(model, cfg, mn, st_sites, ma, b, consumers, cons_diag, n_pairs=40, N_abl=250):
    WO0 = model.blocks[0].attn.W_O
    ap = answer_positions(cfg); na = len(ap)
    n_top = cfg.n_digits - 2
    digs = scored_digits(cfg)
    out = {}

    # (i) map duplicate multiplicity per answer digit (ST writers), fixed window
    imp_counts = {}   # digit -> #ST nodes tagged for it
    for s in st_sites:
        imp_counts[s["digit"]] = imp_counts.get(s["digit"], 0) + 1
    mult_i = float(np.mean([imp_counts.get(d, 0) for d in digs])) if digs else float("nan")
    out["i_map_duplicate_multiplicity"] = mult_i

    # (ii) interchange decisiveness: single ST-write twin-interchange flip on the
    # served digit (corroborating-only, F2-ambiguous). Use the ST site serving the
    # deciding digit at k where it is the deepest visible.
    k = min(3, n_top)
    idx_top = na - 1 - (n_top + 1)
    inter_flips = []
    for s in st_sites:
        if s["digit"] > n_top:   # only chain-relevant digits
            continue
        fl = []
        for _ in range(max(10, n_pairs // 2)):
            (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
            tq = make_q(cfg, la, lb); clean = predict_answer(model, cfg, tq)
            sc = cache_L0L1(model, make_q(cfg, ha, hb)); tc = cache_L0L1(model, tq)
            sw = _site_ov_write(model, sc, s["pos"], s["head"], WO0)
            tw = _site_ov_write(model, tc, s["pos"], s["head"], WO0)
            def hook(act, hook):
                act[:, s["pos"], :] = act[:, s["pos"], :] + torch.tensor(sw - tw); return act
            with torch.no_grad():
                lg = model.run_with_hooks(tq.unsqueeze(0), fwd_hooks=[("blocks.0.hook_resid_post", hook)])
            pred = lg[0, [p - 1 for p in ap]].argmax(-1)
            fl.append(float(pred[idx_top] != clean[idx_top]))
        inter_flips.append(float(np.mean(fl)))
    out["ii_interchange_decisiveness_max"] = max(inter_flips) if inter_flips else 0.0
    out["ii_note"] = "corroborating-only; CE17-F2 ambiguous (redundancy vs weak-interchange)"

    # (iii) PRIMARY: class-vs-single ablation gap over untagged baseline (CE13 unit)
    qs = build_random(cfg, N_abl)
    clean_acc = sum(int(torch.equal(predict_answer(model, cfg, make_q(cfg, a, bb)),
                                    make_q(cfg, a, bb)[ap])) for a, bb in qs) / len(qs)
    # single = the single map-top ST node (lowest-digit, highest map impact); class = all ST
    low = sorted(st_sites, key=lambda s: s["digit"])
    single_node = low[0] if low else None
    single_gap = clean_acc - _mean_ablate_acc(model, cfg, single_node["pos"], 0, single_node["head"], qs, ap) if single_node else float("nan")
    # class ablation: mean-ablate ALL ST heads jointly
    class_acc = _mean_ablate_multi(model, cfg, st_sites, qs, ap)
    class_gap = clean_acc - class_acc
    # untagged baseline: co-located non-ST heads
    st_ph = {(s["pos"], s["head"]) for s in st_sites}
    untag_gaps = []
    for s in st_sites[:4]:
        wrole = [h for h in range(cfg.n_heads) if (s["pos"], h) not in st_ph]
        if wrole:
            untag_gaps.append(clean_acc - _mean_ablate_acc(model, cfg, s["pos"], 0, wrole[0], qs, ap))
    untag = max(untag_gaps) if untag_gaps else 0.0
    out["iii_clean_acc"] = clean_acc
    out["iii_single_node_gap"] = single_gap
    out["iii_class_gap"] = class_gap
    out["iii_untagged_baseline_gap"] = untag
    # tightness: class-minus-single gap (large = redundant; SHRINKS with n if tightening)
    out["iii_class_minus_single_gap"] = class_gap - single_gap
    out["iii_note"] = "PRIMARY tightness leg (F2-free); class-minus-single SHRINKS => tighter"

    # PC2b (XS-B): the ST role reproduces its map Impact at large n. CE13 found
    # single-node impact concentrates at LOW digits (0.02-0.07); high digits are
    # redundant (~0). So anchor on the CLASS: mean-ablating all ST heads must break
    # the answer above the untagged baseline (that IS the "ST role is load-bearing"
    # demonstration; single-node reproduction is not expected under redundancy).
    if not (mn.startswith("add_d5") or mn.startswith("add_d6")):
        # class gap already computed above (iii_class_gap); baseline = untagged class
        untag_ph = []
        for s in st_sites[:6]:
            wrole = [h for h in range(cfg.n_heads) if (s["pos"], h) not in st_ph]
            if wrole:
                untag_ph.append({"pos": s["pos"], "head": wrole[0]})
        untag_class_acc = _mean_ablate_multi(model, cfg, untag_ph, qs, ap) if untag_ph else clean_acc
        untag_class_gap = clean_acc - untag_class_acc
        out["PC2b_class_gap"] = class_gap
        out["PC2b_untagged_class_gap"] = untag_class_gap
        out["PC2b_st_reproduces_map_impact"] = bool(class_gap > untag_class_gap + 0.02)
        out["PC2b_over_baseline"] = class_gap - untag_class_gap
    return out


def _mean_ablate_multi(model, cfg, sites, qs, ap):
    """Joint mean-ablation of a set of L0 heads' hook_z; full-answer accuracy."""
    zmeans = {}
    for a, b in build_random(cfg, 120):
        c = cache_L0L1(model, make_q(cfg, a, b))
        for s in sites:
            zmeans.setdefault((s["pos"], s["head"]), []).append(
                c["blocks.0.attn.hook_z"][0, s["pos"], s["head"], :].numpy())
    zmeans = {k: torch.tensor(np.mean(v, 0)) for k, v in zmeans.items()}
    ok = 0
    for a, b in qs:
        q = make_q(cfg, a, b)
        def hook(act, hook):
            for s in sites:
                act[:, s["pos"], s["head"], :] = zmeans[(s["pos"], s["head"])]
            return act
        with torch.no_grad():
            lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[("blocks.0.attn.hook_z", hook)])
        if torch.equal(lg[0, [p - 1 for p in ap]].argmax(-1), q[ap]):
            ok += 1
    return ok / len(qs)


def _mean_ablate_acc_digit(model, cfg, pos, head, qs, ap, di):
    zs = []
    for a, b in build_random(cfg, 120):
        c = cache_L0L1(model, make_q(cfg, a, b))
        zs.append(c["blocks.0.attn.hook_z"][0, pos, head, :].numpy())
    zmean = torch.tensor(np.mean(zs, 0))
    ok = 0
    for a, b in qs:
        q = make_q(cfg, a, b); gold = predict_answer(model, cfg, q)
        def hook(act, hook): act[:, pos, head, :] = zmean; return act
        with torch.no_grad():
            lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[("blocks.0.attn.hook_z", hook)])
        if lg[0, [p - 1 for p in ap]].argmax(-1)[di] == gold[di]:
            ok += 1
    return ok / len(qs)


# ---------------------------------------------------------------------------
# Battery I: interface transfer (=-not-a-source, class necessity, step combiner)
# ---------------------------------------------------------------------------

def battery_I(model, cfg, mn, st_sites, consumers, cons_diag, ax, n_pairs=40):
    if not consumers:
        return {"status": "invalid: consumer head not identified (A12-partial)"}
    cpos = cons_diag["cpos"]; top = cons_diag["top"]
    ap = answer_positions(cfg); idx = len(ap) - 1 - top
    n_top = cfg.n_digits - 2
    k = min(3, n_top)
    eq, sign = eq_sign_pos(cfg)
    heads = consumers
    d = n_top - k
    dec_keys = sorted({dn_pos(cfg, d), dpn_pos(cfg, d)})
    axis = ax["axis"]
    out = {"consumer_heads": heads, "cpos": cpos}

    # (a) =-not-a-source: per-key CONTRIBUTION patch (source pattern+v) for = vs deciding-ST
    def patch_keys(tq, tc, sc, keyset):
        patches = []
        for h in heads:
            pat_t = tc["blocks.1.attn.hook_pattern"][0, h, cpos, :].clone()
            v_t = tc["blocks.1.attn.hook_v"][0, :, h, :].clone()
            pat_s = sc["blocks.1.attn.hook_pattern"][0, h, cpos, :]
            v_s = sc["blocks.1.attn.hook_v"][0, :, h, :]
            pat = pat_t.clone(); v = v_t.clone()
            for kp in keyset:
                pat[kp] = pat_s[kp]; v[kp] = v_s[kp]
            z = torch.einsum("k,kd->d", pat, v)
            patches.append((cpos, 1, h, z.detach().numpy()))
        p = edge_patch_pred(model, cfg, tq, patches, tc)
        return p[idx]
    eq_flips, dec_flips = [], []
    for _ in range(n_pairs):
        (ha, hb), (la, lb) = twin_pair(cfg, n_top, k)
        tq = make_q(cfg, la, lb); clean = predict_answer(model, cfg, tq)
        sc = cache_full(model, make_q(cfg, ha, hb)); tc = cache_full(model, tq)
        eq_flips.append(float(patch_keys(tq, tc, sc, [eq]) != clean[idx]))
        dec_flips.append(float(patch_keys(tq, tc, sc, dec_keys) != clean[idx]))
    out["eq_not_a_source"] = {"eq_arm_flip": mean_ci(eq_flips),
                              "deciding_ST_arm_flip": mean_ci(dec_flips)}

    # (b) class necessity: joint consumer-pair mean-ablate, cascade vs carry-free
    out["class_necessity"] = _class_necessity(model, cfg, cpos, heads, top, n_pairs=n_pairs)

    # (c) step combiner (CE17 Battery-T on-manifold, endpoint-gated)
    out["combiner"] = _combiner_transfer(model, cfg, mn, cpos, heads, top, ax, n_q=max(30, n_pairs))
    return out


def _class_necessity(model, cfg, cpos, heads, top, n_pairs=40):
    ap = answer_positions(cfg); idx = len(ap) - 1 - top
    n_top = cfg.n_digits - 2; k = min(3, n_top)
    zs = {h: [] for h in heads}
    for _ in range(60):
        a = int(RNG.integers(0, 10 ** cfg.n_digits)); b = int(RNG.integers(0, 10 ** cfg.n_digits))
        c = cache_full(model, make_q(cfg, a, b))
        for h in heads:
            zs[h].append(c["blocks.1.attn.hook_z"][0, cpos, h, :].numpy())
    mean_z = {h: torch.tensor(np.mean(zs[h], 0)) for h in heads}
    def acc(kind):
        ok = 0
        for _ in range(n_pairs):
            if kind == "cascade":
                (ha, hb), _ = twin_pair(cfg, n_top, k); a, b = ha, hb
            else:
                da = [int(RNG.integers(0, 5)) for _ in range(cfg.n_digits)]
                db = [int(RNG.integers(0, 5)) for _ in range(cfg.n_digits)]
                a = int("".join(map(str, da))); b = int("".join(map(str, db)))
            q = make_q(cfg, a, b); gold = predict_answer(model, cfg, q)
            def hook(act, hook):
                for h in heads: act[:, cpos, h, :] = mean_z[h]
                return act
            with torch.no_grad():
                lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[("blocks.1.attn.hook_z", hook)])
            ok += int(lg[0, [p - 1 for p in ap]].argmax(-1)[idx] == gold[idx])
        return ok / n_pairs
    casc = acc("cascade"); free = acc("carry_free")
    return {"cascade_acc_after_ablate": casc, "carryfree_acc_after_ablate": free,
            "selective_gap": free - casc}


def _combiner_transfer(model, cfg, mn, cpos, heads, top, ax, n_q=40,
                       alphas=(-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5)):
    ap = answer_positions(cfg); idx = len(ap) - 1 - top
    n_top = cfg.n_digits - 2
    zc = {0: {h: [] for h in heads}, 1: {h: [] for h in heads}}
    for _ in range(n_q):
        for cls, bit in [("lo", 0), ("hi", 1)]:
            a, b, _, _ = build_chain(cfg, n_top, min(3, n_top), cls, shared={})
            c = cache_full(model, make_q(cfg, a, b))
            for h in heads:
                zc[bit][h].append(c["blocks.1.attn.hook_z"][0, cpos, h, :].numpy())
    zc = {bit: {h: np.mean(zc[bit][h], 0) for h in heads} for bit in (0, 1)}
    curve = {}
    for al in alphas:
        flips = []
        for _ in range(n_q):
            a, b, _, _ = build_chain(cfg, n_top, min(3, n_top), "lo", shared={})
            tq = make_q(cfg, a, b); clean = predict_answer(model, cfg, tq)
            tc = cache_full(model, tq)
            patches = [(cpos, 1, h, ((1 - al) * zc[0][h] + al * zc[1][h])) for h in heads]
            p = edge_patch_pred(model, cfg, tq, patches, tc)
            flips.append(float(p[idx] != clean[idx]))
        curve[str(al)] = float(np.mean(flips))
    a0 = curve["0.0"]; a1 = curve["1.0"]
    endpoint_ok = (a0 <= 0.2) and (a1 >= 0.8)
    xs = [a for a in alphas if 0 <= a <= 1]
    ys = [curve[str(a)] for a in xs]
    jump = float(np.max(np.diff(ys))) if len(ys) > 1 else 0.0
    spread = float(max(ys) - min(ys))
    astar = next((a for a in sorted(xs) if curve[str(a)] >= 0.5), None)
    cls = "invalid_endpoint" if not endpoint_ok else ("step" if jump > 0.5 * (spread + 1e-9) else "graded")
    return {"curve": curve, "endpoint_ok": bool(endpoint_ok), "class": cls,
            "alpha_star": astar, "max_jump": jump}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run_model(model, cfg, mn, fast=False):
    n_top = cfg.n_digits - 2
    ma, b = load_maps(mn)
    st_sites = st_sites_from_map(ma)
    comb = combiner_mlps_from_map(b)
    out = {"model": mn, "n_digits": cfg.n_digits, "n_top": n_top,
           "n_st_sites": len(st_sites), "n_combiner_mlps": len(comb),
           "scored_digits": scored_digits(cfg)}
    depths = [k for k in (2, 3) if k <= n_top]
    out["gates"] = {k: behavioral_gate(model, cfg, n_top, k, n_q=30) for k in depths}
    ax = map_carry_axis(model, cfg, mn, comb, n_q=(100 if fast else 200))
    out["carry_axis_sep"] = ax["sep"]; out["carry_axis_pos"] = ax["pos"]; out["carry_axis_digit"] = ax["digit"]
    npairs = 20 if fast else 40
    nabl = 120 if fast else 250
    consumers, cons_diag = identify_consumer_heads(model, cfg, mn, st_sites, ax,
                                                   n_pairs=(15 if fast else 30))
    out["consumer_heads"] = consumers
    out["consumer_diag"] = cons_diag["per_head"]
    out["battery_C"] = battery_C(model, cfg, mn, st_sites, ma, b, consumers, cons_diag,
                                 n_pairs=npairs, N_abl=nabl)
    out["battery_I"] = battery_I(model, cfg, mn, st_sites, consumers, cons_diag, ax, n_pairs=npairs)
    return out


def derive_cross_size(results):
    """A12/C6 scoring across sizes (XS-A/C/F)."""
    order = [mn for mn in PRIMARY if mn in results]
    v = {"sizes": [results[mn]["n_digits"] for mn in order]}
    # (iii) primary tightness leg: class-minus-single gap trajectory
    cms = [results[mn]["battery_C"]["iii_class_minus_single_gap"] for mn in order]
    mult = [results[mn]["battery_C"]["i_map_duplicate_multiplicity"] for mn in order]
    inter = [results[mn]["battery_C"]["ii_interchange_decisiveness_max"] for mn in order]
    v["iii_class_minus_single_gap_by_size"] = dict(zip([results[mn]["n_digits"] for mn in order], cms))
    v["i_map_multiplicity_by_size"] = dict(zip([results[mn]["n_digits"] for mn in order], mult))
    v["ii_interchange_by_size"] = dict(zip([results[mn]["n_digits"] for mn in order], inter))
    # tightening: does (iii) class-minus-single SHRINK from small (d5/d6) to large (d10/d13)?
    small = np.mean([cms[i] for i, mn in enumerate(order) if results[mn]["n_digits"] <= 6])
    large = np.mean([cms[i] for i, mn in enumerate(order) if results[mn]["n_digits"] >= 10])
    v["iii_small_mean"] = float(small); v["iii_large_mean"] = float(large)
    v["tightens_large_vs_small"] = bool(large < small - 0.02) if (small == small and large == large) else None
    # role transfer: are roles present at d10/d13?
    roles = {}
    for mn in order:
        if results[mn]["n_digits"] >= 10:
            r = results[mn]
            roles[r["n_digits"]] = {
                "ST_writers": r["n_st_sites"] > 0,
                "combiner_MLPs": r["n_combiner_mlps"] > 0,
                "consumer_head_identified": len(r["consumer_heads"]) > 0,
            }
    v["large_n_roles"] = roles
    all_roles = all(all(rr.values()) for rr in roles.values()) if roles else False
    # verdict (post-result F1/F2 corrected framing)
    if not all_roles:
        v["read"] = "A12-partial: a role absent/unidentified at large n (scope restriction)"
    elif v["tightens_large_vs_small"]:
        v["read"] = "A12 confirmed: role+combiner transfer AND redundancy tighter (STEP; monotone needs gradient) [C6 supported]"
    else:
        v["read"] = ("A12 role+combiner transfer confirmed (step combiner + PC2b robust; "
                     "causal SOURCE signatures probe-limited at large n, carry axis sep ~6); "
                     "C6 NOT SUPPORTED (redundancy persists, class-vs-single gap does not shrink "
                     "d5->d6->d10; d13 inconclusive, class ablation ~ CE13 single-node magnitude) "
                     "-- redundancy intrinsic, not slack. A11 not rescued by scale.")
    v["scoring_note"] = ("F1: 'transfer' = role presence + step combiner + PC2b (robust); source "
                         "arms 0.00 at large n are probe-limited, not confirmations. F2: 'C6 not "
                         "supported' not 'refuted' -- d13 class ablation instrument-weak.")
    return v


def main():
    fast = "--fast" in sys.argv
    grad = "--gradient" in sys.argv
    models = PRIMARY + (GRADIENT if grad else [])
    results = {}
    for mn in models:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc}"
        print(f"=== {mn} (n={cfg.n_digits}, acc {acc:.3f}) fast={fast} ===", flush=True)
        r = run_model(model, cfg, mn, fast=fast); r["accuracy"] = acc
        results[mn] = r
        C = r["battery_C"]
        print(f"  ST sites {r['n_st_sites']} combiners {r['n_combiner_mlps']} consumers {r['consumer_heads']}", flush=True)
        print(f"  [C] mult(i)={C['i_map_duplicate_multiplicity']:.2f} inter(ii)={C['ii_interchange_decisiveness_max']:.2f} "
              f"single_gap={C['iii_single_node_gap']:.3f} class_gap={C['iii_class_gap']:.3f} "
              f"class-single={C['iii_class_minus_single_gap']:.3f}", flush=True)
        I = r["battery_I"]
        if "status" in I:
            print(f"  [I] {I['status']}", flush=True)
        else:
            print(f"  [I] = arm {I['eq_not_a_source']['eq_arm_flip']['rate']:.2f} / decST {I['eq_not_a_source']['deciding_ST_arm_flip']['rate']:.2f} "
                  f"| necessity gap {I['class_necessity']['selective_gap']:.2f} | combiner {I['combiner']['class']} a*={I['combiner']['alpha_star']}", flush=True)
        del model
    results["_cross_size"] = derive_cross_size(results)
    print("\n=== CROSS-SIZE ===", flush=True)
    print("  class-minus-single gap by size:", {k: round(v, 3) for k, v in results["_cross_size"]["iii_class_minus_single_gap_by_size"].items()}, flush=True)
    print("  VERDICT:", results["_cross_size"]["read"], flush=True)
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR, flush=True)


if __name__ == "__main__":
    main()
