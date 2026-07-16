"""Output encoding of the map-named ST/SA/SC nodes (study-node-output-encoding.md).

C5 step 1: characterize how each map-named output-only node (ST/SA/SC attention
heads named in the HF features.json) encodes its sub-task value at its WRITE
(head output / OV-projected residual), with nulls that separate "written here"
from "decodable at this position", and a matched-pair causal re-examination of
CE3.

Implements the pre-run design + Gate-1 amendments N-1..N-6:
  * N-1 Battery C fixes the exact local pair (Dn,D'n), toggles cin; locality null
  * N-2 attention-window check + OV-projected write (z @ W_O)
  * N-3 Battery P = matched-pair interchange (tristate_test), SA-stratified
  * N-4 same-position wrong-role baseline (+ cross-digit) for "encodes its tag"
  * N-5 A2 scoped out (post-attention write, not pre-MLP)
  * N-6 outside-view sweep recorded in the study note

CPU-only. Run:
    PYTHONPATH=. python3 scripts/node_output_encoding.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from scripts.confirm_st_node import (
    load_model, make_q, answer_positions, predict_answer, verify_accuracy,
    patched_prediction, _digits_to_int, tristate_test,
)

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-node-output-encoding")
os.makedirs(RESULT_DIR, exist_ok=True)
SEED = 20260716
RNG = np.random.default_rng(SEED)

# Map-named nodes from HF features.json (read 2026-07-16). (pos, layer, head, digit)
ST_NODES = {
    "add_d5_l2_h3_t15K_s372001": [(6,0,2,4),(9,0,1,2),(10,0,0,2),(10,0,1,1),
                                   (10,0,2,1),(11,0,2,0),(12,0,1,3),(12,1,2,4)],
    "add_d6_l2_h3_t20K_s173289": [(10,0,2,3),(11,0,2,2),(12,0,1,1),(12,0,2,0),
                                   (14,0,1,5),(14,0,2,4)],
}
SC_NODES = {
    "add_d5_l2_h3_t15K_s372001": [(13,0,0,3),(15,0,0,1),(16,0,0,0)],
    "add_d6_l2_h3_t20K_s173289": [(15,0,2,4),(16,0,2,3),(17,0,2,2),(19,0,2,0)],
}
SA_NODES = {  # SA may be shared across two heads (5-digit)
    "add_d5_l2_h3_t15K_s372001": [(13,0,1,4),(13,0,2,4),(14,0,1,3),(14,0,2,3),
                                   (15,0,1,2),(15,0,2,2),(16,0,1,1),(16,0,2,1),
                                   (17,0,1,0),(17,0,2,0)],
    "add_d6_l2_h3_t20K_s173289": [(15,0,1,5),(16,0,1,4),(17,0,1,3),(18,0,1,2),
                                   (19,0,1,1),(20,0,1,0)],
}
# CE3 SA head positive control for patching (pos,layer,head,digit)
CE3_SA_HEAD = {"add_d5_l2_h3_t15K_s372001": (14,0,1,3),
               "add_d6_l2_h3_t20K_s173289": (20,0,1,0)}


def digits_msb(x, nd):
    return [int(d) for d in str(x).zfill(nd)]


def sub_value(a, b, nd, n, task):
    da = digits_msb(a, nd); db = digits_msb(b, nd)
    s = da[nd - 1 - n] + db[nd - 1 - n]
    if task == "ST":
        return 0 if s <= 8 else (1 if s >= 10 else 2)
    if task == "SC":
        return 1 if s >= 10 else 0
    # SA = the answer digit (Dn+D'n+cin)%10
    cin = 0; 
    for k in range(n):
        sk = da[nd-1-k] + db[nd-1-k] + cin
        cin = 1 if sk >= 10 else 0
    return (s + cin) % 10


def chance(task):
    return {"ST": 1/3, "SC": 0.5, "SA": 0.10}[task]


def fit(X, y):
    return LogisticRegression(max_iter=2000, C=0.5).fit(X, y)


def bacc(clf, X, y):
    return float(balanced_accuracy_score(y, clf.predict(X)))


def balance(y):
    classes = np.unique(y); per = max(np.bincount(y, minlength=int(classes.max())+1))
    idx = []
    for c in classes:
        ci = np.where(y == c)[0]; idx.extend(RNG.choice(ci, size=per, replace=len(ci) < per))
    return np.array(idx)


def build_random(cfg, n_q):
    """Random accurate additions."""
    nd = cfg.n_digits; lim = 10 ** nd
    return [(int(RNG.integers(0, lim // 2)), int(RNG.integers(0, lim // 2))) for _ in range(n_q)]


# ===========================================================================
# Battery E: full-space output encoding + N-4 baselines
# ===========================================================================

def cache_head_out(model, cfg, qs, positions_layers):
    """Return {(pos,layer,head): [n_q, d_head]} head outputs (hook_z) and the
    OV-projected write {(pos,layer,head): [n_q, d_model]}."""
    hooks = sorted({f"blocks.{L}.attn.hook_z" for (_, L, _) in positions_layers})
    z = {k: [] for k in positions_layers}
    ov = {k: [] for k in positions_layers}
    WO = {L: model.blocks[L].attn.W_O for L in {L for (_, L, _) in positions_layers}}
    for (a, b) in qs:
        q = make_q(cfg, a, b)
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm in hooks)
        for (pos, L, head) in positions_layers:
            zz = c[f"blocks.{L}.attn.hook_z"][0, pos, head, :]
            z[(pos, L, head)].append(zz.numpy())
            ov[(pos, L, head)].append((zz @ WO[L][head]).detach().numpy())
    return ({k: np.array(v) for k, v in z.items()}, {k: np.array(v) for k, v in ov.items()})


def battery_E(model, cfg, model_name, qs):
    nd = cfg.n_digits
    node_sets = {"ST": ST_NODES, "SC": SC_NODES, "SA": SA_NODES}
    # all nodes to cache
    all_nodes = []
    for task, ns in node_sets.items():
        for (pos, L, head, dig) in ns[model_name]:
            all_nodes.append((pos, L, head))
    all_nodes = sorted(set(all_nodes))
    z, ov = cache_head_out(model, cfg, qs, all_nodes)
    ntr = int(0.7 * len(qs)); tr = slice(0, ntr); te = slice(ntr, len(qs))
    out = {}
    for task, ns in node_sets.items():
        for (pos, L, head, dig) in ns[model_name]:
            y = np.array([sub_value(a, b, nd, dig, task) for (a, b) in qs])
            if len(np.unique(y[:ntr])) < 2:
                continue
            X = z[(pos, L, head)]
            bi = balance(y[tr])
            acc = bacc(fit(X[tr][bi], y[tr][bi]), X[te], y[te])
            # permutation null
            yp = y.copy(); RNG.shuffle(yp)
            bip = balance(yp[tr])
            acc_perm = bacc(fit(X[tr][bip], yp[tr][bip]), X[te], yp[te])
            # cross-digit wrong-node baseline: same task+role, DIFFERENT digit's node
            others = [(p2, L2, h2, d2) for (p2, L2, h2, d2) in ns[model_name] if d2 != dig]
            cross = float("nan")
            if others:
                p2, L2, h2, d2 = others[0]
                X2 = z[(p2, L2, h2)]
                cross = bacc(fit(X2[tr][bi], y[tr][bi]), X2[te], y[te])
            # N-4 same-position wrong-role baseline: co-located head NOT tagged for this
            tagged_heads = {h for (p, LL, h, d) in ns[model_name] if p == pos and LL == L}
            wrong_role = float("nan")
            for h2 in range(cfg.n_heads):
                if h2 not in tagged_heads:
                    z2, _ = cache_head_out(model, cfg, qs, [(pos, L, h2)])
                    Xw = z2[(pos, L, h2)]
                    wrong_role = bacc(fit(Xw[tr][bi], y[tr][bi]), Xw[te], y[te])
                    break
            encodes = (acc >= chance(task) + 0.20 and acc >= 2 * max(cross if cross==cross else 0, chance(task)) - chance(task)
                       and acc > acc_perm + 0.10
                       and (wrong_role != wrong_role or acc >= wrong_role + 0.15))
            out[f"P{pos}L{L}H{head}"] = {
                "task": task, "digit": dig, "acc": acc, "chance": chance(task),
                "perm_null": acc_perm, "cross_digit_baseline": cross,
                "same_pos_wrong_role_baseline": wrong_role, "encodes": bool(encodes)}
    return out


# ===========================================================================
# Battery C: cascade locality (fixed pair, toggle cin) on the OV write  [N-1/N-2]
# ===========================================================================

def build_fixed_pair_cin(cfg, n, a_n, b_n, cin_target):
    """Build a question with digit n's operands FIXED at (a_n,b_n) and the carry
    INTO digit n forced to cin_target via digit n-1 (>=10 or <10), other lower
    digits no-carry, higher digits 0."""
    nd = cfg.n_digits
    d1 = [0]*nd; d2 = [0]*nd; idx = nd-1-n
    d1[idx] = a_n; d2[idx] = b_n
    if n >= 1:
        ikm = nd-1-(n-1)
        if cin_target == 1:
            while True:
                x = int(RNG.integers(0,10)); y = int(RNG.integers(0,10))
                if x+y >= 10: break
        else:
            x = int(RNG.integers(0,10)); y = int(RNG.integers(0,10-x))
        d1[ikm] = x; d2[ikm] = y
        for k in range(n-1):
            ik = nd-1-k; u = int(RNG.integers(0,10)); v = int(RNG.integers(0,10-u))
            d1[ik] = u; d2[ik] = v
    return _digits_to_int(d1), _digits_to_int(d2)


def _cin_locality(model, cfg, pos, L, head, dig, pair_kind, n_pairs):
    """Measure the OV write's dependence on cin at a fixed local pair. Returns
    cin_effect / null_effect (norms) and their ratio. pair_kind: 'definite' (sum<=8,
    cin cannot change carry-out) or 'U' (sum=9, cin DECIDES carry-out)."""
    nd = cfg.n_digits
    WO = model.blocks[L].attn.W_O[head]
    a_n, b_n = pick_pair(nd, dig, pair_kind)
    w0 = []; w1 = []; wn = []
    for _ in range(n_pairs):
        a0, b0 = build_fixed_pair_cin(cfg, dig, a_n, b_n, 0)
        a1, b1 = build_fixed_pair_cin(cfg, dig, a_n, b_n, 1)
        an, bn = build_fixed_pair_cin(cfg, dig, a_n, b_n, 0)
        w0.append(ov_write(model, cfg, a0, b0, pos, L, head, WO))
        w1.append(ov_write(model, cfg, a1, b1, pos, L, head, WO))
        wn.append(ov_write(model, cfg, an, bn, pos, L, head, WO))
    w0 = np.array(w0); w1 = np.array(w1); wn = np.array(wn)
    cin_eff = float(np.linalg.norm(w0.mean(0) - w1.mean(0)))
    null_eff = float(np.linalg.norm(w0.mean(0) - wn.mean(0)))
    return {"cin_effect_norm": cin_eff, "null_effect_norm": null_eff,
            "cin_over_null": float(cin_eff / (null_eff + 1e-9)),
            "pair": [int(a_n), int(b_n)]}


def battery_C(model, cfg, model_name, n_pairs=200):
    """N-8-corrected: for each ST node, measure the OV write's cin-dependence on
    BOTH a definite pair (cin can't change carry-out) AND a U pair (sum=9, cin
    decides carry-out — the case where cin genuinely matters). Report cin/null
    RATIO (not a slack-dominated threshold). Positive control: an answer-position
    node whose write SHOULD depend on cin (the CE3 SA head reads the resolved digit).
    Locality verdict: cin/null ratio near 1 on the U pair = local write."""
    res = {}
    # positive control: does the metric register a cin-dependence where one must
    # exist? Use the answer-position SA head reading digit `dig` (its answer digit
    # (Dn+D'n+cin)%10 depends on cin, so its OV write should vary with cin).
    sp, sl, sh, sd = CE3_SA_HEAD[model_name]
    res["_pos_control_SA_head"] = {"node": f"P{sp}L{sl}H{sh}", "digit": sd,
                                   **_cin_locality(model, cfg, sp, sl, sh, sd, "U", n_pairs)}
    for (pos, L, head, dig) in ST_NODES[model_name]:
        if dig < 1:
            continue
        attn_lowcarry = attn_mass_on_digit(model, cfg, pos, L, head, dig-1, n_q=20)
        defn = _cin_locality(model, cfg, pos, L, head, dig, "definite", n_pairs)
        u = _cin_locality(model, cfg, pos, L, head, dig, "U", n_pairs)
        # local write if the write barely moves with cin on the U pair (ratio ~1)
        res[f"P{pos}L{L}H{head}"] = {
            "digit": dig, "attn_on_lower_carry_digit": attn_lowcarry,
            "definite_cin_over_null": defn["cin_over_null"],
            "U_cin_over_null": u["cin_over_null"],
            "U_cin_effect_norm": u["cin_effect_norm"], "U_null_effect_norm": u["null_effect_norm"],
            "local_write": bool(u["cin_over_null"] <= 2.0)}
    return res


def pick_pair(nd, n, kind):
    if kind == "definite":  # sum != 9, use a no-carry definite (sum<=8)
        while True:
            a = int(RNG.integers(0,9)); b = int(RNG.integers(0,9))
            if a+b <= 8: return a, b
    a = int(RNG.integers(0,10)); return a, 9-a


def ov_write(model, cfg, a, b, pos, L, head, WO):
    q = make_q(cfg, a, b)
    with torch.no_grad():
        _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == f"blocks.{L}.attn.hook_z")
    return (c[f"blocks.{L}.attn.hook_z"][0, pos, head, :] @ WO).detach().numpy()


def attn_mass_on_digit(model, cfg, pos, L, head, n, n_q=20):
    nd = cfg.n_digits; dn = nd-1-n; dpn = 2*nd-n
    mass = []
    for _ in range(n_q):
        a = int(RNG.integers(0,10**nd//2)); b = int(RNG.integers(0,10**nd//2))
        q = make_q(cfg, a, b)
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == f"blocks.{L}.attn.hook_pattern")
        p = c[f"blocks.{L}.attn.hook_pattern"][0, head, pos, :].numpy()
        mass.append(float(p[dn] + p[dpn]))
    return float(np.mean(mass))


# ===========================================================================
# Battery P: matched-pair causal re-examination of CE3  [N-3]
# ===========================================================================

def battery_P(model, cfg, model_name, n_pairs=40):
    """Matched-pair interchange at each map-named ST node via the tristate_test
    (fix Dn+D'n=9, toggle lower carry -> resolved ST outcome flips). Measures the
    tagged answer-digit flip. Positive control: CE3 SA head must flip A_n."""
    res = {}
    for (pos, L, head, dig) in ST_NODES[model_name]:
        if dig < 1:
            continue
        def hb(pos=pos, L=L, head=head):
            return [{"name": f"blocks.{L}.attn.hook_z", "pos": pos, "head": head}]
        r = tristate_test(model, cfg, dig, hb, n_pairs=n_pairs)
        res[f"P{pos}L{L}H{head}"] = {"digit": dig, "tristate_A_n+1_flip": r["A_n+1_flip"]}
    # positive control
    sp, sl, sh, sd = CE3_SA_HEAD[model_name]
    def hbc():
        return [{"name": f"blocks.{sl}.attn.hook_z", "pos": sp, "head": sh}]
    from scripts.confirm_st_node import flip_signature
    sig = flip_signature(model, cfg, sd, hbc, "local", n_pairs=40, src_cls=1, tgt_cls=0)
    res["_control_CE3_SA_head"] = {"node": f"P{sp}L{sl}H{sh}", "A_n_flip": sig["A_n_flip"]}
    return res


# ===========================================================================
# driver
# ===========================================================================

def _mean_ablate_acc(model, cfg, pos, L, head, qs, ap):
    zs = []
    for a, b in build_random(cfg, 150):
        q = make_q(cfg, a, b)
        with torch.no_grad():
            _, c = model.run_with_cache(q.unsqueeze(0), names_filter=lambda nm: nm == f"blocks.{L}.attn.hook_z")
        zs.append(c[f"blocks.{L}.attn.hook_z"][0, pos, head, :].numpy())
    zmean = torch.tensor(np.mean(zs, axis=0))
    ok = 0
    for a, b in qs:
        q = make_q(cfg, a, b)
        def hook(act, hook): act[:, pos, head, :] = zmean; return act
        with torch.no_grad():
            lg = model.run_with_hooks(q.unsqueeze(0), fwd_hooks=[(f"blocks.{L}.attn.hook_z", hook)])
        if torch.equal(lg[0, [p - 1 for p in ap]].argmax(-1), q[ap]):
            ok += 1
    return ok / len(qs)


def battery_Ab(model, cfg, model_name, N=300):
    """N-3 companion + N-8: mean-ablation impact per map-named ST node vs a
    RANDOM/untagged same-position-region head BASELINE (F1). A node's ablation
    impact counts as causal only if it exceeds the untagged-head baseline
    distribution. Reconciles interchange=0.00 (redundancy) with real ablation."""
    ap = answer_positions(cfg); nd = cfg.n_digits
    res = {}
    qs = build_random(cfg, N)
    clean = sum(int(torch.equal(predict_answer(model, cfg, make_q(cfg, a, b)),
                                make_q(cfg, a, b)[ap])) for a, b in qs) / N
    st_set = {(p, L, h) for (p, L, h, d) in ST_NODES[model_name]}
    # baseline: mean-ablation impact of UNTAGGED L0 heads (not any Algo node) at
    # the same question-side positions as the ST nodes.
    tagged_all = st_set | {(p, L, h) for (p, L, h, d) in SC_NODES[model_name]} \
                 | {(p, L, h) for (p, L, h, d) in SA_NODES[model_name]}
    baseline_impacts = []
    st_positions = sorted({p for (p, L, h) in st_set})
    for p in st_positions:
        for h in range(cfg.n_heads):
            if (p, 0, h) in tagged_all:
                continue
            acc = _mean_ablate_acc(model, cfg, p, 0, h, qs, ap)
            baseline_impacts.append(clean - acc)
    base_mean = float(np.mean(baseline_impacts)) if baseline_impacts else 0.0
    base_max = float(np.max(baseline_impacts)) if baseline_impacts else 0.0
    res["_untagged_head_baseline"] = {"mean_impact": base_mean, "max_impact": base_max,
                                      "n_heads": len(baseline_impacts)}
    for (pos, L, head, dig) in ST_NODES[model_name]:
        acc = _mean_ablate_acc(model, cfg, pos, L, head, qs, ap)
        impact = clean - acc
        res[f"P{pos}L{L}H{head}"] = {"digit": dig, "ablated_acc": acc, "clean_acc": clean,
                                     "impact": impact,
                                     "above_untagged_baseline": bool(impact > base_max)}
    return res


def run_model(model, cfg, model_name):
    qs = build_random(cfg, 4000)
    out = {"model": model_name}
    out["battery_E"] = battery_E(model, cfg, model_name, qs)
    out["battery_C"] = battery_C(model, cfg, model_name, n_pairs=150)
    out["battery_P"] = battery_P(model, cfg, model_name, n_pairs=40)
    out["battery_Ab"] = battery_Ab(model, cfg, model_name, N=300)
    return out


def main():
    results = {}
    for mn in ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289"]:
        model, cfg = load_model(mn)
        acc = verify_accuracy(model, cfg, n=64)
        assert acc > 0.99, f"{mn} acc {acc}"
        print(f"=== {mn} (acc {acc:.3f}) ===", flush=True)
        r = run_model(model, cfg, mn); r["accuracy"] = acc
        results[mn] = r
        # console summary
        enc = r["battery_E"]
        print("  [E] encoding (node: task digit acc/chance encodes):")
        for node, d in enc.items():
            print(f"    {node}: {d['task']}{d['digit']} acc={d['acc']:.2f}/{d['chance']:.2f} "
                  f"perm={d['perm_null']:.2f} xdig={d['cross_digit_baseline']:.2f} "
                  f"wrole={d['same_pos_wrong_role_baseline']:.2f} -> {d['encodes']}")
        print("  [C] ST-node cascade locality (U-pair cin/null ratio, local?):")
        for node, d in r["battery_C"].items():
            if node.startswith("_"):
                print(f"    POS-CONTROL {d['node']} d{d['digit']} U_cin/null={d['cin_over_null']:.1f} (should be >>1)")
            else:
                print(f"    {node} d{d['digit']}: U_cin/null={d['U_cin_over_null']:.1f} "
                      f"def_cin/null={d['definite_cin_over_null']:.1f} attn_low={d['attn_on_lower_carry_digit']:.2f} "
                      f"-> local={d['local_write']}")
        print("  [P] CE3 re-exam (tristate flip):")
        for node, d in r["battery_P"].items():
            if node.startswith("_"):
                print(f"    control {d['node']} A_n_flip={d['A_n_flip']:.2f}")
            else:
                print(f"    {node} d{d['digit']} tristate_flip={d['tristate_A_n+1_flip']:.2f}")
        print("  [Ab] mean-ablation impact vs untagged-head baseline:")
        for node, d in r["battery_Ab"].items():
            if node.startswith("_"):
                print(f"    BASELINE untagged heads: mean={d['mean_impact']:.3f} max={d['max_impact']:.3f} (n={d['n_heads']})")
            else:
                print(f"    {node} d{d['digit']} impact={d['impact']:.3f} above_baseline={d['above_untagged_baseline']}")
        del model
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
