"""Confirm an ST compute node via path-patching (study-confirm-st-node.md).

Causal interchange-intervention locator for the tri-state carry (ST) node.
A node computes ST_n if patching its activation from a source question with a
DIFFERENT ST_n class into a target question flips answer digit A_{n+1} (carry
signature) while leaving A_n unchanged; an SA node instead flips A_n.

Implements amendments A-1..A-8:
  * node-level positive control (single-head z patch on a confirmed SA head) [A-1]
  * local (no-lower-carry, 0<->1) and cascade (U-exercising) batteries [A-2]
  * signature -> {R-ST,R-SV,R-SA,mixed,none} decision table + attention gate [A-3]
  * SA-stratified counterfactuals [A-4]
  * z-alone, joint z+MLP, and resid patch arms [A-5]
  * per-cell null, control-tied bar [A-6]

CPU-only. Run:
    PYTHONPATH=. python3 scripts/confirm_st_node.py control
    PYTHONPATH=. python3 scripts/confirm_st_node.py models
    PYTHONPATH=. python3 scripts/confirm_st_node.py all
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

RESULT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "results", "study-confirm-st-node")
os.makedirs(RESULT_DIR, exist_ok=True)
RNG = np.random.default_rng(20260714)
torch.manual_seed(0)


# ---------------------------------------------------------------------------
# model + token helpers
# ---------------------------------------------------------------------------

def load_model(model_name):
    from huggingface_hub import hf_hub_download
    from quanta_maths.maths_config import MathsConfig
    from transformer_lens import HookedTransformer
    cfg = MathsConfig(); cfg.set_model_names(model_name)
    htc = cfg.get_HookedTransformerConfig(); htc.device = "cpu"; htc.init_weights = False
    model = HookedTransformer(htc)
    p = hf_hub_download(repo_id="PhilipQuirke/VerifiedArithmetic", filename=f"{model_name}.pth")
    sd = torch.load(p, map_location="cpu")
    if "model" in sd and "embed.W_E" not in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg


def make_q(cfg, a, b):
    from quanta_maths.maths_utilities import make_a_maths_question_and_answer
    from quanta_maths.maths_constants import MathsToken
    q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
    make_a_maths_question_and_answer(cfg, q, 0, a, b, MathsToken.PLUS)
    return q[0]


def answer_positions(cfg):
    na = cfg.n_digits + 2
    return list(range(cfg.n_ctx - na, cfg.n_ctx))  # sign, A_top..A0


def predict_answer(model, cfg, q):
    """Return predicted answer tokens (na of them) for a single question row."""
    with torch.no_grad():
        logits = model(q.unsqueeze(0))
    ap = answer_positions(cfg)
    return logits[0, [p - 1 for p in ap]].argmax(-1)  # (na,)


def verify_accuracy(model, cfg, n=64):
    lim = 10 ** cfg.n_digits
    ok = 0
    for _ in range(n):
        a = int(RNG.integers(0, lim // 2)); b = int(RNG.integers(0, lim // 2))
        q = make_q(cfg, a, b)
        pred = predict_answer(model, cfg, q)
        true = q[answer_positions(cfg)]
        if torch.equal(pred, true):
            ok += 1
    return ok / n


# ---------------------------------------------------------------------------
# digit-n stimulus construction (A-2, A-7)
# ---------------------------------------------------------------------------

def _digits_to_int(digs):
    return int("".join(str(d) for d in digs))


def make_pair(cfg, n, cls_target, lower_carry):
    """Build ONE question with digit n in ST class cls_target
    (0 = no-carry sum<=8, 1 = carry sum>=10, 'U' = sum==9), controlling lower
    carries. Returns (a, b, sa_n) where sa_n = (Dn+D'n)%10.

    lower_carry: if False, lower digits sum<10 (no carry into n). If True,
    force a carry to propagate into n from digit n-1 (cascade battery)."""
    nd = cfg.n_digits
    d1 = [0] * nd; d2 = [0] * nd  # index 0 = most significant
    idx = nd - 1 - n              # array index for digit n
    # choose Dn, D'n for the target class
    while True:
        a = int(RNG.integers(0, 10)); b = int(RNG.integers(0, 10))
        s = a + b
        if cls_target == 0 and s <= 8: break
        if cls_target == 1 and s >= 10: break
        if cls_target == "U" and s == 9: break
    d1[idx] = a; d2[idx] = b
    # lower digits
    for k in range(n):
        ik = nd - 1 - k
        if lower_carry and k == n - 1:
            # force digit n-1 to carry (sum >= 10)
            while True:
                x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
                if x + y >= 10:
                    break
            d1[ik] = x; d2[ik] = y
        else:
            x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))  # no carry
            d1[ik] = x; d2[ik] = y
    # higher digits: small fixed filler 0 (avoid overflow / extra carries)
    for k in range(n + 1, nd):
        d1[nd - 1 - k] = 0; d2[nd - 1 - k] = 0
    a_int = _digits_to_int(d1); b_int = _digits_to_int(d2)
    return a_int, b_int, (a + b) % 10


# ---------------------------------------------------------------------------
# patching
# ---------------------------------------------------------------------------

def patched_prediction(model, cfg, source_q, target_q, hooks_spec):
    """Run source to cache activations, then run target with those activations
    patched at the positions/heads in hooks_spec. hooks_spec: list of dicts
    {name, pos, head(optional)}. Returns predicted answer tokens for target."""
    with torch.no_grad():
        _, src_cache = model.run_with_cache(source_q.unsqueeze(0))
    fwd = []
    for hs in hooks_spec:
        name = hs["name"]; pos = hs["pos"]; head = hs.get("head")
        src_act = src_cache[name]
        def make_hook(name=name, pos=pos, head=head, src_act=src_act):
            def hook(act, hook):
                if head is not None:
                    act[:, pos, head, :] = src_act[:, pos, head, :]
                else:
                    act[:, pos, :] = src_act[:, pos, :]
                return act
            return hook
        fwd.append((name, make_hook()))
    with torch.no_grad():
        logits = model.run_with_hooks(target_q.unsqueeze(0), fwd_hooks=fwd)
    ap = answer_positions(cfg)
    return logits[0, [p - 1 for p in ap]].argmax(-1)


def flip_signature(model, cfg, n, hooks_builder, battery, n_pairs=60,
                   src_cls=1, tgt_cls=0):
    """Over n_pairs source/target pairs (source ST class src_cls, target tgt_cls),
    patch node into target and measure per-answer-digit flip rate vs the target's
    clean prediction. Returns dict: per-digit flip rate, A_n flip, A_{n+1} flip,
    plus SA-change stratification."""
    ap = answer_positions(cfg)
    na = len(ap)
    # answer digit index within ap: A_k is at ap position (na-1-k) since ap is
    # [sign, A_top, ..., A0]; A0 is last. A_k sits at ap[-1-k].
    flips = np.zeros(na)
    an_flips = []; an1_flips = []; sa_changed = []
    lower_carry = (battery == "cascade")
    for _ in range(n_pairs):
        _sa_t = None
        # target
        ta, tb, tsa = make_pair(cfg, n, tgt_cls, lower_carry)
        sa, sb, ssa = make_pair(cfg, n, src_cls, lower_carry)
        tq = make_q(cfg, ta, tb); sq = make_q(cfg, sa, sb)
        clean = predict_answer(model, cfg, tq)
        hooks = hooks_builder()
        patched = patched_prediction(model, cfg, sq, tq, hooks)
        diff = (patched != clean).numpy().astype(float)
        flips += diff
        # A_k at ap index -1-k
        def ak_flip(k):
            if k > cfg.n_digits: return 0.0
            return float(diff[na - 1 - k]) if (na - 1 - k) >= 0 else 0.0
        an_flips.append(ak_flip(n))
        an1_flips.append(ak_flip(n + 1))
        sa_changed.append(1.0 if ssa != tsa else 0.0)
    flips /= n_pairs
    an_flips = np.array(an_flips); an1_flips = np.array(an1_flips); sa_changed = np.array(sa_changed)
    # A_n flip conditioned on SA_n having changed (A-4)
    m = sa_changed > 0
    an_flip_given_sa = float(an_flips[m].mean()) if m.any() else float('nan')
    return {
        "per_digit_flip": flips.tolist(),  # [sign, A_top..A0]
        "A_n_flip": float(an_flips.mean()),
        "A_n+1_flip": float(an1_flips.mean()),
        "A_n_flip_given_SA_change": an_flip_given_sa,
        "frac_SA_changed": float(sa_changed.mean()),
    }


def tristate_test(model, cfg, n, hooks_builder, n_pairs=60):
    """GENUINE tri-state (U) test [Gate-2 F1 fix]. Hold digit n at sum==9 (U) in
    BOTH source and target, and toggle the LOWER carry: source has a lower carry
    (so U resolves to carry-out=1), target has no lower carry (U resolves to
    carry-out=0). Patch the node from source->target. If the node computes the
    tri-state ST (which must consult the lower carry to resolve U), the patch
    flips A_{n+1}. A purely BINARY make-carry node keyed only on Dn+D'n>=10 has
    the SAME value in both (U is sum==9 < 10 -> 0 always) and will NOT flip.

    Requires n>=1 (need a lower digit to carry). Returns A_{n+1} flip rate."""
    if n < 1:
        return {"A_n+1_flip": float("nan"), "note": "n<1: no lower digit"}
    ap = answer_positions(cfg); na = len(ap)
    flips = []
    nd = cfg.n_digits
    for _ in range(n_pairs):
        # both: digit n has sum==9 (U). source: lower carry; target: no lower carry.
        while True:
            a = int(RNG.integers(0, 10)); b = 9 - a
            if 0 <= b <= 9:
                break
        def build(lower_carry):
            d1 = [0] * nd; d2 = [0] * nd; idx = nd - 1 - n
            d1[idx] = a; d2[idx] = b
            ik = nd - 1 - (n - 1)
            if lower_carry:
                while True:
                    x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
                    if x + y >= 10: break
                d1[ik] = x; d2[ik] = y
            else:
                x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
                d1[ik] = x; d2[ik] = y
            for k in range(n - 1):
                ik2 = nd - 1 - k
                x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
                d1[ik2] = x; d2[ik2] = y
            return _digits_to_int(d1), _digits_to_int(d2)
        sa, sb = build(True)    # source: U resolves to carry
        ta, tb = build(False)   # target: U resolves to no-carry
        sq = make_q(cfg, sa, sb); tq = make_q(cfg, ta, tb)
        clean = predict_answer(model, cfg, tq)
        patched = patched_prediction(model, cfg, sq, tq, hooks_builder())
        diff = (patched != clean).numpy().astype(float)
        idx_an1 = na - 1 - (n + 1)
        flips.append(float(diff[idx_an1]) if 0 <= idx_an1 < na else 0.0)
    return {"A_n+1_flip": float(np.mean(flips)), "n_pairs": n_pairs}


def same_class_null(model, cfg, n, hooks_builder, battery, cls=0, n_pairs=60):
    """Patch source->target where BOTH share the same ST class AND the same
    (Dn,D'n) at digit n (so SA_n and ST_n are identical; only the other digits'
    noise differs). This is the true false-positive floor: a node whose patch
    changes an answer digit here is responding to irrelevant context, not to
    ST_n. Amendment A-6 (per-cell null)."""
    ap = answer_positions(cfg); na = len(ap)
    an_flips = []; an1_flips = []
    lower_carry = (battery == "cascade")
    # fix one (Dn,D'n) of the given class
    while True:
        a0 = int(RNG.integers(0, 10)); b0 = int(RNG.integers(0, 10)); s = a0 + b0
        if (cls == 0 and s <= 8) or (cls == 1 and s >= 10):
            break
    for _ in range(n_pairs):
        # both source and target have digit-n = (a0,b0); other digits vary
        def build():
            nd = cfg.n_digits; d1 = [0] * nd; d2 = [0] * nd
            idx = nd - 1 - n; d1[idx] = a0; d2[idx] = b0
            for k in range(n):
                ik = nd - 1 - k
                if lower_carry and k == n - 1:
                    while True:
                        x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10))
                        if x + y >= 10: break
                    d1[ik] = x; d2[ik] = y
                else:
                    x = int(RNG.integers(0, 10)); y = int(RNG.integers(0, 10 - x))
                    d1[ik] = x; d2[ik] = y
            return _digits_to_int(d1), _digits_to_int(d2)
        ta, tb = build(); sa, sb = build()
        tq = make_q(cfg, ta, tb); sq = make_q(cfg, sa, sb)
        clean = predict_answer(model, cfg, tq)
        patched = patched_prediction(model, cfg, sq, tq, hooks_builder())
        diff = (patched != clean).numpy().astype(float)
        def ak_flip(k):
            return float(diff[na - 1 - k]) if 0 <= (na - 1 - k) < na and k <= cfg.n_digits else 0.0
        an_flips.append(ak_flip(n)); an1_flips.append(ak_flip(n + 1))
    return {"A_n_flip": float(np.mean(an_flips)), "A_n+1_flip": float(np.mean(an1_flips))}


# ---------------------------------------------------------------------------
# attention-read gate (A-3)
# ---------------------------------------------------------------------------

def operand_attention(model, cfg, pos, layer, head, n, n_q=40):
    dn_pos = cfg.n_digits - 1 - n
    dp_pos = 2 * cfg.n_digits - n
    mass = []
    for _ in range(n_q):
        a, b, _ = make_pair(cfg, n, int(RNG.integers(0, 2)), False)
        q = make_q(cfg, a, b)
        with torch.no_grad():
            _, cache = model.run_with_cache(q.unsqueeze(0))
        patt = cache[f"blocks.{layer}.attn.hook_pattern"][0, head, pos, :].numpy()
        mass.append(patt[dn_pos] + patt[dp_pos])
    return float(np.mean(mass))


# ---------------------------------------------------------------------------
# classification (A-3, A-6)
# ---------------------------------------------------------------------------

def classify_base(sig_local, null, attn_mass, bar):
    """Base signature without the tri-state distinction (F1). Returns
    'carry' (flips A_{n+1}, reads operands), 'R-SV/cascade' (flips A_{n+1}, does
    not read operands), 'R-SA' (flips A_n), or 'R-none'."""
    an = sig_local["A_n_flip"]; an1 = sig_local["A_n+1_flip"]
    an_sig = an >= bar and an > null["A_n_flip"] + 0.2
    an1_sig = an1 >= bar and an1 > null["A_n+1_flip"] + 0.2
    if an_sig and an1_sig:
        return "R-mixed"
    if an1_sig and not an_sig:
        return "carry" if attn_mass >= 0.30 else "R-SV/cascade"
    if an_sig and not an1_sig:
        return "R-SA"
    return "R-none"


def classify(sig_local, sig_cascade, null, attn_mass, bar):
    an = sig_local["A_n_flip"]; an1 = sig_local["A_n+1_flip"]
    an_null = null["A_n_flip"]; an1_null = null["A_n+1_flip"]
    casc_an1 = sig_cascade["A_n+1_flip"]
    an1_sig = an1 >= bar and an1 > an1_null + 0.2
    an_sig = an >= bar and an > an_null + 0.2
    tri = casc_an1 >= bar  # tri-state causal in cascade battery
    if an_sig and an1_sig:
        return "R-mixed"
    if an1_sig and not an_sig:
        if attn_mass >= 0.30:
            return "R-ST(tri)" if tri else "R-SC(binary)"
        return "R-SV/cascade"
    if an_sig and not an1_sig:
        return "R-SA"
    return "R-none"


# ---------------------------------------------------------------------------
# positive control (A-1)
# ---------------------------------------------------------------------------

CONFIRMED_SA_HEAD = {  # from pair-sum study (answer-position operand-fetch)
    "add_d5_l2_h3_t15K_s372001": (14, 0, 1, 3),   # pos,layer,head,digit
    "add_d6_l2_h3_t20K_s173289": (20, 0, 1, 0),
}


def run_control(model_name):
    model, cfg = load_model(model_name)
    acc = verify_accuracy(model, cfg)
    pos, layer, head, digit = CONFIRMED_SA_HEAD[model_name]
    # node-level control: single-head z patch on a confirmed SA head; should
    # flip A_n (SA signature) between different-SA sources.
    def hb():
        return [{"name": f"blocks.{layer}.attn.hook_z", "pos": pos, "head": head}]
    sig = flip_signature(model, cfg, digit, hb, "local", n_pairs=60, src_cls=1, tgt_cls=0)
    null = same_class_null(model, cfg, digit, hb, "local", cls=0, n_pairs=60)
    # coarse harness sanity: patch resid at '=' should flip A_{n+1}
    eq_pos = 2 * cfg.n_digits + 1
    def hb_resid():
        return [{"name": "blocks.0.hook_resid_post", "pos": eq_pos}]
    sig_resid = flip_signature(model, cfg, digit, hb_resid, "local", n_pairs=40, src_cls=1, tgt_cls=0)
    out = {"model": model_name, "accuracy": acc,
           "node_level_control": {"node": f"P{pos}L{layer}H{head}", "digit": digit,
                                  "A_n_flip": sig["A_n_flip"], "A_n+1_flip": sig["A_n+1_flip"],
                                  "same_class_null_A_n": null["A_n_flip"]},
           "coarse_resid_eq_control": {"A_n_flip": sig_resid["A_n_flip"],
                                       "A_n+1_flip": sig_resid["A_n+1_flip"],
                                       "any_flip": float(np.mean(sig_resid["per_digit_flip"]))}}
    with open(os.path.join(RESULT_DIR, f"control_{model_name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    del model
    return out


# ---------------------------------------------------------------------------
# real models
# ---------------------------------------------------------------------------

CANDIDATES = {
    # Paper-2 named candidates PLUS the carry-computing head-0 answer-position
    # nodes discovered by the A_{n+1}-flip sweep (2026-07-14). H0 = carry/ST,
    # H1 = base-add/SA at answer positions.
    "add_d5_l2_h3_t15K_s372001": [(8, 0, 1), (9, 0, 1), (11, 0, 2), (14, 0, 1),
                                   (13, 0, 0), (14, 0, 0), (15, 0, 0), (16, 0, 0)],
    # 6-digit rep: carry nodes found by the A_{n+1} sweep (head 2 here)
    "add_d6_l2_h3_t20K_s173289": [(14, 0, 1), (15, 0, 2), (16, 0, 2), (17, 0, 2),
                                   (18, 0, 2), (19, 0, 2)],
}


def attended_operand_digit(model, cfg, pos, layer, head):
    """Which digit's operands does this node attend to most? (For answer-position
    carry nodes the relevant operands are the NEXT-lower digit that generates the
    carry.) Returns the digit n maximizing operand attention."""
    best = None
    for n in range(cfg.n_digits):
        m = operand_attention(model, cfg, pos, layer, head, n, n_q=20)
        if best is None or m > best[1]:
            best = (n, m)
    return best


def sweep_model(model_name, control_bar):
    model, cfg = load_model(model_name)
    acc = verify_accuracy(model, cfg)
    assert acc > 0.9, f"{model_name} accuracy {acc} INVALID"
    bar = max(0.50, control_bar - 0.10)
    nodes = {}
    cand = CANDIDATES.get(model_name)
    if cand is None:
        # data-driven: all layer-0 and layer-1 heads at question positions
        cand = [(p, L, h) for L in range(cfg.n_layers) for p in range(2 * cfg.n_digits + 2)
                for h in range(cfg.n_heads)]
    for (pos, layer, head) in cand:
        node = f"P{pos}L{layer}H{head}"
        # which digit does it plausibly serve? test each digit; keep strongest A_{n+1}
        best = None
        for n in range(cfg.n_digits):
            def hb(pos=pos, layer=layer, head=head):
                return [{"name": f"blocks.{layer}.attn.hook_z", "pos": pos, "head": head}]
            def hb_joint(pos=pos, layer=layer, head=head):
                return [{"name": f"blocks.{layer}.attn.hook_z", "pos": pos, "head": head},
                        {"name": f"blocks.{layer}.hook_mlp_out", "pos": pos}]
            sig_local = flip_signature(model, cfg, n, hb, "local", n_pairs=40)
            # prefer the digit with the strongest single-digit signature (either
            # A_n for SA or A_{n+1} for ST); pick by the max of the two.
            score = max(sig_local["A_n+1_flip"], sig_local["A_n_flip"])
            if best is None or score > best["score"]:
                best = {"digit": n, "score": score, "hb": hb, "hb_joint": hb_joint}
        n = best["digit"]
        sig_local = flip_signature(model, cfg, n, best["hb"], "local", n_pairs=60,
                                   src_cls=1, tgt_cls=0)
        sig_local_rev = flip_signature(model, cfg, n, best["hb"], "local", n_pairs=60,
                                       src_cls=0, tgt_cls=1)  # direction symmetry (A-3)
        sig_joint = flip_signature(model, cfg, n, best["hb_joint"], "local", n_pairs=60)
        sig_cascade = flip_signature(model, cfg, n, best["hb"], "cascade", n_pairs=60)
        sig_tristate = tristate_test(model, cfg, n, best["hb"], n_pairs=60)  # F1 fix
        null = same_class_null(model, cfg, n, best["hb"], "local", cls=0, n_pairs=60)
        # attend to the CARRY-GENERATING lower digit n (the operands of digit n)
        attn = operand_attention(model, cfg, pos, layer, head, n)
        both_dir = min(sig_local["A_n+1_flip"], sig_local_rev["A_n+1_flip"])
        # tri-state gate (F1): a node is R-ST(tri) only if the genuine U test
        # (fix sum=9, toggle lower carry) flips A_{n+1}; else R-SC(binary).
        tri_ok = (not np.isnan(sig_tristate["A_n+1_flip"])) and sig_tristate["A_n+1_flip"] >= bar
        base = classify_base(sig_local, null, attn, bar)  # ST-region / SA / SV / none
        if base == "carry":
            cls_z = "R-ST(tri)" if tri_ok else "R-SC(binary)"
        else:
            cls_z = base
        nodes[node] = {
            "target_digit": n, "operand_attn_mass": attn, "bar": bar,
            "both_direction_A_n+1_flip": both_dir,
            "z": {"local": sig_local, "local_reverse": sig_local_rev,
                  "cascade": sig_cascade, "tristate": sig_tristate,
                  "null": null, "classification": cls_z},
            "joint_zmlp": {"local": sig_joint},
        }
    del model
    return {"model": model_name, "accuracy": acc, "bar": bar, "nodes": nodes}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    results = {}
    control_bars = {}
    if mode in ("control", "models", "all"):
        for mn in ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289"]:
            print(f"=== CONTROL {mn} ===", flush=True)
            c = run_control(mn)
            control_bars[mn] = c["node_level_control"]["A_n_flip"]
            nc = c["node_level_control"]; cc = c["coarse_resid_eq_control"]
            print(f"  acc={c['accuracy']:.3f} node-ctrl {nc['node']} A_n_flip={nc['A_n_flip']:.2f} "
                  f"A_n+1_flip={nc['A_n+1_flip']:.2f} null_A_n={nc['same_class_null_A_n']:.2f}")
            print(f"  coarse resid@= A_n+1_flip={cc['A_n+1_flip']:.2f} any_flip={cc['any_flip']:.2f}")
            results.setdefault(mn, {})["control"] = c
    if mode in ("models", "all"):
        for mn in ["add_d5_l2_h3_t15K_s372001", "add_d6_l2_h3_t20K_s173289"]:
            print(f"=== SWEEP {mn} ===", flush=True)
            bar = control_bars.get(mn, 0.6)
            r = sweep_model(mn, bar)
            results.setdefault(mn, {})["sweep"] = r
            for node, nr in r["nodes"].items():
                z = nr["z"]
                tri = z['tristate']['A_n+1_flip']
                print(f"  {node} D{nr['target_digit']} attn={nr['operand_attn_mass']:.2f} "
                      f"| z: A_n={z['local']['A_n_flip']:.2f} A_n+1={z['local']['A_n+1_flip']:.2f} "
                      f"tristate_A_n+1={tri:.2f} null_A_n+1={z['null']['A_n+1_flip']:.2f} "
                      f"-> {z['classification']}")
    with open(os.path.join(RESULT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    print("\nArtifacts in", RESULT_DIR)


if __name__ == "__main__":
    main()
