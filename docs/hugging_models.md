# HuggingFace resources
Two HuggingFace layouts hold the model + analysis artifacts (verified 2026-07-16):
- **`PhilipQuirke/VerifiedArithmetic`** (legacy flat repo): **48** model weight-sets,
  each present both as `<name>.pth` and `<name>/model.pth`, with a paired
  `<name>_train.json`; 33 also carry legacy `<name>_behavior.json` + `<name>_maths.json`.
- **`PhilipQuirke/QuantaMaths_<name>`** (per-model analysis repos): **53** repos, each
  with `model.pth` + `training_loss.json` + `behaviors.json` + `features.json` (a
  superset — includes 5 subtraction models not in the flat repo).

The [Complete model inventory](#complete-model-inventory-hf-verified-2026-07-16) below
lists every model. Only the annotated subset in the sections that follow is
**behaviorally accuracy-verified** (in this thread or the papers); for the rest,
presence is documented but per-model accuracy lives in each model's
`training_loss.json`/`_train.json` (the file stores `Config` + the 50K-step
`TrainingLoss` curve — accuracy itself is a behavioral million-question test, not a
stored scalar) and is not asserted here.

A model name like add_d5_l2_h3_t30K_s372001 can be "read" as: Performs addition, n_digits=5, n_layers=2, n_heads=3, training_epochs=30K, training_seed=372001

## Training Resources
For each model the 'VerifiedArithmeticTrain' Colab notebook generates two files:
- A "XXXXXX.pth" file containing the model weights
- A "XXXXXX_train.json" file containing configuration information and training loss data

These files are available on HuggingFace for these models:

### 5-digit and 6-digit digit Addition models
- add_d5_l1_h3_t30K_s372001: Inaccurate **5-digit, 1-layer, 3-attention-head**, addition model. Reproduces Paper 1 model. Can predict S0, S1 and S2 complexity addition questions.
- add_d5_l2_h3_t15K_s372001: **Accurate** 5-digit, **2-layers**, 3-head addition model trained for 15K epochs. Training loss is 9e-9
- add_d6_l2_h3_t15K_s372001: **Accurate** **6-digit**, 2-layers, 3-head addition model trained for 15K epochs. Training seed is 372001
- add_d6_l2_h3_t20K_s173289: **Accurate** 6-digit, 2-layers, 3-head addition model trained for **20K** epochs. Training seed is 173289
- add_d6_l2_h3_t20K_s572091: **Accurate** 6-digit, 2-layers, 3-head addition model trained for **20K** epochs. Training seed is 572091

### Larger addition models (8/10/13-digit — cross-size sweep, CE18)
Accurate 2-layer/3-head addition models used for the cross-size SV generalization
sweep (CE18) and the latent-geometry dynamic-range battery (CE27 certificate).
- add_d8_l2_h3_t45K_s173289: **Accurate** **8-digit** addition model, 45K epochs, seed 173289 (8-digit fill-in, acc 1.000).
- add_d10_l2_h3_t40K_s572091: **Accurate** **10-digit** addition model, 40K epochs, seed 572091 (acc 1.000).
- add_d13_l2_h3_t50K_s572091: **Accurate** **13-digit** addition model, 50K epochs, seed 572091 (acc 1.000).

### 6-digit Subtraction model
- sub_d6_l2_h3_t30K_s372001: Inaccurate 6-digit, 2-layers, 3-head subtraction model trained for 30K epochs.

### 6-digit Mixed (addition and subtraction) model
- mix_d6_l3_h4_t40K_s372001: Inaccurate 6-digit, **3-layers, 4-head mixed** (add and subtract) model trained for 40K epochs. Training loss is 8e-09

### "ins1" 6-digit Mixed models initialised with 6-digit addition model
- ins1_mix_d6_l3_h4_t40K_s372001: **Accurate** 6-digit, 3-layers, 4-head mixed initialised with addition model. Handles 1m Qs for Add and Sub. 
- ins1_mix_d6_l3_h4_t40K_s173289: Inaccurate. AvgFinalLoss=1.6e-08. 936K for Add, 1M Qs for Sub 
- ins1_mix_d6_l3_h3_t40K_s572091: Inaccurate. AvgFinalLoss=1.8e-08. Fails on 1M Qs. For 099111-099111=+0000000 gives -0000000. Improve training data.
- ins1_mix_d6_l3_h4_t50K_s572091: Inaccurate. AvgFinalLoss=2.9e-08. 1M for Add. 300K for Sub. For 000041-000047=-0000006 gives +0000006. Improve training data.
- ins1_mix_d8_l3_h4_t70K_s572091: **Accurate** **8-digit** mixed model, 3-layers, 4-head, 70K epochs, seed 572091. Per-class accuracy ADD 1.00 / SUB 1.00 / NEG 0.997 (CE26). First 8-digit mixed model.

### "ins2" 6-digit Mixed model initialised with 6-digit addition model. Reset useful heads every 100 epochs.
- ins2_mix_d6_l4_h4_t40K_s372001: Inaccurate 6-digit, 3-layers, 4-head mixed initialise with addition model. Reset useful heads every 100 epochs. Training loss is 7e-09. Fails 1m Qs

### "ins3" 6-digit Mixed model initialised with 6-digit addition model. Reset useful heads & MLPs every 100 epochs.
- ins3_mix_d6_l4_h3_t40K_s372001: Inaccurate 6-digit, 3-layers, 4-head mixed initialise with addition model. Reset useful heads and MLP every 100 epochs. 

## Complete model inventory  

The sections above annotate accuracy for the studied models. This is the **full**
set actually present on HuggingFace. Accuracy is NOT asserted for the unstudied
models here (see each model's `training_loss.json`). Naming: a `gf` token before the
seed marks a training-method variant (e.g. `add_d10_l2_h3_t40K_gf_s572091` vs the
non-`gf` twin).

**Addition (`add_*`, 19 flat weight-sets):** d5 — `add_d5_l1_h3_t15K_s372001`,
`add_d5_l1_h3_t30K_s372001`, `add_d5_l2_h3_t15K_s372001`, `add_d5_l2_h3_t40K_s372001`;
d6 — `add_d6_l2_h3_t15K_s372001`, `add_d6_l2_h3_t20K_s173289`,
`add_d6_l2_h3_t20K_s572091`, `add_d6_l2_h3_t40K_s372001`; d7–d9 —
`add_d7_l2_h3_t45K_s173289`, `add_d8_l2_h3_t45K_s173289`, `add_d9_l2_h3_t45K_s173289`;
d10+ — `add_d10_l2_h3_t40K_s572091` (+ `_gf_` twin), `add_d11_l2_h3_t50K_s572091`,
`add_d12_l2_h3_t50K_s572091`, `add_d13_l2_h3_t50K_s572091`, `add_d14_l2_h3_t60K_s572091`,
`add_d15_l2_h3_t80K_s572091`, `add_d20_l2_h3_t80K_s572091`.

**Subtraction (`sub_*`):** flat weights — `sub_d6_l2_h3_t30K_s372001`,
`sub_d10_l2_h3_t75K_s173289` (+ `_gf_` twin). Analysis-repo-only (no flat `.pth`) —
`sub_d5_l2_h3_t30K_s372001`, `sub_d6_l2_h3_t30K_s572091`, `sub_d8_l2_h3_t50K_s173289`,
`sub_d8_l2_h3_t50K_s371793`, `sub_d12_l2_h3_t75K_s371793`.

**Mixed, from-scratch (`mix_*`, 3-layer/4-head, 10):** `mix_d5_l3_h4_t40K_s372001`,
`mix_d6_l3_h4_t40K_s372001`, `mix_d7_l3_h4_t50K_s372001`, `mix_d8_l3_h4_t60K_s173289`,
`mix_d9_l3_h4_t60K_s173289`, `mix_d10_l3_h4_t75K_s173289` (+ `_gf_` twin),
`mix_d11_l3_h4_t80K_s572091`, `mix_d12_l3_h4_t85K_s572091`, `mix_d13_l3_h4_t85K_s572091`.

**Mixed, addition-initialised (`ins1_mix_*`, 14):** d5/d6 —
`ins1_mix_d5_l2_h3_t40K_s572091`, `ins1_mix_d6_l2_h3_t40K_s572091`,
`ins1_mix_d6_l3_h3_t40K_s572091`, `ins1_mix_d6_l3_h3_t80K_s572091`,
`ins1_mix_d6_l3_h4_t40K_s173289`, `ins1_mix_d6_l3_h4_t40K_s372001` (accurate — studied),
`ins1_mix_d6_l3_h4_t50K_s572091`; d7+ — `ins1_mix_d7_l3_h4_t50K_s572091`,
`ins1_mix_d8_l3_h4_t70K_s572091` (accurate — studied),
`ins1_mix_d9_l3_h4_t70K_s572091`, `ins1_mix_d10_l3_h3_t50K_s572091` (+ `_gf_` twin),
`ins1_mix_d11_l3_h4_t75K_s572091`, `ins1_mix_d12_l3_h4_t85K_s572091`.

**Mixed, reset-heads variants:** `ins2_mix_d6_l4_h4_t40K_s372001` (reset useful heads
every 100 epochs), `ins3_mix_d6_l4_h3_t40K_s372001` (reset heads + MLPs).

## Analysis Resources
For each model two analysis files are generated:
- A **behavior** file containing "behavior" facts automatically learnt about the model
  (`Fail%`, `Impact`, `Math.Add`/`Math.Sub`/`Math.Neg`, `Attn`, PCA tri-case `.SP`/`.MP`).
- A **maths/feature** file containing "maths-specific" algorithmic facts
  (`Algo:` role tags).

Each file is a JSON list of node dicts `{position, layer, is_head, num, tags}` where
every tag is a single-colon `major:minor` string (scalars ride as a `=NN` suffix).
The behavior file keeps ALL non-`Algo` tags; the maths/feature file keeps ONLY
`Algo:` tags (a save-time major-tag filter). See
[useful_tags.md](useful_tags.md) for the canonical, complete tag list.

Two production paths write these (same JSON schema):
- The 'VerifiedArithmeticAnalysis' **Colab notebook** → `XXXXXX_behavior.json` +
  `XXXXXX_maths.json` in the `PhilipQuirke/VerifiedArithmetic` repo.
- The headless **`quanta_maths.maths_hf_update`** pipeline (the same discovery code,
  extracted into `quanta_maths.maths_analysis`, plus reusable "techniques") →
  `behaviors.json` + `features.json` in each model's per-model
  `PhilipQuirke/QuantaMaths_<name>` analysis repo. It CREATES the map from scratch
  when absent (e.g. the 8-digit mixed model `ins1_mix_d8_l3_h4_t70K_s572091`) and
  otherwise extends it, idempotently, round-trip verified before upload. Run:
  `python -m quanta_maths.maths_hf_update` (dry-run default; `--execute` to upload).

### Behavior json example
The ins1_mix_d6_l3_h4_t40K_s372001_behavior.json file starts with:
```
[{"position": 0, "layer": 0, "is_head": true, "num": 3, "tags": ["Fail%:5", "Impact:A7", "Math.Sub:M0", "Math.Neg:N1234", "Attn:P0=100"]}, 
{"position": 6, "layer": 0, "is_head": true, "num": 0, "tags": ["Fail%:2", "Impact:A43210", "Math.Sub:M123", "Math.Neg:N1", "Attn:P1=73", "Attn:P6=23", "Attn:P4=2", "Attn:P2=1"]}, 
{"position": 9, "layer": 0, "is_head": true, "num": 0, "tags": ["Fail%:3", "Impact:A765", "Math.Add:S1234", "Math.Sub:M3", "Math.Neg:N1", "Attn:P8=50", "Attn:P1=49"]}, 
{"position": 9, "layer": 0, "is_head": true, "num": 1, "tags": ["Fail%:1", "Impact:A654", "Math.Add:S124", "Attn:P8=52", "Attn:P1=46"]}, {"position": 9, "layer": 0, "is_head": false, "num": 0, "tags": ["Fail%:8", "Impact:A765", "Math.Add:S12345",
```

### Maths json example
Some lines of the ins1_mix_d6_l3_h4_t40K_s372001_maths.json file are:
```
[{"position": 0, "layer": 0, "is_head": true, "num": 3, "tags": []},
{"position": 6, "layer": 0, "is_head": true, "num": 0, "tags": ["Algo:OPR"]},
{"position": 9, "layer": 0, "is_head": true, "num": 0, "tags": ["Algo:D4.GT"]},
...
{"position": 15, "layer": 0, "is_head": true, "num": 2, "tags": ["Algo:A5.SA", "Algo:A5.MD", "Algo:A5.ND.A5"]},
{"position": 15, "layer": 0, "is_head": true, "num": 3, "tags": ["Algo:OPR", "Algo:SGN"]},
```

### New analysis tags added by the `quanta_maths` technique pipeline
Beyond the auto-discovered `Algo:` role tags (`SA`/`SC`/`SS`/`ST`, `MD`/`MB`/`MT`/`GT`,
`ND`/`NB`, `OPR`/`SGN`/`SLT`), the `maths_hf_update` techniques add these (idempotent,
gated by `applies_to(cfg)`; canonical list in [useful_tags.md](useful_tags.md)):

Into the **maths/feature** file (`Algo:`):
- `Algo:A{d}.STC` — addition ST-combiner: the answer-position last-layer MLP that
  combines the resolved **carry** into digit `A{d}`.
- `Algo:A{d}.MTC` — positive-answer-subtraction parallel (combines the resolved **borrow**).
- `Algo:A{d}.NTC` — negative-answer-subtraction parallel (combines the resolved **neg-borrow**).

Into the **behavior** file (`Probe:`):
- `Probe:A{d}.LINXFER=NN` — operand digit is linearly decodable (balanced-acc `NN`%) at
  its first-layer fetch site.
- `Probe:A{top}.CARRYLAYER=NN` / `Probe:A{top}.CARRYDEFER=NN` — token-time finalization
  of the propagated carry (read layer; tokens deferred past full-input availability).
- `Probe:DELIVERY.{ADD|SUB|NEG}=route` — how the resolved carry/borrow reaches the
  combiner per class: `res` (residual only), `resatt` (residual + last-layer attention),
  `att`, or `none`.

### Analysis coverage
Per-model analysis repos (`QuantaMaths_<name>`, `behaviors.json` + `features.json`)
exist for **all 53 models** (verified 2026-07-16); the legacy flat repo additionally
holds `<name>_behavior.json` + `<name>_maths.json` for 33 models. Map *depth* varies:
- The **5/6-digit accurate add / mixed models** have the fullest verified maps
  (`Algo:` roles, `Fail%`, `Impact`, attention targets, `SP`/`MP` tags).
- The accurate **8-digit mixed models** carry combiner-complete maps (fixed
  2026-07-17, CE32): `ins1_mix_d8_l3_h4_t70K_s572091` features.json previously had
  **no** combiner tags (a CE26 upload gap) → now **STC6/MTC6/NTC6** (co-located at
  {P20-P25 L2 M0}); the from-scratch worked example `mix_d8_l3_h4_t60K_s173289` was
  **missing MTC** → now **STC4/MTC6/NTC5**. Both refreshed via `maths_hf_update.update_model`
  (library `_combiner_is_causal` redundancy-proof fallback). The 6-digit mixed
  `ins1_mix_d6_l3_h4_t40K_s372001` carries the STC/MTC/NTC + `Probe:` tags above.
- The larger **10/13-digit addition** maps (`add_d10_l2_h3_t40K_s572091`,
  `add_d13_l2_h3_t50K_s572091`) carry question-tail/sign **ST-writer** tags and
  answer-position **combiner-MLP** (`STC`) tags, but **no L1 consumer-head tags** — the
  consumer role was identified empirically in the cross-size sweep (CE18), itself a
  role-transfer datapoint.
- The remaining repos (d11/d12/d14/d15/d20 addition, the `mix_*` from-scratch zoo,
  larger `ins1_mix_*`, and the analysis-repo-only `sub_*` models) exist but are not
  all fully tagged / accuracy-verified in this thread.

Note on studies that upload nothing: the latent-geometry stream (CE27
rail-factorization, and the paired G2/G3 dominance-certificate) is
**local-results-only** (`results/study-geometry-*/`) — it adds **no** HF node tags or
artifacts (per-prompt geometry read off existing maps' carry rail), so it does not
change this folder contract.
