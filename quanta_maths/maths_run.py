"""Model-execution helpers: build a question, read the predicted answer, check accuracy.

Promoted from the copy-pasted ``make_q`` / ``predict_answer`` / ``verify_accuracy``
that appeared in nearly every experiment script (via ``scripts/confirm_st_node.py``).
Operation-aware so addition and subtraction models share one implementation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch


def answer_positions(cfg) -> List[int]:
    """Token positions that hold the answer (sign + digits), top..units order."""
    na = cfg.n_digits + 2
    return list(range(cfg.n_ctx - na, cfg.n_ctx))


def make_question(cfg, a: int, b: int, operation=None) -> torch.Tensor:
    """Build a single question row tensor for ``a <op> b`` (default PLUS)."""
    from quanta_maths.maths_utilities import make_a_maths_question_and_answer
    from quanta_maths.maths_constants import MathsToken
    if operation is None:
        operation = MathsToken.PLUS
    q = torch.zeros((1, cfg.n_ctx), dtype=torch.int64)
    make_a_maths_question_and_answer(cfg, q, 0, a, b, operation)
    return q[0]


def predict_answer(model, cfg, q: torch.Tensor) -> torch.Tensor:
    """Return the predicted answer tokens (n_digits+2 of them) for one question row."""
    with torch.no_grad():
        logits = model(q.unsqueeze(0))
    ap = answer_positions(cfg)
    return logits[0, [p - 1 for p in ap]].argmax(-1)


def verify_accuracy(model, cfg, n: int = 64, operation=None,
                    rng: Optional[np.random.Generator] = None) -> float:
    """Fraction of ``n`` random questions the model answers exactly correctly.

    For subtraction the operands span the full range (answers may be negative);
    for addition they are drawn in the lower half so the sum stays in range.
    """
    from quanta_maths.maths_constants import MathsToken
    if operation is None:
        operation = MathsToken.PLUS
    if rng is None:
        rng = np.random.default_rng(0)
    lim = 10 ** cfg.n_digits
    ok = 0
    for _ in range(n):
        if operation == MathsToken.MINUS:
            a, b = int(rng.integers(0, lim)), int(rng.integers(0, lim))
        else:
            a, b = int(rng.integers(0, lim // 2)), int(rng.integers(0, lim // 2))
        q = make_question(cfg, a, b, operation=operation)
        if torch.equal(predict_answer(model, cfg, q), q[answer_positions(cfg)]):
            ok += 1
    return ok / n
