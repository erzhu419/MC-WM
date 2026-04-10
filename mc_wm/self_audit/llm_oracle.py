"""
Optional LLM Oracle (dev manual §4.4).

ONLY called after max_rounds of automated expansion still fails.
This file is the ONLY place in MC-WM that imports an LLM client.
Delete this file and the entire system still works.

Safety:
  - ASTEval sandbox: proposed features are parsed and validated before use
  - Max 3 features per query
  - SINDy quality gate must still pass after LLM features are added
"""

import ast
import numpy as np
from typing import List, Optional


try:
    import asteval
    HAS_ASTEVAL = True
except ImportError:
    HAS_ASTEVAL = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


SYSTEM_PROMPT = """You are a physics feature engineer specializing in sim-to-real dynamics gaps.
You will receive a diagnosis report showing which statistical tests still fire after automated expansion.
Your job: propose up to 3 symbolic features (as Python lambda strings) that might capture the remaining structure.

Rules:
- Features must be valid Python expressions using numpy (as 'np') and obs array 'x'
- x has shape (N, feature_dim) where first obs_dim columns are state, rest are action
- Example valid feature: "x[:, 3] * np.abs(x[:, 5])" (cross-dimensional coupling)
- Example valid feature: "np.sign(x[:, 2]) * x[:, 2]**2" (signed quadratic)
- Return ONLY a JSON list of strings, nothing else
- Maximum 3 features
"""


class ASTEvalSandbox:
    """
    Validates that LLM-proposed feature strings are safe to execute.
    Only allows numpy operations and array indexing.
    """

    ALLOWED_NAMES = {"np", "x", "abs", "sign", "exp", "log", "sin", "cos", "sqrt"}
    MAX_DEPTH = 5

    def validate(self, expr_str: str) -> bool:
        try:
            tree = ast.parse(expr_str, mode="eval")
        except SyntaxError:
            return False
        return self._check_node(tree.body, 0)

    def _check_node(self, node, depth: int) -> bool:
        if depth > self.MAX_DEPTH:
            return False
        if isinstance(node, ast.Name):
            return node.id in self.ALLOWED_NAMES
        if isinstance(node, (ast.Constant, ast.Num)):
            return True
        if isinstance(node, ast.Attribute):
            return isinstance(node.value, ast.Name) and node.value.id in self.ALLOWED_NAMES
        if isinstance(node, (ast.BinOp, ast.UnaryOp)):
            return all(self._check_node(c, depth + 1) for c in ast.iter_child_nodes(node))
        if isinstance(node, ast.Subscript):
            return self._check_node(node.value, depth + 1)
        if isinstance(node, (ast.Slice, ast.Index)):
            return True
        if isinstance(node, ast.Call):
            return all(self._check_node(c, depth + 1) for c in ast.iter_child_nodes(node))
        if isinstance(node, ast.Tuple):
            return True
        return False


class LLMOracle:
    """
    Optional LLM fallback for feature proposal.

    To use: set ANTHROPIC_API_KEY in environment and install anthropic.
    If neither is available, the oracle gracefully returns empty list.
    """

    def __init__(self, model: str = "claude-sonnet-4-6", max_features: int = 3):
        self.model = model
        self.max_features = max_features
        self.sandbox = ASTEvalSandbox()
        self._client = None
        if HAS_ANTHROPIC:
            try:
                self._client = anthropic.Anthropic()
            except Exception:
                pass

    def query(self, diagnosis_report: str) -> List[str]:
        """
        Ask LLM for feature suggestions given diagnosis report.
        Returns list of validated expression strings.
        """
        if self._client is None:
            return []

        try:
            import json
            response = self._client.messages.create(
                model=self.model,
                max_tokens=256,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": diagnosis_report}],
            )
            raw = response.content[0].text.strip()
            # Extract JSON list
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start == -1 or end == 0:
                return []
            candidates = json.loads(raw[start:end])
        except Exception:
            return []

        # Validate each candidate
        validated = []
        for expr in candidates[:self.max_features]:
            if isinstance(expr, str) and self.sandbox.validate(expr):
                validated.append(expr)

        return validated

    def build_library(self, feature_exprs: List[str], base_library) -> object:
        """
        Compile validated expression strings into pysindy CustomLibrary.
        """
        import pysindy as ps

        fns = []
        names = []
        for expr in feature_exprs:
            try:
                fn = eval(f"lambda x: ({expr}).reshape(-1, 1)", {"np": np})
                fns.append(fn)
                names.append(lambda x, e=expr: e[:20])  # truncated name
            except Exception:
                continue

        if not fns:
            return base_library

        custom = ps.CustomLibrary(library_functions=fns, function_names=names)
        from mc_wm.self_audit.auto_expand import combine_libraries
        return combine_libraries(base_library, custom)
