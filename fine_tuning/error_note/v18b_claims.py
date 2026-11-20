"""Conservative source checks for v18b; generated text is never rewritten."""
from __future__ import annotations

import ast
from copy import deepcopy
from fractions import Fraction
import re

from . import self_note_v16 as bounded
from . import v16b_rows as previous
from . import v17_rows as v17
from .audit import digest


def expression(text: str) -> ast.Expression:
    text = text.strip().replace("−", "-").replace("×", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d)(?=[a-zA-Z])", "*", text)
    if not text or len(text) > 160 or not re.fullmatch(r"[\d\sa-zA-Z.+*/()\-]+", text):
        raise ValueError("UNSUPPORTED_EXPRESSION")
    tree = ast.parse(text, mode="eval")
    if len(list(ast.walk(tree))) > 80:
        raise ValueError("EXPRESSION_TOO_LONG")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
                                 ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.UAdd, ast.USub)):
            raise ValueError("UNSUPPORTED_EXPRESSION")
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float) or not re.fullmatch(r"\d+(?:\.\d+)?", str(node.value))):
            raise ValueError("UNSUPPORTED_NUMBER")
        if isinstance(node, ast.Name) and not re.fullmatch(r"[a-zA-Z]", node.id):
            raise ValueError("UNSUPPORTED_VARIABLE")
    return tree


def calculate(text: str, variables: dict | None = None) -> Fraction:
    variables = variables or {}

    def value(node):
        if isinstance(node, ast.Constant):
            return Fraction(str(node.value))
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError("UNBOUND_VARIABLE")
            return variables[node.id]
        if isinstance(node, ast.UnaryOp):
            return -value(node.operand) if isinstance(node.op, ast.USub) else value(node.operand)
        left, right = value(node.left), value(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ValueError("ZERO_DIVISOR")
        return left / right

    return value(expression(text).body)


def same_label(left, right) -> bool:
    try:
        first, second = expression(str(left)), expression(str(right))
        if not any(isinstance(node, ast.Name) for tree in (first, second) for node in ast.walk(tree)):
            return calculate(str(left)) == calculate(str(right))
        return ast.dump(first) == ast.dump(second)
    except (ValueError, SyntaxError):
        return str(left).strip() == str(right).strip()


def substitution_values(record: dict, source: dict, name: str) -> set[Fraction]:
    """Use explicit question assignments or annotated equal segments, never the known final answer."""
    from .v17_data import _annotation

    question = str(record.get("question") or record.get("evaluation", {}).get("question") or record.get("p0_text", ""))
    values = {Fraction(match[1]) for match in re.finditer(rf"(?<!\w){re.escape(name)}\s*=\s*(-?\d+(?:\.\d+)?)(?![\w.])", question)}
    raw, _ = _annotation(record)
    if not raw or source.get("annotation_sha256") != digest(raw):
        return values
    forms = raw.get("diagram_logic_forms", [])
    if "candidates" in raw:
        forms = raw["candidates"][raw["matched_id"]].get("diagram_logic_forms", [])

    def affine(node):
        if isinstance(node, ast.Constant):
            return Fraction(0), Fraction(str(node.value))
        if isinstance(node, ast.Name):
            if node.id != name:
                raise ValueError("UNBOUND_VARIABLE")
            return Fraction(1), Fraction(0)
        if isinstance(node, ast.UnaryOp):
            a, b = affine(node.operand)
            return (-a, -b) if isinstance(node.op, ast.USub) else (a, b)
        a, b = affine(node.left)
        c, d = affine(node.right)
        if isinstance(node.op, ast.Add):
            return a + c, b + d
        if isinstance(node.op, ast.Sub):
            return a - c, b - d
        if isinstance(node.op, ast.Mult) and not (a and c):
            return a * d + b * c, b * d
        if isinstance(node.op, ast.Div) and c == 0 and d != 0:
            return a / d, b / d
        raise ValueError("NONLINEAR_OR_UNSUPPORTED")

    for form in forms:
        match = re.fullmatch(r"Equals\(LengthOf\(Line\(([A-Z]),\s*([A-Z])\)\),\s*LengthOf\(Line\(([A-Z]),\s*([A-Z])\)\)\)", form)
        if not match:
            continue
        labels = []
        for entity in (list(match.groups()[:2]), list(match.groups()[2:])):
            facts = [fact for fact in source.get("facts", []) if fact["kind"] == "SEGMENT_LABEL" and
                     sorted(fact["entity"]) == sorted(entity)]
            if len(facts) != 1:
                break
            labels.append(str(facts[0]["value"]))
        if len(labels) != 2:
            continue
        try:
            a, b = affine(expression(labels[0]).body)
            c, d = affine(expression(labels[1]).body)
            if a != c:
                values.add((d - b) / (a - c))
        except (ValueError, SyntaxError, ArithmeticError):
            continue
    return values


def verdict(claim: dict, source: dict) -> tuple[str, str]:
    if claim["kind"] in {"SEGMENT_LABEL", "ANGLE_LABEL"}:
        facts = [f for f in source.get("facts", []) if v17._key(f, source) == v17._key(claim, source)]
        if facts:
            return ("VERIFIED", "PRINTED_LABEL_MATCH") if any(same_label(claim["value"], f["value"]) for f in facts) else ("CONTRADICTED", "PRINTED_LABEL_MISMATCH")
    if claim["kind"] == "OBJECT_ATTRIBUTE":
        domains = {"material": {"rubber", "metal"}, "size": {"small", "large"},
                   "shape": {"cube", "sphere", "cylinder"},
                   "color": {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}}
        entity = claim.get("entity", [])
        if len(entity) != 2 or str(claim["value"]) not in domains.get(entity[1], set()):
            return "UNKNOWN", "INVALID_ATTRIBUTE_DOMAIN"
    return v17._verdict(claim, source)


def source_for(record: dict, source: dict) -> dict:
    """Expand graph scope only from the same sealed annotation and image binding."""
    if source == v17._verifier(record, None):
        return v17._seal({"schema": "v18b-no-source-v1", "problem_id": record["problem_id"],
                          "image_sha256": record.get("image_sha256", []), "facts": [],
                          "supported_claim_kinds": [], "closed_world_kinds": []})
    v17._verifier(record, source)
    if source.get("full_graph_receipt") or record.get("layer") != "graph":
        return source
    from .content_v16 import parse_gvlqa_structure
    from .v17_data import _annotation

    raw, metadata = _annotation(record)
    if not raw or source.get("annotation_sha256") != digest(raw) or not source.get("obs_eligible"):
        return source
    directed = metadata.get("directed", raw.get("directed"))
    if type(directed) is not bool or directed != source.get("directed"):
        raise ValueError("FULL_GRAPH_DIRECTION_BINDING")
    text = raw.get("structure_text")
    if text is None:
        text = f"{raw['n']} {raw['m']}\n" + "\n".join(" ".join(map(str, edge)) for edge in raw["edges"])
    graph = parse_gvlqa_structure(text)
    facts = [{"kind": "VISIBLE_NODE", "entity": node, "value": True} for node in graph["visible_nodes"]]
    for edge in graph["edges"]:
        facts.append({"kind": "EDGE_WEIGHT" if len(edge) == 3 else "EDGE_PRESENT", "entity": list(edge[:2]),
                      "value": str(edge[2]) if len(edge) == 3 else True})
    expanded = {k: deepcopy(v) for k, v in source.items() if k != "sha256"}
    expanded.update(facts=facts, closed_world_scopes=source.get("closed_world_scopes", {}) | {"visible_nodes": graph["visible_nodes"]},
                    full_graph_receipt={"annotation_sha256": digest(raw), "image_sha256": record.get("image_sha256", []),
                                        "original_verifier_sha256": source["sha256"], "nodes": len(graph["visible_nodes"]), "edges": graph["m"]})
    result = v17._seal(expanded)
    for fact in source.get("facts", []):
        if verdict(fact, result)[0] != "VERIFIED":
            raise ValueError("FULL_GRAPH_SCOPED_FACT_MISMATCH")
    return result


def balanced_query(record: dict, source: dict) -> dict:
    nodes = source.get("closed_world_scopes", {}).get("visible_nodes", [])
    if not nodes:
        nodes = sorted(f["entity"] for f in source.get("facts", []) if f["kind"] == "VISIBLE_NODE")
    if "EDGE_PRESENT" not in source.get("closed_world_kinds", []):
        raise ValueError("OBS_ABSENCE_UNCERTIFIED")
    if "EDGE_PRESENT" not in source.get("supported_claim_kinds", []):
        raise ValueError("OBS_UNSUPPORTED_EDGE_PRESENCE")
    directed = source.get("directed", False)
    existing = set()
    for fact in source.get("facts", []):
        if fact["kind"] == "EDGE_WEIGHT" or fact["kind"] == "EDGE_PRESENT" and v17._same_value(fact["value"], True):
            pair = tuple(fact["entity"]) if directed else tuple(sorted(fact["entity"]))
            existing.add(pair)
    pools: dict[bool, list[list[int]]] = {True: [], False: []}
    for first in nodes:
        for second in nodes:
            if first == second or not source.get("directed") and first > second:
                continue
            pair = (first, second) if directed else tuple(sorted((first, second)))
            pools[pair in existing].append([first, second])
    seed = int(digest([record["problem_id"], 3407, "v18b-balanced-edge"]), 16)
    selected_class = record.get("v18b_obs_class", bool(seed % 2))
    assignment = record.get("v18b_obs_balance")
    if assignment is not None:
        v17._check_seal(assignment)
        if (assignment.get("problem_id") != record["problem_id"] or assignment.get("image_sha256") != record.get("image_sha256", [])
                or assignment.get("verifier_sha256") != source["sha256"] or type(assignment.get("present")) is not bool
                or assignment.get("present") != bool(assignment.get("rank", -1) % 2)):
            raise ValueError("OBS_BALANCE_BINDING_CHANGED")
        selected_class = assignment["present"]
    selected = pools[selected_class]
    if not selected:
        raise ValueError("OBS_BALANCED_CLASS_UNAVAILABLE")
    first, second = selected[(seed // 2) % len(selected)]
    return {"kind": "EDGE_PRESENT", "entity": [first, second], "unit": "",
            "target": f"whether an edge goes from visible numeric node {first} to visible numeric node {second}; preserve arrow direction when directed"}


def _claim(text: str, start: int, end: int, *, kind: str, entity, value, unit: str,
           result: str, reason: str, source: dict, **extra) -> dict:
    quote = text[start:end]
    assertion = previous._natural_assertion(text, [start, end], quote)
    return {"kind": kind, "original_entity": entity, "original_value": value, "original_quote": quote,
            "unit": unit, "canonical_fact_key": list(v17._key({"kind": kind, "entity": entity, "unit": unit}, source)),
            "verdict": result if assertion == "ASSERTED" else "UNKNOWN", "assertion": assertion, "reason": reason,
            "original_byte_spans": [[len(text[:start].encode()), len(text[:end].encode())]], **extra}


def direct_edge_claims(text: str, source: dict) -> list[dict]:
    """Read explicit one-edge assertions; connectivity alone does not specify a path or direction."""
    mark = r"(?:\*\*|__)?"
    first = rf"{mark}\bnode[ \t]+{mark}(?P<a>\d+)\b{mark}"
    second = rf"{mark}\bnode[ \t]+{mark}(?P<b>\d+)\b{mark}"
    patterns = (
        rf"{first}[ \t]+is[ \t]+(?P<negative>not[ \t]+)?directly[ \t]+connected[ \t]+to[ \t]+{second}{mark}",
        rf"{first}[ \t]+and[ \t]+{second}[ \t]+are[ \t]+(?P<negative>not[ \t]+)?directly[ \t]+connected\b{mark}",
        rf"{mark}\bthere[ \t]+is[ \t]+(?P<negative>no[ \t]+|not[ \t]+an?[ \t]+)?"
        rf"(?:an?[ \t]+)?(?:direct[ \t]+)?edge[ \t]+(?P<link>from|between)[ \t]+"
        rf"{first}[ \t]+(?P<join>to|and)[ \t]+{second}{mark}",
    )
    claims = []
    for pattern in patterns:
        for match in re.finditer(pattern + r"(?:\.(?!\w))?", text, re.I):
            groups = match.groupdict()
            ordered = (groups.get("link") or "").lower() == "from"
            if groups.get("link") and groups["join"].lower() != ("to" if ordered else "and"):
                continue
            entity, value = [int(match["a"]), int(match["b"])], not bool(match["negative"])
            result, reason = verdict({"kind": "EDGE_PRESENT", "entity": entity, "value": value}, source)
            claim = _claim(text, *match.span(), kind="EDGE_PRESENT", entity=entity, value=value,
                           unit="", result=result, reason=reason, source=source)
            # Only the parsed denial is projected away; doubt and withdrawal outside it remain intact.
            if not value:
                surrogate = "bound_fact = value"
                projected = text[:match.start()] + surrogate + text[match.end():]
                assertion = previous._natural_assertion(projected,
                    [match.start(), match.start() + len(surrogate)], surrogate)
                claim.update(assertion=assertion, verdict=result if assertion == "ASSERTED" else "UNKNOWN")
            # Embedded propositions and quoted speech are not source assertions.
            line_start = text.rfind("\n", 0, match.start()) + 1
            line_end = text.find("\n", match.end())
            line_end = len(text) if line_end < 0 else line_end
            prefix = text[line_start:match.start()].strip()
            prior = text[:line_start].rstrip().rsplit("\n", 1)[-1]
            if not prefix and prior and not re.search(r"[.!?]$", prior):
                prefix = prior
            prefix = re.sub(r"^(?:[-*][ \t]+|\d+[.)][ \t]+)", "", prefix)
            assertive = (r"(?:(?:관측|observation|connections?)\s*:\s*)?"
                         r"(?:(?:from the (?:visual layout(?: and edge connections)?|graph|image|diagram),\s*)?"
                         r"(?:we|i) (?:can )?(?:see|observe) that\s*|"
                         r"(?:the graph shows that|therefore|thus|so|hence|clearly)[, ]*)?")
            tail = text[match.end():line_end]
            continuation = "" if match[0].endswith(".") else re.split(r"[.!?]", tail, 1)[0].strip()
            if (not re.fullmatch(assertive, prefix, re.I)
                    or not re.fullmatch(r"(?:via (?:an? )?(?:direct )?edge)?", continuation, re.I)
                    or re.search(r"""["'“”‘’`?]|\b(?:impossible|inconceivable|denied|deny|reject|dispute|hypothesis)\b""",
                                 text[line_start:line_end], re.I)):
                claim.update(assertion="UNKNOWN", verdict="UNKNOWN")
            if source.get("directed") is not False and not ordered:
                claim.update(verdict="UNKNOWN", reason="EDGE_DIRECTION_UNSPECIFIED")
            claims.append(claim)
    claims.sort(key=lambda claim: claim["original_byte_spans"])
    # A later opposite assertion withdraws the earlier one, including reversed undirected pairs.
    for index, claim in enumerate(claims):
        if any(later["canonical_fact_key"] == claim["canonical_fact_key"]
               and later["original_value"] != claim["original_value"]
               and later["assertion"] == "ASSERTED" for later in claims[index + 1:]):
            claim.update(assertion="RETRACTED", verdict="UNKNOWN")
    return claims


def claims_for(record: dict, source: dict, claims: list[dict]) -> list[dict]:
    text = record["t1"]["text"]
    claims = deepcopy(claims)
    # Whole equalities own their entire arithmetic region; never certify a suffix of a larger equation.
    equations = []
    for match in re.finditer(r"(?<![\w])(?P<lhs>[\d().+*/×÷−\-]+(?:\s*[+*/×÷−\-]\s*[\d().+*/×÷−\-]+)+)\s*=\s*(?P<rhs>[\d().+*/×÷−\-]+(?:\s*[+*/×÷−\-]\s*[\d().+*/×÷−\-]+)*)", text):
        if any(ch in text[max(0, match.start()-1):match.start()] for ch in "*/+-()"):
            continue
        try:
            computed = calculate(match["lhs"])
            actual = calculate(match["rhs"])
        except (ValueError, SyntaxError, ArithmeticError):
            continue
        equations.append((match.start(), match.end()))
        claims.append(_claim(text, *match.span(), kind="ARITHMETIC", entity=match["lhs"].strip(), value=match["rhs"], unit="",
                             result="VERIFIED" if computed == actual else "CONTRADICTED", reason="WHOLE_EXACT_EQUATION", source=source,
                             computed_value=str(computed), operation="EXACT_RATIONAL", rounding=None))
    claims = [c for c in claims if c.get("reason") == "WHOLE_EXACT_EQUATION" or c["kind"] != "ARITHMETIC"
              or not any(a <= len(text.encode()[:c["original_byte_spans"][0][0]].decode()) and
                         len(text.encode()[:c["original_byte_spans"][0][1]].decode()) <= b for a, b in equations)]
    # Read a printed symbolic label as an expression, rather than its first numeral.
    if record.get("layer") == "geometry":
        for fact in source.get("facts", []):
            if fact["kind"] not in {"SEGMENT_LABEL", "ANGLE_LABEL"}:
                continue
            name = "".join(fact["entity"])
            pattern = rf"(?<!\w){re.escape(name)}\s*(?:=|is|:)\s*(?P<value>[0-9a-zA-Z().+*/×÷−\-]+(?:\s*[+*/×÷−\-]\s*[0-9a-zA-Z().+*/×÷−\-]+)*)"
            for match in re.finditer(pattern, text):
                value = match["value"]
                if re.fullmatch(r"-?\d+(?:\.\d+)?(?:cm|mm|km|m|degrees?)", value):
                    continue
                try:
                    expression(value)
                except (ValueError, SyntaxError):
                    continue
                start, end = len(text[:match.start()].encode()), len(text[:match.end()].encode())
                claims = [claim for claim in claims if not (claim["kind"] == fact["kind"] and
                          claim["original_entity"] == fact["entity"] and
                          start < claim["original_byte_spans"][0][1] and claim["original_byte_spans"][0][0] < end)]
                result, reason = verdict(fact | {"value": value}, source)
                claims.append(_claim(text, *match.span(), kind=fact["kind"], entity=fact["entity"], value=value,
                                     unit=fact.get("unit", "length"), result=result, reason=reason, source=source))
    # Any unsupported larger equation makes its old partial arithmetic claims UNKNOWN.
    for claim in claims:
        if claim.get("reason") == "UNIT_MISMATCH":
            continue
        if claim["kind"] == "ARITHMETIC" and claim.get("reason") != "WHOLE_EXACT_EQUATION":
            a, b = claim["original_byte_spans"][0]
            left = text.encode()[:a].decode().rsplit("\n", 1)[-1].strip()
            right = text.encode()[b:].decode().split("\n", 1)[0].strip()
            if left.endswith(tuple("0123456789*/+-()")) or right.startswith(tuple("*/+-()")):
                claim.update(verdict="UNKNOWN", reason="PARTIAL_EQUATION_UNSUPPORTED")
        if claim["kind"] in {"SEGMENT_LABEL", "ANGLE_LABEL"}:
            if claim.get("reason") == "UNIT_MISMATCH":
                continue
            facts = [f for f in source.get("facts", []) if v17._key(f, source) == v17._key({"kind": claim["kind"], "entity": claim["original_entity"], "unit": claim.get("unit", "")}, source)]
            if not facts:
                continue
            printed = facts[0]["value"]
            try:
                names = {node.id for node in ast.walk(expression(str(printed))) if isinstance(node, ast.Name)}
                numeric = calculate(str(claim["original_value"]))
            except (ValueError, SyntaxError, ArithmeticError):
                names, numeric = set(), None
            if names and numeric is not None:
                assignments = {}
                before = text.encode()[:claim["original_byte_spans"][0][0]].decode()
                for name in names:
                    matches = list(re.finditer(rf"(?<!\w){name}\s*=\s*(-?\d+(?:\.\d+)?)(?![\w.])", before))
                    proposed = {Fraction(match[1]) for match in matches if previous._natural_assertion(text, list(match.span()), match[0]) == "ASSERTED"}
                    independent = substitution_values(record, source, name)
                    if len(proposed) == 1 and proposed == independent:
                        assignments[name] = next(iter(proposed))
                try:
                    expected = calculate(str(printed), assignments)
                    claim.update(verdict="VERIFIED" if expected == numeric else "CONTRADICTED", reason="SOURCE_BOUND_SUBSTITUTION",
                                 computed_value=str(expected), printed_value=printed)
                except (ValueError, ArithmeticError):
                    claim.update(verdict="UNKNOWN", reason="DERIVED_LENGTH_UNVERIFIED", printed_value=printed)
            else:
                result, reason = verdict({"kind": claim["kind"], "entity": claim["original_entity"], "value": claim["original_value"], "unit": claim.get("unit", "")}, source)
                if claim.get("assertion") == "ASSERTED":
                    claim.update(verdict=result, reason=reason)
    if record.get("layer") == "graph":
        claims.extend(direct_edge_claims(text, source))
        # A claimed path is one assertion, checked against every edge in the certified scope.
        for match in re.finditer(r"(?<!\w)(\d+(?:\s*(?:->|→)\s*\d+)+)", text, re.I):
            line_start, line_end = text.rfind("\n", 0, match.start()) + 1, text.find("\n", match.end())
            line_end = len(text) if line_end < 0 else line_end
            line = text[line_start:line_end]
            lead = text[:line_start].rstrip().rsplit("\n", 1)[-1]
            if not re.search(r"\b(?:path|route)\b", line + " " + lead, re.I):
                continue
            nodes = [int(n) for n in re.findall(r"\d+", match[1])]
            edge_results = [{"edge": [a, b], "verdict": verdict({"kind": "EDGE_PRESENT", "entity": [a, b], "value": True}, source)[0]}
                            for a, b in zip(nodes, nodes[1:])]
            result = "UNKNOWN" if any(e["verdict"] == "UNKNOWN" for e in edge_results) else "CONTRADICTED" if any(e["verdict"] == "CONTRADICTED" for e in edge_results) else "VERIFIED"
            claim = _claim(text, *match.span(), kind="GRAPH_PATH", entity=nodes, value=nodes, unit="", result=result,
                           reason="ALL_PATH_EDGES_CHECKED", source=source, edge_checks=edge_results)
            committed = re.match(r"\s*[—–]*\s*(?:forms?|is|gives?)\s+(?:a\s+)?(?:valid\s+|connected\s+|correct\s+|shortest\s+)?(?:path|route)\b", text[match.end():line_end], re.I)
            scope_end = match.end() + committed.end() if committed else line_end
            assertion_scope = text[line_start:scope_end]
            if committed and claim["assertion"] == "UNKNOWN":
                local_span = [match.start() - line_start, match.end() - line_start]
                local = previous._natural_assertion(assertion_scope, local_span, match[0])
                if local == "ASSERTED":
                    claim.update(verdict=result, assertion=local)
            if re.search(r"\b(?:if|suppose|assuming|maybe|might|would|could|hypothetical|not|invalid|incorrect)\b", assertion_scope, re.I):
                claim.update(verdict="UNKNOWN", assertion="UNKNOWN", reason="PATH_NOT_COMMITTED")
            claims.append(claim)
    claims.sort(key=lambda c: c["original_byte_spans"])
    for index, claim in enumerate(claims):
        if text.count(claim["original_quote"]) != 1:
            claim.update(verdict="UNKNOWN", reason="DUPLICATE_QUOTE", assertion="UNKNOWN")
        claim.update(claim_id=f"t{index}", verifier_evidence_sha256=source["sha256"])
    return claims


def match_quote(original: str, quote: str, claims: list[dict], stage: str) -> list[dict]:
    """One unique original byte interval must contain one complete claim and nothing else asserted."""
    if not quote:
        return []
    offsets = [m.start() for m in re.finditer(re.escape(quote), original)]
    # OBS strings may quote their decoded scalar without JSON's surrounding quotes.
    if len(offsets) != 1:
        return []
    start, end = len(original[:offsets[0]].encode()), len(original[:offsets[0] + len(quote)].encode())
    found = []
    for claim in claims:
        if claim["stage"] != stage:
            continue
        a, b = claim["original_byte_spans"][0]
        if stage == "OBS" and a <= start < end <= b:
            fragment = original.encode()[a:b].decode()
            if quote == fragment or fragment.startswith('"') and quote == fragment[1:-1]:
                found.append(claim)
        elif start <= a < b <= end:
            # Surrounding prose can broaden a quote, but not add another assertion or final answer.
            residual = (original.encode()[start:a] + original.encode()[b:end]).decode()
            if not re.search(r"\d|=|->|→|\b(?:answer|if|maybe|not|actually|instead)\b", residual, re.I):
                found.append(claim)
    if len(found) != 1:
        return []
    if previous._natural_assertion(original, [offsets[0], offsets[0] + len(quote)], quote) != "ASSERTED" and stage != "OBS":
        return []
    return found


def repair_verdict(claim: dict, corrected: str, source: dict) -> bool:
    if claim["kind"] == "GRAPH_PATH":
        if not re.fullmatch(r"\d+(?:\s*(?:->|→)\s*\d+)+", corrected):
            return False
        nodes = [int(n) for n in re.findall(r"\d+", corrected)]
        old = claim["original_entity"]
        return nodes[0] == old[0] and nodes[-1] == old[-1] and len(nodes) == len(set(nodes)) and all(
            verdict({"kind": "EDGE_PRESENT", "entity": [a, b], "value": True}, source)[0] == "VERIFIED"
            for a, b in zip(nodes, nodes[1:]))
    if claim.get("reason") == "SOURCE_BOUND_SUBSTITUTION":
        return bounded._same(corrected, claim["computed_value"])
    return verdict({"kind": claim["kind"], "entity": claim["original_entity"], "value": corrected, "unit": claim.get("unit", "")}, source)[0] == "VERIFIED"
