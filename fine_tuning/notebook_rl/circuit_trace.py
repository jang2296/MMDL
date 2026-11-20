"""Bounded, gold-independent checks of authored DC ladder traces."""
from __future__ import annotations

import json
import re
from fractions import Fraction

from .schema import ActorItem, VerifierItem

VERSION = "native-circuit-ladder-trace-v1"
SCOPE = ("Authored 1–2-source, 2–3-stage ideal DC ladder trace only. Exact native nodal and "
         "series-parallel solvers must agree. Node voltages, resistor incidence/resistance and source "
         "polarity are checked independently of authored local arithmetic. Local arithmetic may be "
         "compliant despite physically wrong operands; it does not certify the whole solution. "
         "Null is explicitly unverified; blocked_by lists exactly the earlier null operand checks, "
         "not repaired values. Final-answer correctness, latent reasoning, visual perception, module "
         "causality and free lesson semantics are unchecked. Bridges are unsupported.")
_COVERAGE = (("CIRCUIT_NODE_COVERAGE", "nodes"), ("CIRCUIT_RESISTOR_COVERAGE", "ohm"),
             ("CIRCUIT_KCL_COVERAGE", "kcl"))
_OHM = ("CIRCUIT_INCIDENCE", "CIRCUIT_RESISTANCE", "CIRCUIT_DROP", "CIRCUIT_OHM")
_KCL = ("CIRCUIT_KCL_INCIDENCE", "CIRCUIT_KCL_SUM", "CIRCUIT_KCL_ZERO")
_INPUT = (("CIRCUIT_INPUT_VOLTAGE", "input[0]"), ("CIRCUIT_INPUT_VOLTAGE_LINK", "input[0]"),
          ("CIRCUIT_INPUT_CURRENT", "input[1]"), ("CIRCUIT_INPUT_CURRENT_LINK", "input[1]"),
          ("CIRCUIT_INPUT_RESISTANCE", "input[2]"), ("CIRCUIT_INPUT_OHM", "input[2]"))
RULES = frozenset(rule for rule, _ in _COVERAGE + _INPUT) | frozenset(
    (*_OHM, *_KCL, "CIRCUIT_NODE_VOLTAGE", "CIRCUIT_QUERY_SPEC", "CIRCUIT_QUERY_VALUE"))


def instruction_for_question(question: str, source: str) -> str:
    from .generators import circuit

    queries = (("current", "R4"), ("resistance", "input"),
               *(("voltage", f"N{i}") for i in range(3, 6)))
    if source != circuit.GENERATOR_VERSION or question not in {circuit._question(q) for q in queries}:
        raise ValueError("circuit trace: unsupported canonical question/source")
    return (f"Trace contract {VERSION}: supports only ideal DC ladders with 1–2 series sources and "
            "2–3 resistor stages. Make the FIRST nonempty line after Solution: exactly Trace: followed "
            "by one JSON object on that same line (at most 4096 characters). Assert your own executed "
            "trace, not an example, quotation, hypothetical or negation. Use exactly kind, nodes, ohm, "
            "kcl, input, query; kind is the string circuit_ladder. nodes lists every node once in numeric "
            "label order as [node-label string, voltage]. ohm lists every resistor once in numeric name "
            "order as [resistor-name string, from-node string, to-node string, resistance, voltage-drop, "
            "current]. Use the pictured series direction from source toward load and shunt direction "
            "toward N0. Drop equals from voltage minus to voltage; current equals drop/resistance. "
            "kcl lists every load node once in numeric order as [node string, signed terms, sum]; signed "
            "terms are [resistor-name string, integer sign] pairs in numeric resistor order, including "
            "every incident resistor with +1 outgoing and -1 incoming. Sum signed currents and assert "
            "zero. input is [total-source-voltage, source-current, equivalent-resistance]; link voltage "
            "to R1's from-node and current to R1, and use voltage/current for resistance. query is "
            "[quantity string current/voltage/resistance, target string, value, unit string A/V/ohm] "
            "for the question, linked to your authored resistor current, node voltage, or input resistance. "
            "All voltages, resistances, currents, sums and query values are canonical reduced rational "
            "JSON STRINGS: signed integer or numerator/positive-denominator, at most 32 digits per "
            "integer, no leading zeros, plus sign, decimal, exponent, spaces or denominator 1. Use JSON "
            "null for an unknown numeric value; null never means correct. Node/name labels are strings. "
            "Use no extra fields, duplicate rows, trace lines or code fences. Before Solution: use only "
            "Observed facts: and assignment lines. Component observations use a positive one- or "
            "two-digit integer, ASCII space, then V for comp[E1]/comp[E2] or ohm for comp[R1]..comp[R6]; "
            "do not omit units or use aliases, joined units, decimals, or explanatory suffixes. "
            "After Trace: write only one Final answer: line. "
            "Use LF or CRLF; section headers and trace padding allow only ASCII spaces/tabs.")


def _rational(value: object) -> bool:
    if value is None:
        return True
    if (type(value) is not str or not re.fullmatch(
            r"(?:0|-?[1-9][0-9]{0,31})(?:/[1-9][0-9]{0,31})?", value)):
        return False
    return str(Fraction(value)) == value


def _node(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"N[0-5]", value) is not None


def _resistor(value: object) -> bool:
    return type(value) is str and re.fullmatch(r"R[1-6]", value) is not None


def schema(trace: dict) -> bool:
    if (set(trace) != {"kind", "nodes", "ohm", "kcl", "input", "query"}
            or trace["kind"] != "circuit_ladder"
            or type(trace["nodes"]) is not list or not 1 <= len(trace["nodes"]) <= 6
            or any(type(row) is not list or len(row) != 2 or not _node(row[0]) or not _rational(row[1])
                   for row in trace["nodes"])
            or len({row[0] for row in trace["nodes"]}) != len(trace["nodes"])):
        return False
    nodes = {row[0] for row in trace["nodes"]}
    if (type(trace["ohm"]) is not list or not 1 <= len(trace["ohm"]) <= 6
            or any(type(row) is not list or len(row) != 6 or not _resistor(row[0])
                   or not _node(row[1]) or not _node(row[2]) or row[1] == row[2]
                   or row[1] not in nodes or row[2] not in nodes
                   or not all(_rational(v) for v in row[3:]) for row in trace["ohm"])
            or len({row[0] for row in trace["ohm"]}) != len(trace["ohm"])):
        return False
    names = {row[0] for row in trace["ohm"]}
    if (type(trace["kcl"]) is not list or not 0 <= len(trace["kcl"]) <= 3
            or any(type(row) is not list or len(row) != 3 or not _node(row[0]) or row[0] not in nodes
                   or not _rational(row[2]) or type(row[1]) is not list or not 1 <= len(row[1]) <= 6
                   or any(type(term) is not list or len(term) != 2 or not _resistor(term[0])
                          or term[0] not in names or type(term[1]) is not int or term[1] not in (-1, 1)
                          for term in row[1])
                   or len({term[0] for term in row[1]}) != len(row[1]) for row in trace["kcl"])
            or len({row[0] for row in trace["kcl"]}) != len(trace["kcl"])
            or type(trace["input"]) is not list or len(trace["input"]) != 3
            or not all(_rational(v) for v in trace["input"])):
        return False
    query = trace["query"]
    return (type(query) is list and len(query) == 4 and type(query[0]) is str
            and query[0] in ("current", "voltage", "resistance")
            and (_resistor(query[1]) if query[0] == "current" else _node(query[1])
                 if query[0] == "voltage" else query[1] == "input")
            and _rational(query[2]) and type(query[3]) is str and query[3] in ("A", "V", "ohm"))


def inventory(trace: dict) -> list[tuple[str, str]]:
    """Only syntactic regions, never native values or verdicts."""
    if not schema(trace):
        raise ValueError("circuit trace: malformed public inventory")
    rows = [("RESPONSE_SCOPE", "response"), *_COVERAGE]
    rows.extend(("CIRCUIT_NODE_VOLTAGE", f"nodes[{i}]") for i in range(len(trace["nodes"])))
    rows.extend((rule, f"ohm[{i}]") for i in range(len(trace["ohm"])) for rule in _OHM)
    rows.extend((rule, f"kcl[{i}]") for i in range(len(trace["kcl"])) for rule in _KCL)
    return [*rows, *_INPUT, ("CIRCUIT_QUERY_SPEC", "query"), ("CIRCUIT_QUERY_VALUE", "query")]


def identity(trace: dict, rule: str, locus: str) -> list | None:
    """Operation identity follows node/resistor/orientation, not a trace array index."""
    if not schema(trace) or (rule, locus) not in inventory(trace) or rule not in RULES:
        return None
    if rule == "CIRCUIT_NODE_VOLTAGE":
        return ["circuit_ladder", rule, trace["nodes"][int(locus[6:-1])][0]]
    if rule in _OHM:
        row = trace["ohm"][int(locus[4:-1])]
        return ["circuit_ladder", rule, *row[:3]]
    if rule in _KCL:
        return ["circuit_ladder", rule, trace["kcl"][int(locus[4:-1])][0]]
    if rule in {"CIRCUIT_QUERY_SPEC", "CIRCUIT_QUERY_VALUE"}:
        return ["circuit_ladder", rule, *trace["query"][:2], trace["query"][3]]
    return ["circuit_ladder", rule, locus]


def supported_unknown(row: dict, checks: list[dict]) -> bool:
    """Only after fresh report validation; accepts intrinsic nulls and exact null dependencies."""
    if row.get("rule") not in RULES or row.get("status") != "UNVERIFIABLE":
        return False
    actual = {(check.get("rule"), check.get("locus")): check.get("status") for check in checks}
    return (actual.get((row.get("rule"), row.get("locus"))) == "UNVERIFIABLE"
            and all(actual.get((ref["rule"], ref["locus"])) == "UNVERIFIABLE"
                    for ref in row.get("blocked_by", [])))


def _public_checks(trace: dict) -> dict[tuple[str, str], tuple[bool | None, list[dict]]]:
    """Recompute all local arithmetic and exact null dependencies without a private world."""
    checks: dict[tuple[str, str], tuple[bool | None, list[dict]]] = {}
    nodes = {row[0]: (row[1], ("CIRCUIT_NODE_VOLTAGE", f"nodes[{i}]"))
             for i, row in enumerate(trace["nodes"])}
    ohm = {row[0]: (row, i) for i, row in enumerate(trace["ohm"])}

    def add(key: tuple[str, str], value: str | None, operands: list[tuple[str | None, tuple[str, str]]],
            valid: bool) -> None:
        dependencies = list(dict.fromkeys(ref for operand, ref in operands if operand is None))
        checks[key] = (None if value is None or dependencies else valid,
                       [{"rule": rule, "locus": locus} for rule, locus in dependencies])

    for value, key in nodes.values():
        if value is None:
            add(key, None, [], False)
    for i, row in enumerate(trace["ohm"]):
        _, a, b, resistance, drop, current = row
        locus = f"ohm[{i}]"
        if resistance is None:
            add(("CIRCUIT_RESISTANCE", locus), None, [], False)
        va, vb = nodes[a][0], nodes[b][0]
        add(("CIRCUIT_DROP", locus), drop, [nodes[a], nodes[b]],
            all(v is not None for v in (drop, va, vb)) and Fraction(drop) == Fraction(va) - Fraction(vb))
        add(("CIRCUIT_OHM", locus), current,
            [(resistance, ("CIRCUIT_RESISTANCE", locus)), (drop, ("CIRCUIT_DROP", locus))],
            all(v is not None for v in (resistance, drop, current)) and Fraction(resistance) != 0
            and Fraction(current) * Fraction(resistance) == Fraction(drop))
    for i, (_, terms, total) in enumerate(trace["kcl"]):
        operands = [(ohm[name][0][5], ("CIRCUIT_OHM", f"ohm[{ohm[name][1]}]")) for name, _ in terms]
        add(("CIRCUIT_KCL_SUM", f"kcl[{i}]"), total, operands,
            total is not None and all(value is not None for value, _ in operands)
            and Fraction(total) == sum((sign * Fraction(ohm[name][0][5]) for name, sign in terms), Fraction(0)))
        add(("CIRCUIT_KCL_ZERO", f"kcl[{i}]"), total, [], total is not None and Fraction(total) == 0)
    voltage, current, resistance = trace["input"]
    for i, (value, native_rule) in enumerate(zip(trace["input"], ("CIRCUIT_INPUT_VOLTAGE", "CIRCUIT_INPUT_CURRENT",
                                                               "CIRCUIT_INPUT_RESISTANCE"), strict=True)):
        if value is None:
            add((native_rule, f"input[{i}]"), None, [], False)
    r1 = ohm.get("R1")
    source_node = nodes[r1[0][1]] if r1 else None
    add(("CIRCUIT_INPUT_VOLTAGE_LINK", "input[0]"), voltage, [source_node] if source_node else [],
        source_node is not None and voltage is not None and source_node[0] is not None
        and Fraction(voltage) == Fraction(source_node[0]))
    r1_current = (r1[0][5], ("CIRCUIT_OHM", f"ohm[{r1[1]}]")) if r1 else None
    add(("CIRCUIT_INPUT_CURRENT_LINK", "input[1]"), current, [r1_current] if r1_current else [],
        r1_current is not None and current is not None and r1_current[0] is not None
        and Fraction(current) == Fraction(r1_current[0]))
    add(("CIRCUIT_INPUT_OHM", "input[2]"), resistance,
        [(voltage, ("CIRCUIT_INPUT_VOLTAGE", "input[0]")),
         (current, ("CIRCUIT_INPUT_CURRENT", "input[1]"))],
        all(v is not None for v in trace["input"]) and Fraction(current) != 0
        and Fraction(resistance) * Fraction(current) == Fraction(voltage))
    quantity, target, value, _ = trace["query"]
    operand = (nodes.get(target) if quantity == "voltage" else
               (ohm[target][0][5], ("CIRCUIT_OHM", f"ohm[{ohm[target][1]}]"))
               if quantity == "current" and target in ohm else
               (resistance, ("CIRCUIT_INPUT_RESISTANCE", "input[2]")) if quantity == "resistance" else None)
    add(("CIRCUIT_QUERY_VALUE", "query"), value, [operand] if operand else [],
        operand is not None and value is not None and operand[0] is not None
        and Fraction(value) == Fraction(operand[0]))
    return checks


def _contract(actor: ActorItem, item: VerifierItem) -> tuple[dict, list[dict], dict, tuple[str, str]]:
    from .generators import circuit

    world, query = item.accept["circuit_world"], item.accept["circuit_query"]
    if len(json.dumps(world, allow_nan=False)) > 16384 or type(world) is not dict or world.get("kind") != "ladder":
        raise ValueError("circuit trace: bounded ladder required")
    circuit._validate(world)
    if type(query) is not list or len(query) != 2 or any(type(v) is not str for v in query):
        raise ValueError("circuit trace: query shape")
    query_tuple = (query[0], query[1])
    unit = {"current": "A", "voltage": "V", "resistance": "ohm"}.get(query[0])
    facts, types = circuit._facts(world)
    if (actor.source != circuit.GENERATOR_VERSION or actor.family != "circuit"
            or actor.problem_id != item.problem_id or not actor.problem_id.startswith(actor.source + ":circuit:")
            or actor.choices or item.choices or actor.answer_format != "unit_number" or item.answer_type != "unit_number"
            or query_tuple not in {circuit._query(world, i) for i in (0, 3, 6)}
            or actor.question != circuit._question(query_tuple)
            or item.accept != {"circuit_world": world, "circuit_query": query, "unit_aliases": [unit], "require_unit": True}
            or item.accept["require_unit"] is not True
            or item.slots != facts or item.slot_types != types or item.required_slots != tuple(facts)):
        raise ValueError("circuit trace: native provenance/fact contract differs")
    first, second = circuit._solutions(world)
    if first != second:
        raise ValueError("circuit trace: independent solver disagreement")
    return world, circuit._components(world), first, query_tuple


def check_trace(*, actor: ActorItem, item: VerifierItem, text: str, finish_reason: str) -> dict:
    from .reasoning_trace import _finish, _response_scope, _sha, _trace
    from .observations import observation_spans

    report: dict = {"version": VERSION, "status": "UNVERIFIABLE", "trace": None, "checks": [], "issues": [],
                    "coverage": {"kind": None, "checked": [], "unverified": [], "complete": False},
                    "semantic_scope": SCOPE, "finish_reason": finish_reason, "text_sha256": _sha(text)}

    def add(rule: str, locus: str, valid: bool | None, dependencies: list[dict] | None = None) -> None:
        report["checks"].append({"rule": rule, "locus": locus, "observation_refs": [],
                                 "blocked_by": [] if dependencies is None else dependencies,
                                 "status": "UNVERIFIABLE" if valid is None else "COMPLIANT" if valid else "VIOLATED"})

    try:
        world, components, solution, query = _contract(actor, item)
        report["coverage"]["kind"] = "circuit_ladder"
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RecursionError, ZeroDivisionError):
        add("NATIVE_CONTRACT", "input", None)
        return _finish(report)
    try:
        trace, report["trace"] = _trace(text)
    except (ValueError, RecursionError):
        add("TRACE_FORMAT", "solution", None)
        return _finish(report)
    if not schema(trace):
        add("TRACE_SCHEMA", "trace", None)
        return _finish(report)
    spans = observation_spans(text)
    if len(spans) > 64 or len({key for _, _, key, _ in spans}) != len(spans):
        add("TRACE_FORMAT", "solution", None)
        return _finish(report)
    nodes = solution["node_voltages_V"]
    resistors = {comp["name"]: comp for comp in components if comp["kind"] == "RESISTOR"}
    load_nodes = [f"N{i}" for i in range(len(world["sources"]) + 1, len(nodes))]
    native = {("RESPONSE_SCOPE", "response"): True if _response_scope(text, report["trace"]["span"][1]) else None,
              _COVERAGE[0]: [row[0] for row in trace["nodes"]] == list(nodes),
              _COVERAGE[1]: [row[0] for row in trace["ohm"]] == list(resistors),
              _COVERAGE[2]: [row[0] for row in trace["kcl"]] == load_nodes}
    for i, (node, value) in enumerate(trace["nodes"]):
        native["CIRCUIT_NODE_VOLTAGE", f"nodes[{i}]"] = value is not None and Fraction(value) == nodes.get(node)
    for i, (name, a, b, resistance, _, _) in enumerate(trace["ohm"]):
        component = resistors.get(name)
        native["CIRCUIT_INCIDENCE", f"ohm[{i}]"] = component is not None and (a, b) == (component["a"], component["b"])
        native["CIRCUIT_RESISTANCE", f"ohm[{i}]"] = (component is not None and resistance is not None
                                                          and Fraction(resistance) == int(component["value"]))
    for i, (node, terms, _) in enumerate(trace["kcl"]):
        expected = [[name, 1 if comp["a"] == node else -1] for name, comp in resistors.items()
                    if node in (comp["a"], comp["b"])]
        native["CIRCUIT_KCL_INCIDENCE", f"kcl[{i}]"] = node in load_nodes and terms == expected
    for (rule, locus), actual, expected in zip(((_INPUT[0]), (_INPUT[2]), (_INPUT[4])), trace["input"],
            (sum(world["sources"]), solution["source_current_A"], solution["equivalent_R_ohm"]), strict=True):
        native[rule, locus] = actual is not None and Fraction(actual) == expected
    native["CIRCUIT_QUERY_SPEC", "query"] = (tuple(trace["query"][:2]) == query
        and trace["query"][3] == {"current": "A", "voltage": "V", "resistance": "ohm"}[query[0]])
    public = _public_checks(trace)
    for rule, locus in inventory(trace):
        if (rule, locus) in public:
            valid, dependencies = public[rule, locus]
            add(rule, locus, valid, dependencies)
        else:
            add(rule, locus, native[rule, locus])
    return _finish(report)


def validate_report(report: dict, text: str, finish_reason: str) -> None:
    """Bind raw evidence, full public inventory, arithmetic and exact null dependencies."""
    from .reasoning_trace import _finish, _response_scope, _sha, _trace
    from .observations import observation_spans

    try:
        statuses = {"COMPLIANT", "VIOLATED", "UNVERIFIABLE"}
        if (len(json.dumps(report, allow_nan=False)) > 65536 or set(report) != {
                "version", "status", "trace", "checks", "issues", "coverage", "semantic_scope",
                "finish_reason", "text_sha256"} or report["version"] != VERSION or report["semantic_scope"] != SCOPE
                or report["status"] not in statuses or report["text_sha256"] != _sha(text)
                or report["finish_reason"] != finish_reason or type(finish_reason) is not str or len(finish_reason) > 32):
            raise ValueError("report contract")
        payload = None
        if report["trace"] is not None:
            payload, evidence = _trace(text)
            if report["trace"] != evidence:
                raise ValueError("trace evidence differs")
        coverage = report["coverage"]
        if (type(coverage) is not dict or set(coverage) != {"kind", "checked", "unverified", "complete"}
                or type(coverage["complete"]) is not bool or coverage["kind"] not in (None, "circuit_ladder")
                or report != _finish({**report, "coverage": dict(coverage)})):
            raise ValueError("report summary differs")
        rows = report["checks"]
        if type(rows) is not list or not 1 <= len(rows) <= 64:
            raise ValueError("check bounds")
        for row in rows:
            if (type(row) is not dict or set(row) != {"rule", "locus", "observation_refs", "blocked_by", "status"}
                    or row["status"] not in statuses or row["observation_refs"] != [] or type(row["blocked_by"]) is not list):
                raise ValueError("check contract")
        roots = {("NATIVE_CONTRACT", "input"), ("TRACE_FORMAT", "solution"), ("TRACE_SCHEMA", "trace")}
        if len(rows) == 1 and (rows[0]["rule"], rows[0]["locus"]) in roots:
            root = rows[0]
            if root["status"] != "UNVERIFIABLE" or root["blocked_by"]:
                raise ValueError("root contract")
            if root["rule"] == "NATIVE_CONTRACT":
                if coverage["kind"] is not None or payload is not None:
                    raise ValueError("native root evidence")
            elif coverage["kind"] != "circuit_ladder":
                raise ValueError("root coverage")
            elif root["rule"] == "TRACE_SCHEMA":
                if payload is None or schema(payload):
                    raise ValueError("false schema root")
            elif payload is not None:
                spans = observation_spans(text)
                if not schema(payload) or len(spans) <= 64 and len({key for _, _, key, _ in spans}) == len(spans):
                    raise ValueError("false format root")
            else:
                try:
                    _trace(text)
                except (ValueError, RecursionError):
                    return
                raise ValueError("missing parsed evidence")
            return
        if payload is None or coverage["kind"] != "circuit_ladder" or not schema(payload):
            raise ValueError("unbound circuit trace")
        if [(row["rule"], row["locus"]) for row in rows] != inventory(payload):
            raise ValueError("missing or reordered regions")
        expected_scope = "COMPLIANT" if _response_scope(text, report["trace"]["span"][1]) else "UNVERIFIABLE"
        if rows[0]["status"] != expected_scope or rows[0]["blocked_by"]:
            raise ValueError("response scope differs")
        public = _public_checks(payload)
        for row in rows[1:]:
            key = row["rule"], row["locus"]
            if key in public:
                valid, dependencies = public[key]
                expected = "UNVERIFIABLE" if valid is None else "COMPLIANT" if valid else "VIOLATED"
                if row["status"] != expected or row["blocked_by"] != dependencies:
                    raise ValueError("public arithmetic/null dependency differs")
            elif row["status"] == "UNVERIFIABLE" or row["blocked_by"]:
                raise ValueError("unsupported private uncertainty")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RecursionError, ZeroDivisionError) as exc:
        raise ValueError("circuit trace: invalid report or original evidence binding") from exc
