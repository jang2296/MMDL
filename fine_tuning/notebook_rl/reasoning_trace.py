"""Private checks of model-authored native traces; never supplies a reference trace."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .observations import _normalize, observation_spans, slot_id
from .repair_contract import parse_retry_sections, project_solution
from .schema import ActorItem, VerifierItem

VERSION = "native-reasoning-trace-v4"
_STATUSES = {"COMPLIANT", "VIOLATED", "UNVERIFIABLE"}
_SCOPE = ("Authored trace only, not latent reasoning, visual perception, "
          "or module causality. BFS checks declared FIFO/discovery steps; tree checks declared "
          "root/child/enter/emit/exit events. Native graph/tree/DFA have no edge-ID grounding. Final-answer correctness "
          "and free lesson semantics are separate and unchecked here. Native numeric observations "
          "must contain only dimensionless ASCII integer literals; response line breaks must be LF or CRLF. "
          "Section-header suffixes and trace-line padding allow ASCII spaces/tabs only, with original evidence unchanged. blocked_by identifies "
          "proven failed prerequisites only; dependent regions remain unverified, with no repaired suffix.")
_RULES = {"TRACE_FORMAT", "NATIVE_CONTRACT", "TRACE_SCHEMA", "RESPONSE_SCOPE", "PATH_ENDPOINTS", "DIRECTED_EDGE",
          "EDGE_WEIGHT", "COST_SUM", "MINIMALITY", "BFS_FIFO", "BFS_NEW_NEIGHBORS",
          "BFS_COMPLETENESS", "TREE_START", "TREE_CHILD_EDGE", "TREE_EVENT_ORDER", "TREE_COMPLETENESS", "DFA_START", "DFA_COMPLETE_WORD",
          "DFA_TRANSITION", "DFA_ACCEPTANCE"}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _task(question: str, source: str) -> tuple[str, str]:
    from .generators import graph, graph_pair

    if source == graph_pair.GENERATOR_VERSION and question == graph_pair._QUESTION:
        return "shortest", ""
    if source == graph.GENERATOR_VERSION:
        for kind in ("shortest", "bfs", "dfa", "tree"):
            for traversal in (("preorder", "inorder", "postorder") if kind == "tree" else ("postorder",)):
                if question == graph._question(kind, traversal):
                    return kind, traversal if kind == "tree" else ""
    raise ValueError("trace: unsupported canonical question/source")


def instruction_for_question(question: str, source: str) -> str:
    """Public output shape and task rules only; no generated values or teacher trace."""
    if source == "gen-circuit-v1":
        from .circuit_trace import instruction_for_question as circuit_instruction
        return circuit_instruction(question, source)
    kind, traversal = _task(question, source)
    pair = source == "gen-graph-pair-v1"
    fields = {
        "shortest": ('kind: "shortest"; nodes: array of node-label strings in path order; '
                     'edge_ids: ' + ('array of edge-ID strings in path order' if pair else
                                    'the JSON null value (unquoted, not "null")')
                     + '; weights: array of integer weights used for each edge; cost: integer sum'),
        "bfs": ('kind: "bfs"; steps: array of objects with exactly node (dequeued node-label string) '
                'and new (array of newly enqueued node-label strings)'),
        "tree": ('kind: "tree"; traversal: "' + traversal + '"; events: array of objects. '
                 'Each event has exactly op and node for op "enter", "emit", or "exit", or exactly '
                 'op, parent, side, and node for op "child". op and side are JSON strings; '
                 'parent and node are node-label strings; side is "L" or "R". '
                 'Enter each node, follow each present child before entering it, emit the node at '
                 'the requested traversal position, then exit; visit left before right'),
        "dfa": ('kind: "dfa"; states: array of state-label strings, initial state then one state '
                'per symbol of 0110; decision: the JSON string "accept" or "reject"'),
    }
    return (f"Trace contract {VERSION}: make the FIRST nonempty line after Solution: exactly "
            "Trace: followed by one JSON object on the same line. Assert your own executed trace, "
            "not a quotation, example, hypothetical, or negation. Within the Trace JSON, all node/state "
            "labels and edge IDs must be double-quoted JSON strings. Numeric trace fields must be JSON integers, not "
            "strings, decimals, or booleans. Use actual labels and values; do not copy descriptions "
            "or placeholders. Use exactly the fields in FIELD SPECIFICATIONS (descriptions, not "
            f"output JSON): {fields[kind]}. Include no extra trace lines or code fences. "
            "Before Solution, use only Observed facts with assignment lines. Numeric edge[...] and "
            "weight[...] values must be plain unsigned ASCII decimal integers, with no units or prose. After the trace, "
            "write only one Final answer: line. Section headers may end with spaces/tabs; the trace line "
            "may have spaces/tabs before Trace:, after its colon, and after the JSON object. "
            "Use LF or CRLF line breaks only. Put all executed operations in the trace.")


def _contract(actor: ActorItem, item: VerifierItem) -> tuple[dict, str, str]:
    from .generators import graph, graph_pair

    kind, traversal = _task(actor.question, actor.source)
    world = item.accept["graph_world"]
    if len(json.dumps(world, allow_nan=False)) > 16384:
        raise ValueError("trace: oversized world")
    try:
        graph._validate_world(world)
    except graph.nx.NetworkXException as exc:
        raise ValueError("trace: invalid native world") from exc
    pair = actor.source == graph_pair.GENERATOR_VERSION
    facts = ({key: value for i, (a, b, weight) in enumerate(world["edges"])
              for key, value in ((f"dir[e{i}]", f"{a}->{b}"), (f"weight[e{i}]", str(weight)))}
             if pair else graph._facts(world))
    types = ({key: "relation" if key.startswith("dir[") else "number" for key in facts}
             if pair else graph._slot_types(facts))
    answer_type = "int" if kind == "shortest" else "short_text" if kind == "dfa" else "list"
    policy = {"ordered": True, "element_type": "short_text"} if answer_type == "list" else {}
    if (actor.problem_id != item.problem_id or not actor.problem_id.startswith(actor.source + ":graph:")
            or actor.family != "graph" or actor.choices or item.choices or world["kind"] != kind
            or actor.answer_format != answer_type or item.answer_type != answer_type
            or item.accept != {"graph_world": world, **policy} or item.slots != facts
            or item.slot_types != types or item.required_slots != tuple(facts)):
        raise ValueError("trace: native provenance/fact contract differs")
    return world, kind, traversal


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("trace: duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("trace: nonfinite JSON value")


def _trace(text: str) -> tuple[dict, dict]:
    if len(text) > 131072 or text.count("Trace:") != 1 or "```" in text or "~~~" in text:
        raise ValueError("trace: missing, repeated, fenced, or oversized trace")
    headers = list(re.finditer(r"^Solution:[ \t]*\r?$", text, re.M))
    if len(headers) != 1:
        raise ValueError("trace: unique Solution header required")
    body = text[headers[0].end():]
    match = re.match(r"(?:[ \t]*\r?\n)*(?P<line>[ \t]*Trace:[ \t]*(?P<payload>\{[^\r\n]*\})[ \t]*)(?:\r?$)", body, re.M)
    if not match or len(match["line"]) > 4096:
        raise ValueError("trace: first solution line must assert bounded JSON")
    # An explicit disavowal cannot turn a quoted or hypothetical trace into evidence.
    if re.search(r"\b(?:hypothetical|example|quoted|ignore|not|never|invalid|false)\b[^\n]{0,60}\btrace\b"
                 r"|\btrace\b[^\n]{0,60}\b(?:hypothetical|example|quoted|invalid|false)\b", body, re.I):
        raise ValueError("trace: disavowed trace")
    start = headers[0].end() + match.start("line")
    end = headers[0].end() + match.end("line")
    payload = json.loads(match["payload"], object_pairs_hook=_object, parse_constant=_reject_constant)
    if type(payload) is not dict:
        raise ValueError("trace: JSON object required")
    line = text[start:end]
    return payload, {"line": line, "span": [start, end], "sha256": _sha(line)}


def _labels(value: Any) -> bool:
    return (type(value) is list and len(value) <= 32
            and all(type(v) is str and re.fullmatch(r"[A-Za-z0-9_]{1,16}", v) for v in value))


def native_integer_value(value: object) -> bool:
    """Native dimensionless syntax: 0 or nonzero-led ASCII digits (<=32), outer space/tab only.

    No sign, leading zeros, decimal, exponent, fraction, Unicode digit, unit, or prose is
    accepted. This predicate never rewrites the original value or certifies its correctness.
    """
    return type(value) is str and re.fullmatch(r"[ \t]*(?:0|[1-9][0-9]{0,31})[ \t]*", value) is not None


def native_scalar_line(value: object) -> bool:
    """A nonblank printable/tab-only scalar with no splitlines separators; no rewriting."""
    return (type(value) is str and bool(value.strip(" \t"))
            and value.splitlines() == [value] and all(char == "\t" or char.isprintable() for char in value))


def _response_scope(text: str, trace_end: int) -> bool:
    if not native_scalar_line(text.replace("\r\n", "").replace("\n", "")):
        return False
    header = re.search(r"^Solution:[ \t]*\r?$", text, re.M)
    if header is None:
        return False
    before = text[:header.start()]
    prefix = before
    facts_header = re.match(r"Observed facts:[ \t]*\r?\n", before)
    if facts_header is not None:
        prefix = before[facts_header.end():]
        for start, end, key, value in reversed(observation_spans(text)):
            if end <= header.start():
                if key.startswith(("edge[", "weight[")) and not native_integer_value(value):
                    return False
                start, end = start - facts_header.end(), end - facts_header.end()
                prefix = prefix[:start] + prefix[end:]
    return not prefix.strip() and re.fullmatch(r"Final answer: [^\r\n]+", text[trace_end:].strip()) is not None


def _schema(trace: dict, kind: str, traversal: str, pair: bool) -> bool:
    if trace.get("kind") != kind:
        return False
    if kind == "shortest":
        return (set(trace) == {"kind", "nodes", "edge_ids", "weights", "cost"}
                and _labels(trace["nodes"]) and 2 <= len(trace["nodes"]) <= 32
                and set(trace["nodes"]) <= set("ABCDEFGH")
                and (trace["edge_ids"] is None if not pair else _labels(trace["edge_ids"]))
                and type(trace["weights"]) is list and len(trace["weights"]) == len(trace["nodes"]) - 1
                and all(type(v) is int and 1 <= v <= 1000000 for v in trace["weights"])
                and type(trace["cost"]) is int and 0 <= trace["cost"] <= 32000000
                and (not pair or len(trace["edge_ids"]) == len(trace["weights"])
                     and all(re.fullmatch(r"e(0|[1-9][0-9]?)", value) for value in trace["edge_ids"])))
    if kind == "bfs":
        return (set(trace) == {"kind", "steps"} and type(trace["steps"]) is list
                and 1 <= len(trace["steps"]) <= 32 and all(type(step) is dict
                    and set(step) == {"node", "new"} and _labels([step["node"]])
                    and _labels(step["new"]) for step in trace["steps"]))
    if kind == "tree":
        return (set(trace) == {"kind", "traversal", "events"} and trace["traversal"] == traversal
                and type(trace["events"]) is list and 1 <= len(trace["events"]) <= 32
                and all(type(event) is dict and (
                    set(event) == {"op", "node"} and event["op"] in ("enter", "emit", "exit")
                    and _labels([event["node"]]) or set(event) == {"op", "parent", "side", "node"}
                    and event["op"] == "child" and event["side"] in ("L", "R")
                    and _labels([event["parent"], event["node"]])) for event in trace["events"]))
    return (set(trace) == {"kind", "states", "decision"} and _labels(trace["states"])
            and bool(trace["states"]) and trace["decision"] in ("accept", "reject"))


def _shift_evidence(report: dict, delta: int, lower: int, upper: int) -> dict:
    """Copy each evidence location; checks/issues may alias the same row in _finish."""
    def span(value: object) -> list[int]:
        if (type(value) is not list or len(value) != 2 or any(type(v) is not int for v in value)
                or not lower <= value[0] < value[1] <= upper):
            raise ValueError("trace: invalid projection span")
        return [value[0] + delta, value[1] + delta]

    shifted = dict(report)
    if report["trace"] is not None:
        if type(report["trace"]) is not dict:
            raise ValueError("trace: invalid projection trace")
        shifted["trace"] = {**report["trace"], "span": span(report["trace"]["span"])}
    for field in ("checks", "issues"):
        if (type(report[field]) is not list or any(
                type(row) is not dict or type(row.get("observation_refs")) is not list
                or any(type(ref) is not dict for ref in row["observation_refs"])
                for row in report[field])):
            raise ValueError("trace: invalid projection rows")
        shifted[field] = [{**row, "observation_refs": [
            {**ref, "span": span(ref["span"])} for ref in row["observation_refs"]]}
            for row in report[field]]
    return shifted


def check_trace(*, actor: ActorItem, item: VerifierItem, text: str, finish_reason: str,
                response_contract: str | None = None) -> dict:
    """Gold-independent audit. A true final answer never erases a false trace operation."""
    if response_contract is not None:
        sections = parse_retry_sections(text, contract_version=response_contract)
        solution = project_solution(text, sections)
        projected = check_trace(actor=actor, item=item, text=solution, finish_reason=finish_reason)
        projected = _shift_evidence(projected, sections.solution_span.start, 0, len(solution))
        projected.update(version=projected["version"] + "+" + response_contract,
                         response_contract=response_contract, text_sha256=sections.raw_sha,
                         solution_projection={"span": [sections.solution_span.start, sections.solution_span.end],
                                              "sha256": _sha(solution)})
        return projected
    if actor.source == "gen-circuit-v1":
        from .circuit_trace import check_trace as check_circuit
        return check_circuit(actor=actor, item=item, text=text, finish_reason=finish_reason)
    checks: list[dict] = []
    report: dict = {"version": VERSION, "status": "UNVERIFIABLE", "trace": None, "checks": checks,
                    "issues": [], "coverage": {"kind": None, "checked": [], "unverified": [], "complete": False},
                    "semantic_scope": _SCOPE, "finish_reason": finish_reason, "text_sha256": _sha(text)}
    spans = observation_spans(text) if len(text) <= 131072 else []

    def add(rule: str, locus: str, valid: bool | None, keys: tuple[str, ...] = (),
            blocked_by: tuple[dict, ...] = ()) -> None:
        refs = []
        normalized = {slot_id(k): k for k in item.slots} if keys else {}
        for start, end, key, value in spans:
            if key not in {slot_id(k) for k in keys} or key not in normalized:
                continue
            original = normalized[key]
            observed = (_normalize(value, item.slot_types[original], key)
                        if item.slot_types[original] != "number" or native_integer_value(value) else None)
            expected = _normalize(item.slots[original], item.slot_types[original], key)
            refs.append({"slot": key, "span": [start, end], "sha256": _sha(text[start:end]),
                         "status": "UNVERIFIABLE" if observed is None else "MATCH" if observed == expected else "MISMATCH"})
        checks.append({"rule": rule, "locus": locus, "observation_refs": refs,
                       "blocked_by": list(blocked_by),
                       "status": "UNVERIFIABLE" if valid is None else "COMPLIANT" if valid else "VIOLATED"})

    try:
        world, kind, traversal = _contract(actor, item)
        report["coverage"]["kind"] = kind
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RecursionError):
        add("NATIVE_CONTRACT", "input", None)
        return _finish(report)
    try:
        trace, report["trace"] = _trace(text)
    except (ValueError, RecursionError):
        add("TRACE_FORMAT", "solution", None)
        return _finish(report)
    from .generators import graph

    pair = actor.source == "gen-graph-pair-v1"
    if not _schema(trace, kind, traversal, pair):
        add("TRACE_SCHEMA", "trace", None)
        return _finish(report)
    if len(spans) > 64 or len({key for _, _, key, _ in spans}) != len(spans):
        add("TRACE_FORMAT", "solution", None)
        return _finish(report)
    complete_scope = _response_scope(text, report["trace"]["span"][1])
    add("RESPONSE_SCOPE", "response", True if complete_scope else None)
    keys: tuple[str, ...]
    blocked: tuple[dict, ...]
    if kind == "shortest":
        nodes, weights, ids = trace["nodes"], trace["weights"], trace["edge_ids"]
        add("PATH_ENDPOINTS", "nodes", nodes[0] == "A" and nodes[-1] == "H")
        for i, (a, b) in enumerate(zip(nodes, nodes[1:])):
            edge = None
            if pair:
                match = re.fullmatch(r"e(0|[1-9][0-9]?)", ids[i])
                index = int(match[1]) if match else len(world["edges"])
                edge = world["edges"][index] if index < len(world["edges"]) else None
                keys = (f"dir[{ids[i]}]", f"weight[{ids[i]}]")
            else:
                edge = next((e for e in world["edges"] if e[:2] == [a, b]), None)
                keys = (f"edge[{a},{b}]", f"edge[{b},{a}]")
            locus = f"nodes[{i}:{i+2}]"
            add("DIRECTED_EDGE", locus, edge is not None and edge[:2] == [a, b], keys)
            blocked = ({"rule": "DIRECTED_EDGE", "locus": locus},) if edge is None else ()
            add("EDGE_WEIGHT", f"weights[{i}]", None if edge is None else weights[i] == edge[2], keys, blocked)
        add("COST_SUM", "cost", trace["cost"] == sum(weights))
        failed = tuple({"rule": row["rule"], "locus": row["locus"]} for row in checks
                       if row["rule"] in {"PATH_ENDPOINTS", "DIRECTED_EDGE", "EDGE_WEIGHT", "COST_SUM"}
                       and row["status"] == "VIOLATED")
        add("MINIMALITY", "nodes", None if failed else sum(weights) == graph._dijkstra(world), blocked_by=failed)
    elif kind == "bfs":
        queue, seen = ["A"], {"A"}
        blocked = ()
        for i, step in enumerate(trace["steps"]):
            if blocked:
                add("BFS_FIFO", f"steps[{i}].node", None, blocked_by=blocked)
                add("BFS_NEW_NEIGHBORS", f"steps[{i}].new", None, blocked_by=blocked)
                continue
            node = step["node"]
            fifo = bool(queue) and node == queue[0]
            add("BFS_FIFO", f"steps[{i}].node", fifo)
            if not fifo:
                blocked = ({"rule": "BFS_FIFO", "locus": f"steps[{i}].node"},)
                add("BFS_NEW_NEIGHBORS", f"steps[{i}].new", None, blocked_by=blocked)
                continue
            queue.pop(0)
            new = sorted(b for a, b, _ in world["edges"] if a == node and b not in seen)
            keys = tuple(f"edge[{a},{b}]" for a, b, _ in world["edges"] if a == node)
            matches = step["new"] == new
            add("BFS_NEW_NEIGHBORS", f"steps[{i}].new", matches, keys)
            if not matches:
                blocked = ({"rule": "BFS_NEW_NEIGHBORS", "locus": f"steps[{i}].new"},)
                continue
            queue.extend(step["new"])
            seen.update(step["new"])
        add("BFS_COMPLETENESS", "steps", None if blocked else not queue, blocked_by=blocked)
    elif kind == "tree":
        def events(node):
            yield {"op": "enter", "node": node}
            if traversal == "preorder":
                yield {"op": "emit", "node": node}
            for side in ("L", "R"):
                child = world["children"].get(node, {}).get(side)
                if child is not None:
                    yield {"op": "child", "parent": node, "side": side, "node": child}
                    yield from events(child)
                if side == "L" and traversal == "inorder":
                    yield {"op": "emit", "node": node}
            if traversal == "postorder":
                yield {"op": "emit", "node": node}
            yield {"op": "exit", "node": node}

        claimed = trace["events"]
        add("TREE_START", "events[0]", claimed[0] == {"op": "enter", "node": world["root"]}, ("root",))
        expected_events = list(events(world["root"]))
        for i, event in enumerate(claimed):
            keys = ((f"child[{event['parent']},{event['side']}]",) if event["op"] == "child" else ())
            if event["op"] == "child":
                add("TREE_CHILD_EDGE", f"events[{i}]", world["children"].get(event["parent"], {}).get(
                    event["side"]) == event["node"], keys)
            add("TREE_EVENT_ORDER", f"events[{i}]", i < len(expected_events) and event == expected_events[i], keys)
        add("TREE_COMPLETENESS", "events", len(claimed) == len(expected_events))
    else:
        states = trace["states"]
        add("DFA_START", "states[0]", states[0] == world["start"], ("start",))
        add("DFA_COMPLETE_WORD", "states", len(states) == 5)
        for i, symbol in enumerate("0110"):
            if i + 1 >= len(states):
                break
            targets = world["trans"].get(states[i])
            add("DFA_TRANSITION", f"states[{i}:{i+2}]",
                targets is not None and states[i + 1] == targets[symbol], (f"trans[{states[i]},{symbol}]",))
        blocked = (({"rule": "DFA_COMPLETE_WORD", "locus": "states"},) if len(states) != 5
                   else ({"rule": "DFA_TRANSITION", "locus": "states[3:5]"},)
                   if states[-1] not in world["trans"] else ())
        add("DFA_ACCEPTANCE", "decision", (trace["decision"] == ("accept" if states[-1] in world["accepting"]
            else "reject")) if not blocked else None, ("accepting",), blocked_by=blocked)
    return _finish(report)


def _finish(report: dict) -> dict:
    rows = report["checks"]
    report["issues"] = [row for row in rows if row["status"] != "COMPLIANT"]
    report["status"] = ("VIOLATED" if any(row["status"] == "VIOLATED" for row in rows)
                        else "UNVERIFIABLE" if report["issues"] else "COMPLIANT")
    report["coverage"]["checked"] = sorted({r["rule"] for r in rows if r["status"] != "UNVERIFIABLE"})
    report["coverage"]["unverified"] = sorted({r["rule"] for r in rows if r["status"] == "UNVERIFIABLE"})
    report["coverage"]["complete"] = not report["coverage"]["unverified"]
    return report


def _inventory(trace: dict) -> list[tuple[str, str]]:
    """Derive exact public regions from authored JSON, without solving the private world."""
    if trace.get("kind") == "circuit_ladder":
        from .circuit_trace import inventory
        return inventory(trace)
    rows = [("RESPONSE_SCOPE", "response")]
    kind = trace["kind"]
    if kind == "shortest":
        rows.append(("PATH_ENDPOINTS", "nodes"))
        for i in range(len(trace["weights"])):
            rows.extend((("DIRECTED_EDGE", f"nodes[{i}:{i+2}]"), ("EDGE_WEIGHT", f"weights[{i}]")))
        rows.extend((("COST_SUM", "cost"), ("MINIMALITY", "nodes")))
    elif kind == "bfs":
        for i in range(len(trace["steps"])):
            rows.extend((("BFS_FIFO", f"steps[{i}].node"), ("BFS_NEW_NEIGHBORS", f"steps[{i}].new")))
        rows.append(("BFS_COMPLETENESS", "steps"))
    elif kind == "tree":
        rows.append(("TREE_START", "events[0]"))
        for i, event in enumerate(trace["events"]):
            if event["op"] == "child":
                rows.append(("TREE_CHILD_EDGE", f"events[{i}]"))
            rows.append(("TREE_EVENT_ORDER", f"events[{i}]"))
        rows.append(("TREE_COMPLETENESS", "events"))
    else:
        rows.extend((("DFA_START", "states[0]"), ("DFA_COMPLETE_WORD", "states")))
        rows.extend(("DFA_TRANSITION", f"states[{i}:{i+2}]") for i in range(min(4, len(trace["states"]) - 1)))
        rows.append(("DFA_ACCEPTANCE", "decision"))
    return rows


def _dependencies(rows: list[dict], trace: dict | None) -> None:
    earlier: dict[tuple[str, str], dict] = {}
    failed_path: list[dict] = []
    failed_bfs: list[dict] = []
    for row in rows:
        identity = row["rule"], row["locus"]
        expected: list[dict] = []
        if row["status"] != "UNVERIFIABLE" and row["blocked_by"]:
            raise ValueError("definite check has dependencies")
        for dependency in row["blocked_by"]:
            if (type(dependency) is not dict or set(dependency) != {"rule", "locus"}
                    or type(dependency["rule"]) is not str or type(dependency["locus"]) is not str):
                raise ValueError("dependency contract")
            prior = earlier.get((dependency["rule"], dependency["locus"]))
            if prior is None or prior["status"] != "VIOLATED":
                raise ValueError("dependency is not an earlier proven failure")
        if row["rule"] == "EDGE_WEIGHT" and row["status"] == "UNVERIFIABLE":
            match = re.fullmatch(r"weights\[([0-9]+)\]", row["locus"])
            if match is None:
                raise ValueError("unbound weight region")
            i = int(match[1])
            expected = [{"rule": "DIRECTED_EDGE", "locus": f"nodes[{i}:{i+2}]"}]
        elif row["rule"] == "MINIMALITY":
            expected = failed_path
        elif row["rule"] in {"BFS_FIFO", "BFS_NEW_NEIGHBORS", "BFS_COMPLETENESS"}:
            expected = failed_bfs
        elif row["rule"] == "DFA_ACCEPTANCE":
            if trace is None or trace.get("kind") != "dfa":
                raise ValueError("unbound DFA acceptance region")
            if len(trace["states"]) != 5:
                expected = [{"rule": "DFA_COMPLETE_WORD", "locus": "states"}]
            elif trace["states"][-1] not in {"q0", "q1", "q2", "q3"}:
                expected = [{"rule": "DFA_TRANSITION", "locus": "states[3:5]"}]
        if row["blocked_by"] != expected or expected and row["status"] != "UNVERIFIABLE":
            raise ValueError("unsupported, incomplete, or altered dependency inventory")
        if row["status"] == "UNVERIFIABLE" and not expected and row["rule"] not in {
                "TRACE_FORMAT", "NATIVE_CONTRACT", "TRACE_SCHEMA", "RESPONSE_SCOPE"}:
            raise ValueError("unexplained operation uncertainty")
        if identity in earlier:
            raise ValueError("duplicate check identity")
        earlier[identity] = row
        if row["status"] == "VIOLATED":
            ref = {"rule": row["rule"], "locus": row["locus"]}
            if row["rule"] in {"PATH_ENDPOINTS", "DIRECTED_EDGE", "EDGE_WEIGHT", "COST_SUM"}:
                failed_path.append(ref)
            if row["rule"] in {"BFS_FIFO", "BFS_NEW_NEIGHBORS"} and not failed_bfs:
                failed_bfs = [ref]


def validate_trace_report(report: dict, text: str, finish_reason: str, *,
                          response_contract: str | None = None) -> None:
    """Validate bounded JSON and exact original evidence binding across a process boundary."""
    if type(report) is not dict:
        raise ValueError("trace: invalid report or original evidence binding")
    if response_contract is not None:
        try:
            sections = parse_retry_sections(text, contract_version=response_contract)
            solution = project_solution(text, sections)
            if (len(json.dumps(report, allow_nan=False)) > 65536 or type(report) is not dict
                    or set(report) != {"version", "status", "trace", "checks", "issues", "coverage",
                                      "semantic_scope", "finish_reason", "text_sha256",
                                      "response_contract", "solution_projection"}
                    or report["response_contract"] != response_contract
                    or report["text_sha256"] != sections.raw_sha
                    or type(report["version"]) is not str
                    or not report["version"].endswith("+" + response_contract)):
                raise ValueError("report projection contract")
            projection = report["solution_projection"]
            start, end = sections.solution_span.start, sections.solution_span.end
            if (type(projection) is not dict or set(projection) != {"span", "sha256"}
                    or type(projection["span"]) is not list or len(projection["span"]) != 2
                    or any(type(v) is not int for v in projection["span"])
                    or projection["span"] != [start, end] or projection["sha256"] != _sha(solution)):
                raise ValueError("report projection binding")
            original = _shift_evidence(report, -start, start, end)
            del original["response_contract"], original["solution_projection"]
            original["version"] = report["version"][:-len("+" + response_contract)]
            original["text_sha256"] = _sha(solution)
            validate_trace_report(original, solution, finish_reason)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, RecursionError) as exc:
            raise ValueError("trace: invalid report or original evidence binding") from exc
        return
    if report.get("version") == "native-circuit-ladder-trace-v1":
        from .circuit_trace import validate_report
        validate_report(report, text, finish_reason)
        return
    try:
        if (len(json.dumps(report, allow_nan=False)) > 65536 or set(report) != {
                "version", "status", "trace", "checks", "issues", "coverage", "semantic_scope",
                "finish_reason", "text_sha256"} or report["version"] != VERSION
                or report["text_sha256"] != _sha(text) or report["finish_reason"] != finish_reason
                or type(finish_reason) is not str or len(finish_reason) > 32
                or report["semantic_scope"] != _SCOPE or report["status"] not in _STATUSES):
            raise ValueError("report contract")
        trace = report["trace"]
        payload = None
        if trace is not None:
            payload, expected = _trace(text)
            if (type(trace) is not dict or set(trace) != {"line", "span", "sha256"}
                    or type(trace["span"]) is not list or len(trace["span"]) != 2
                    or any(type(v) is not int for v in trace["span"]) or trace != expected):
                raise ValueError("trace binding")
        elif report["status"] != "UNVERIFIABLE":
            raise ValueError("missing authored trace")
        observed = {(s, e, k): (_sha(text[s:e]), value) for s, e, k, value in observation_spans(text)}
        rows = report["checks"]
        if type(rows) is not list or not 1 <= len(rows) <= 128:
            raise ValueError("check bounds")
        for row in rows:
            if (type(row) is not dict or set(row) != {"rule", "locus", "observation_refs", "status", "blocked_by"}
                    or row["rule"] not in _RULES or row["status"] not in _STATUSES
                    or type(row["locus"]) is not str or len(row["locus"]) > 96
                    or type(row["observation_refs"]) is not list or len(row["observation_refs"]) > 64
                    or type(row["blocked_by"]) is not list or len(row["blocked_by"]) > 64):
                raise ValueError("check contract")
            loci = {
                "NATIVE_CONTRACT": r"input", "TRACE_FORMAT": r"solution", "TRACE_SCHEMA": r"trace",
                "RESPONSE_SCOPE": r"response", "PATH_ENDPOINTS": r"nodes", "COST_SUM": r"cost",
                "MINIMALITY": r"nodes", "DIRECTED_EDGE": r"nodes\[[0-9]{1,2}:[0-9]{1,2}\]",
                "EDGE_WEIGHT": r"weights\[[0-9]{1,2}\]",
                "BFS_FIFO": r"steps\[[0-9]{1,2}\]\.node", "BFS_NEW_NEIGHBORS": r"steps\[[0-9]{1,2}\]\.new",
                "BFS_COMPLETENESS": r"steps", "TREE_START": r"events\[0\]",
                "TREE_CHILD_EDGE": r"events\[[0-9]{1,2}\]", "TREE_EVENT_ORDER": r"events\[[0-9]{1,2}\]",
                "TREE_COMPLETENESS": r"events", "DFA_START": r"states\[0\]",
                "DFA_COMPLETE_WORD": r"states", "DFA_TRANSITION": r"states\[[0-3]:[2-5]\]",
                "DFA_ACCEPTANCE": r"decision",
            }
            if not re.fullmatch(loci[row["rule"]], row["locus"]):
                raise ValueError("check locus")
            for ref in row["observation_refs"]:
                if (type(ref) is not dict or set(ref) != {"slot", "span", "sha256", "status"}
                        or ref["status"] not in {"MATCH", "MISMATCH", "UNVERIFIABLE"}
                        or type(ref["span"]) is not list or len(ref["span"]) != 2
                        or any(type(v) is not int for v in ref["span"])
                        or (*ref["span"], ref["slot"]) not in observed
                        or observed[(*ref["span"], ref["slot"])][0] != ref["sha256"]):
                    raise ValueError("observation binding")
                value = observed[(*ref["span"], ref["slot"])][1]
                if (ref["slot"].startswith(("edge[", "weight[")) and not native_integer_value(value)
                        and ref["status"] != "UNVERIFIABLE"):
                    raise ValueError("native numeric syntax differs from original observation")
        coverage = report["coverage"]
        if (type(coverage) is not dict or set(coverage) != {"kind", "checked", "unverified", "complete"}
                or type(coverage["complete"]) is not bool
                or coverage["kind"] not in (None, "shortest", "bfs", "tree", "dfa")
                or report != _finish({**report, "coverage": dict(coverage)})):
            raise ValueError("report consistency")
        _dependencies(rows, payload)
        identities = [(row["rule"], row["locus"]) for row in rows]
        if len(rows) == 1 and rows[0]["rule"] in {"NATIVE_CONTRACT", "TRACE_FORMAT", "TRACE_SCHEMA"}:
            if rows[0]["status"] != "UNVERIFIABLE" or rows[0]["observation_refs"]:
                raise ValueError("invalid unsupported check")
        else:
            kind = coverage["kind"]
            if (payload is None or kind not in {"shortest", "bfs", "tree", "dfa"}
                    or not _schema(payload, kind, payload.get("traversal", ""), payload.get("edge_ids") is not None)
                    or identities != _inventory(payload)):
                raise ValueError("missing or unbound operation regions")
            scope_status = "COMPLIANT" if _response_scope(text, trace["span"][1]) else "UNVERIFIABLE"
            if rows[0]["status"] != scope_status:
                raise ValueError("response scope differs from original text")
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, RecursionError) as exc:
        raise ValueError("trace: invalid report or original evidence binding") from exc
