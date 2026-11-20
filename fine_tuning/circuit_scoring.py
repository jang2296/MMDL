"""Strict CPU scoring for the bounded v2 Circuit response contract.

Only compare submitted fields with separately supplied targets; never solve a
circuit or rescue a candidate using gold. This is not professional G7 or a
certification of circuit/image semantics. No model/evaluator imports are needed.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from typing import Any

VERSION = CONTRACT_VERSION = "independent-dc-circuit-bridge-v2"
MAX_RESPONSE_CHARS = 16384
MAX_NUMBER_CHARS = 128
NODES = {"N0", "N1", "N2", "N3"}
COMPONENTS = {"C0", "C1", "C2", "C3", "C4"}
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
RATIO = re.compile(r"[+-]?[0-9]+/[+-]?[0-9]+\Z")


def _object(value: Any, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("Exactly the registered object fields are required")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Repeated JSON key")
        value[key] = item
    return value


def _nonfinite(value: str) -> Any:
    raise ValueError("Nonfinite JSON constant")


def _json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw, object_pairs_hook=_unique_pairs,
                           parse_float=Decimal, parse_constant=_nonfinite)
    except (ValueError, RecursionError) as error:
        raise ValueError("Exactly one bare complete JSON object required") from error
    if type(value) is not dict:
        raise ValueError("Exactly one bare complete JSON object required")
    return value


def _letter(value: Any) -> str:
    if type(value) is not str or value not in ("A", "B", "C", "D"):
        raise ValueError("Exactly one uppercase A-D letter required")
    return value


def _number(value: Any) -> str:
    """Normalize a finite exact scalar, with no unit or expression interpretation."""
    if type(value) not in (str, int, Decimal):
        raise ValueError("Rational string or finite JSON number required")
    text = str(value)
    if not 1 <= len(text) <= MAX_NUMBER_CHARS:
        raise ValueError("Numeric literal exceeds the bounded contract")
    if RATIO.fullmatch(text):
        try:
            return str(Fraction(text))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("Invalid exact rational") from error
    if not NUMBER.fullmatch(text):
        raise ValueError("Invalid numeric literal; units and expressions are forbidden")
    decimal = Decimal(text)
    exponent = decimal.as_tuple().exponent
    if (not decimal.is_finite() or type(exponent) is not int or abs(exponent) > MAX_NUMBER_CHARS
            or abs(decimal.adjusted()) > MAX_NUMBER_CHARS):
        raise ValueError("Numeric exponent exceeds the bounded contract")
    return str(Fraction(decimal))


def _structure(value: dict[str, Any]) -> dict[str, Any]:
    _object(value, {"nodes", "reference_node", "components"})
    nodes = value["nodes"]
    if (type(nodes) is not list or len(nodes) != 4
            or any(type(node) is not str for node in nodes) or set(nodes) != NODES
            or type(value["reference_node"]) is not str or value["reference_node"] not in NODES):
        raise ValueError("Exactly N0-N3 and a registered reference node required")
    components = value["components"]
    if type(components) is not list or len(components) != 5:
        raise ValueError("Exactly five registered components required")
    seen: set[str] = set()
    kinds: Counter[str] = Counter()
    for item in components:
        if type(item) is not dict:
            raise ValueError("Component object required")
        kind = item.get("type")
        if kind not in ("RESISTOR", "DC_SOURCE", "SWITCH"):
            raise ValueError("Unregistered component type")
        keys = {"id", "type", "pins"} | ({"state"} if kind == "SWITCH" else {"value", "unit"})
        if kind == "DC_SOURCE":
            keys.add("positive_pin")
        _object(item, keys)
        cid, pins = item["id"], item["pins"]
        if type(cid) is not str or cid not in COMPONENTS or cid in seen:
            raise ValueError("Unknown or repeated component ID")
        if (type(pins) is not list or len(pins) != 2
                or any(type(pin) is not str or pin not in NODES for pin in pins)
                or pins[0] == pins[1]):
            raise ValueError("Two distinct registered ordered pins required")
        if kind == "SWITCH":
            if item["state"] not in ("OPEN", "CLOSED"):
                raise ValueError("Unknown switch state")
        else:
            if (type(item["value"]) is not int or not 1 <= item["value"] <= 48
                    or item["unit"] != ("ohm" if kind == "RESISTOR" else "V")):
                raise ValueError("Bounded integer component value and exact unit required")
            if kind == "DC_SOURCE" and (type(item["positive_pin"]) is not str
                                         or item["positive_pin"] not in pins):
                raise ValueError("Source polarity must designate one of its pins")
        seen.add(cid)
        kinds[kind] += 1
    if kinds != {"RESISTOR": 3, "DC_SOURCE": 1, "SWITCH": 1}:
        raise ValueError("Three resistors, one source and one switch required")
    # Preserve ordered pins: they also fix the reported current directions.
    return dict(value, nodes=sorted(nodes), components=sorted(components, key=lambda item: item["id"]))


def _grounded(value: dict[str, Any]) -> dict[str, Any]:
    _object(value, {"source_current_A", "node_voltages_V", "resistor_currents_A", "option"})
    voltages = _object(value["node_voltages_V"], NODES)
    currents = _object(value["resistor_currents_A"], {"C0", "C1", "C2"})
    return {"source_current_A": _number(value["source_current_A"]),
            "node_voltages_V": {node: _number(voltages[node]) for node in sorted(NODES)},
            "resistor_currents_A": {cid: _number(currents[cid]) for cid in sorted(currents)},
            "option": _letter(value["option"])}


def extract_response(raw: str, view: str, *, contract_version: str = VERSION) -> str | dict[str, Any]:
    """Gold-independent extraction; unknown contracts and extra fields fail closed.

    qa accepts only a bare A-D letter. structure/grounded require a bare JSON
    object (no fences/prose). Only node/component array order is normalized;
    pins, IDs, units and polarity remain exact. Rational scalars become strings.
    """
    if contract_version != VERSION or view not in ("qa", "structure", "grounded"):
        raise ValueError("Unknown Circuit version or view")
    if type(raw) is not str or not 1 <= len(raw) <= MAX_RESPONSE_CHARS:
        raise ValueError("Bounded nonempty response string required")
    if view == "qa":
        return _letter(raw.strip())
    value = _json(raw)
    return _structure(value) if view == "structure" else _grounded(value)


def score_circuit(raw: str, target: str, view: str, finish_reason: str = "stop", *,
                  contract_version: str = VERSION) -> dict[str, Any]:
    """Compare with a validated target after extraction; malformed gold raises.

    Only natural stop/eos is accepted. A length/error/unknown finish never gains
    credit. This function does not check physical consistency or solve targets.
    """
    if contract_version != VERSION or view not in ("qa", "structure", "grounded"):
        raise ValueError("Unknown Circuit version or view")
    result: dict[str, Any] = {"correct": False, "parse_status": "NO_PARSE",
                              "extracted_answer": None, "metric_kind": "circuit_"+view}
    if view == "structure":
        result["circuit_structure_exact"] = False
    elif view == "grounded":
        result.update(answer_correct=False, circuit_state_exact=False,
                      option_exact=False, currents_exact=False)
    if finish_reason not in ("stop", "eos"):
        return result
    try:
        prediction = extract_response(raw, view, contract_version=contract_version)
    except ValueError:
        return result
    expected = extract_response(target, view, contract_version=contract_version)
    result.update(correct=prediction == expected, parse_status="PARSED", extracted_answer=prediction)
    if view == "structure":
        result["circuit_structure_exact"] = prediction == expected
    elif view == "grounded":
        assert isinstance(prediction, dict) and isinstance(expected, dict)
        answer_exact = prediction["option"] == expected["option"]
        currents_exact = all(prediction[key] == expected[key] for key in ("source_current_A", "resistor_currents_A"))
        state_exact = currents_exact and prediction["node_voltages_V"] == expected["node_voltages_V"]
        result.update(answer_correct=answer_exact, circuit_state_exact=state_exact,
                      option_exact=answer_exact, currents_exact=currents_exact)
    return result
