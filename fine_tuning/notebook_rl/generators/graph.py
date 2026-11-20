"""Bounded program-authored graph/CS problems; no benchmark, teacher, or CPM inputs."""
from __future__ import annotations

import heapq
import hashlib
import math
import random
from collections import deque

import networkx as nx
from PIL import Image, ImageDraw, ImageFont

from fine_tuning import nvw

from ..schema import Mode, Split, to_json
from .common import GeneratedProblem, build, matches_split, png_bytes, structure_hash

GENERATOR_VERSION = "gen-graph-v1"
FAMILY = "graph"
OBSERVATION_INSTRUCTION = (
    "List directed edges as edge[<u>,<v>] = <weight or 1>; DFA transitions as "
    "trans[<state>,<symbol>] = <state>, start = <state>, accepting = <comma-separated states>; "
    "tree links as child[<parent>,<L or R>] = <child>, root = <node>.\n"
)
_KINDS = ("shortest", "bfs", "dfa", "tree")
_NODES = tuple("ABCDEFGH")
_WORD = "0110"


def _network(world: dict, *, undirected: bool = False):
    graph = nx.Graph() if undirected else nx.DiGraph()
    graph.add_nodes_from(_NODES)
    graph.add_weighted_edges_from(world["edges"])
    return graph


def _dijkstra(world: dict) -> int | None:
    queue, best = [(0, "A")], {}
    while queue:
        cost, node = heapq.heappop(queue)
        if node in best:
            continue
        best[node] = cost
        if node == "H":
            return cost
        for a, b, weight in world["edges"]:
            if a == node and b not in best:
                heapq.heappush(queue, (cost + weight, b))
    return None


def _bfs(world: dict) -> list[str]:
    pending, seen, order = deque(["A"]), {"A"}, []
    while pending:
        node = pending.popleft()
        order.append(node)
        for neighbor in sorted(b for a, b, _ in world["edges"] if a == node):
            if neighbor not in seen:
                seen.add(neighbor)
                pending.append(neighbor)
    return order


def _tree_order(world: dict, traversal: str, *, iterative: bool) -> list[str]:
    children = world["children"]
    if iterative:
        stack, result = [(world["root"], False)], []
        while stack:
            node, emit = stack.pop()
            if emit:
                result.append(node)
                continue
            left, right = children.get(node, {}).get("L"), children.get(node, {}).get("R")
            events = [(left, False), (right, False)]
            events.insert({"preorder": 0, "inorder": 1, "postorder": 2}[traversal], (node, True))
            stack.extend((v, emit) for v, emit in reversed(events) if v is not None)
        return result
    def visit(node):
        if node is None:
            return []
        left = visit(children.get(node, {}).get("L"))
        right = visit(children.get(node, {}).get("R"))
        return {"preorder": [node, *left, *right], "inorder": [*left, node, *right],
                "postorder": [*left, *right, node]}[traversal]
    return visit(world["root"])


def _answers(world: dict, traversal: str = "postorder") -> tuple[str, str]:
    kind = world["kind"]
    if kind == "shortest":
        return str(_dijkstra(world)), str(nx.shortest_path_length(_network(world), "A", "H", weight="weight"))
    if kind == "bfs":
        first = _bfs(world)
        second = list(nx.bfs_tree(_network(world), "A", sort_neighbors=sorted))
        numeric = {"n": 8, "edges": [[_NODES.index(a), _NODES.index(b)] for a, b, _ in world["edges"]],
                   "initial": [0], "steps": 7, "update": "synchronous_monotone"}
        if (nvw.solve_signal(numeric) != nvw.verify_signal(numeric)
                or sorted(_NODES.index(v) for v in first) != nvw.verify_signal(numeric)[-1]):
            raise ValueError("graph: independent reachability disagreement")
        return ",".join(first), ",".join(second)
    if kind == "dfa":
        current = world["start"]
        for symbol in _WORD:
            current = world["trans"][current][symbol]
        expanded = nx.DiGraph()
        for time, symbol in enumerate(_WORD):
            expanded.add_edges_from(((time, state), (time + 1, targets[symbol]))
                                    for state, targets in world["trans"].items())
        final = any(nx.has_path(expanded, (0, world["start"]), (len(_WORD), state))
                    for state in world["accepting"] if (len(_WORD), state) in expanded)
        return ("accept" if current in world["accepting"] else "reject",
                "accept" if final else "reject")
    return (",".join(_tree_order(world, traversal, iterative=False)),
            ",".join(_tree_order(world, traversal, iterative=True)))


def _shortcut_reason(world: dict, traversal: str = "postorder") -> str | None:
    """Task-specific audits: graph_bridge.exclusion_reason only accepts signal-count episodes."""
    kind = world["kind"]
    if kind == "shortest":
        graph = _network(world)
        if not nx.has_path(graph, "A", "H"):
            return "UNREACHABLE_TARGET"
        if len(nx.shortest_path(graph, "A", "H", weight="weight")) < 3:
            return "ONE_EDGE_SHORTCUT"
        if nx.shortest_path_length(graph, "A", "H", weight="weight") == nx.shortest_path_length(
            _network(world, undirected=True), "A", "H", weight="weight"
        ):
            return "DIRECTION_FREE_SHORTCUT"
    elif kind == "bfs":
        order = _bfs(world)
        if len(order) < 3 or order == sorted(order):
            return "ONE_STEP_OR_ALPHABETICAL_SHORTCUT"
        if set(order) == set(nx.node_connected_component(_network(world, undirected=True), "A")):
            return "DIRECTION_FREE_SHORTCUT"
        if len(order) == 1 + len({b for a, b, _ in world["edges"] if a == "A"}):
            return "ONE_STEP_SHORTCUT"
    elif kind == "dfa":
        visited, current = {world["start"]}, world["start"]
        for symbol in _WORD:
            current = world["trans"][current][symbol]
            visited.add(current)
        if len(visited) < 3:
            return "TRIVIAL_DFA_TRACE"
    else:
        orders = [_tree_order(world, order, iterative=True)
                  for order in ("preorder", "inorder", "postorder")]
        if len({tuple(order) for order in orders}) != 3 or sorted(orders[0]) in orders:
            return "TRAVERSAL_OR_ALPHABETICAL_SHORTCUT"
    return None


def _sample(kind: str, rng: random.Random) -> dict | None:
    if kind in ("shortest", "bfs"):
        grid = [(i, i + 1) for i in range(8) if i % 4 < 3] + [(i, i + 4) for i in range(4)]
        edges = []
        for u, v in grid:
            if rng.random() < 0.15:
                continue
            a, b = (u, v) if rng.randrange(2) else (v, u)
            edges.append((a, b, rng.randint(1, 9) if kind == "shortest" else 1))
        graph = nx.DiGraph()
        graph.add_nodes_from(range(8))
        graph.add_weighted_edges_from(edges)
        if kind == "shortest":
            costs = dict(nx.all_pairs_dijkstra_path_length(graph))
            both = dict(nx.all_pairs_dijkstra_path_length(graph.to_undirected()))
            pairs = [(a, b) for a in range(8) for b, cost in costs[a].items()
                        if a != b and cost != both[a][b]
                        and len(nx.shortest_path(graph, a, b, weight="weight")) >= 3]
            if not pairs:
                return None
            start, target = rng.choice(pairs)
        else:
            roots = [a for a in range(8) if 3 <= len(nx.descendants(graph, a)) + 1 <
                        len(nx.node_connected_component(graph.to_undirected(), a))]
            if not roots:
                return None
            start, target = rng.choice(roots), None
        labels = list(_NODES)
        rng.shuffle(labels)
        at = labels.index("A")
        labels[start], labels[at] = labels[at], labels[start]
        if target is not None:
            at = labels.index("H")
            labels[target], labels[at] = labels[at], labels[target]
        return {"kind": kind, "nodes": list(_NODES),
                "edges": sorted([labels[a], labels[b], weight] for a, b, weight in edges),
                "positions": {labels[i]: i for i in range(8)}}
    if kind == "dfa":
        states = [f"q{i}" for i in range(4)]
        return {"kind": kind, "start": "q0", "accepting": sorted(rng.sample(states, rng.randint(1, 3))),
                "trans": {state: {symbol: rng.choice(states) for symbol in "01"} for state in states}}
    labels = list("ABCDEFG")
    rng.shuffle(labels)
    children: dict[str, dict[str, str]] = {}
    free, depths = [(labels[0], "L"), (labels[0], "R")], {labels[0]: 0}
    for node in labels[1:]:
        parent, side = free.pop(rng.randrange(len(free)))
        children.setdefault(parent, {})[side] = node
        depths[node] = depths[parent] + 1
        if depths[node] < 4:
            free.extend([(node, "L"), (node, "R")])
    return {"kind": kind, "root": labels[0], "children": children}


def _facts(world: dict) -> dict[str, str]:
    if world["kind"] in ("shortest", "bfs"):
        facts = {f"edge[{a},{b}]": str(weight) for a, b, weight in world["edges"]}
    elif world["kind"] == "dfa":
        facts = {"start": world["start"], "accepting": ",".join(world["accepting"]),
                **{f"trans[{state},{symbol}]": target for state, trans in world["trans"].items()
                   for symbol, target in trans.items()}}
    else:
        facts = {"root": world["root"], **{f"child[{node},{side}]": child
                 for node, children in world["children"].items() for side, child in children.items()}}
    return dict(sorted(facts.items()))


def _slot_types(slots: dict[str, str]) -> dict[str, str]:
    return {key: "number" if key.startswith("edge[") else "text" if key == "accepting"
            else "relation" for key in slots}


def _validate_world(world: dict) -> None:
    kind = world["kind"]
    if kind in ("shortest", "bfs"):
        if (set(world) != {"kind", "nodes", "edges", "positions"}
                or world["nodes"] != list(_NODES) or set(world["positions"]) != set(_NODES)
                or any(type(v) is not int for v in world["positions"].values())
                or sorted(world["positions"].values()) != list(range(8))):
            raise ValueError("graph: invalid vertex inventory")
        seen = set()
        for a, b, weight in world["edges"]:
            if (a not in _NODES or b not in _NODES or a == b or type(weight) is not int
                    or not 1 <= weight <= 9 or (kind == "bfs" and weight != 1)
                    or frozenset((a, b)) in seen):
                raise ValueError("graph: invalid directed edge")
            i, j = world["positions"][a], world["positions"][b]
            if abs(i % 4 - j % 4) + abs(i // 4 - j // 4) != 1:
                raise ValueError("graph: non-grid edge would make the rendering ambiguous")
            seen.add(frozenset((a, b)))
    elif kind == "dfa":
        states = {f"q{i}" for i in range(4)}
        if (set(world) != {"kind", "start", "accepting", "trans"} or world["start"] != "q0"
                or set(world["trans"]) != states
                or world["accepting"] != sorted(set(world["accepting"]))
                or not 1 <= len(world["accepting"]) <= 3 or not set(world["accepting"]) <= states):
            raise ValueError("graph: invalid DFA state inventory")
        if any(set(trans) != {"0", "1"} or not set(trans.values()) <= states
               for trans in world["trans"].values()):
            raise ValueError("graph: incomplete DFA transition function")
    elif kind == "tree":
        if set(world) != {"kind", "root", "children"}:
            raise ValueError("graph: invalid tree structure")
        tree = nx.DiGraph()
        tree.add_node(world["root"])
        child_labels = []
        for parent, children in world["children"].items():
            if not children or not set(children) <= {"L", "R"}:
                raise ValueError("graph: invalid child roles")
            tree.add_edges_from((parent, child) for child in children.values())
            child_labels.extend(children.values())
        if (set(tree) != set("ABCDEFG") or not nx.is_arborescence(tree)
                or len(child_labels) != 6 or len(set(child_labels)) != 6
                or tree.in_degree(world["root"]) != 0
                or max(nx.single_source_shortest_path_length(tree, world["root"]).values()) > 4):
            raise ValueError("graph: invalid binary tree")
    else:
        raise ValueError("graph: unsupported task")


def _question(kind: str, traversal: str) -> str:
    if kind == "shortest":
        return ("What is the shortest path length from A to H in this directed weighted graph? "
                "Follow arrowheads and sum the weights. Crossings are not junctions. Give an integer.")
    if kind == "bfs":
        return ("Starting at A, list reachable vertices in BFS discovery order, including A. "
                "Follow outgoing arrows only; enqueue unseen neighbors in ascending alphabetical order. "
                "Omit unreachable vertices. Give comma-separated labels. Crossings are not junctions.")
    if kind == "dfa":
        return ("Does this DFA accept the word 0110? Read symbols left to right from the marked start "
                "state; double circles mark accepting states. Respond accept or reject. "
                "In observations, list accepting states in ascending state-label order.")
    convention = {"preorder": "root, left subtree, right subtree",
                  "inorder": "left subtree, root, right subtree",
                  "postorder": "left subtree, right subtree, root"}[traversal]
    return (f"Give the {traversal} traversal ({convention}) of this binary tree. "
            "The top node is the root; L and R label left and right child edges. Give comma-separated labels.")


def _render(world: dict, style: str) -> tuple[bytes, tuple[str, ...]]:
    shift = style == "shift"
    image = Image.new("RGB", (1000, 760), "#fffaf0" if shift else "white")
    draw = ImageDraw.Draw(image)
    ink, width = ("#17365d", 5) if shift else ("black", 3)
    font = ImageFont.load_default_imagefont() if shift else ImageFont.load_default(size=22)
    printed = []
    def label(x, y, value):
        printed.append(value)
        if shift:
            box = font.getbbox(value)
            patch = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]))
            ImageDraw.Draw(patch).text((-box[0], -box[1]), value, font=font, fill=ink)
            patch = patch.resize((patch.width * 2, patch.height * 2), Image.Resampling.NEAREST)
            image.paste(patch, (round(x - patch.width / 2), round(y - patch.height / 2)), patch)
        else:
            draw.text((x, y), value, font=font, fill=ink, anchor="mm")
    def arrow(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        start, end = (a[0] + ux * 30, a[1] + uy * 30), (b[0] - ux * 30, b[1] - uy * 30)
        draw.line((start, end), fill=ink, width=width)
        draw.polygon((end, (end[0] - ux * 15 - uy * 8, end[1] - uy * 15 + ux * 8),
                      (end[0] - ux * 15 + uy * 8, end[1] - uy * 15 - ux * 8)), fill=ink)
    def node(x, y, name, accepting=False):
        draw.ellipse((x - 26, y - 26, x + 26, y + 26), fill=image.getpixel((0, 0)), outline=ink, width=width)
        if accepting:
            draw.ellipse((x - 21, y - 21, x + 21, y + 21), outline=ink, width=2)
        label(x, y, name)
    kind = world["kind"]
    if kind in ("shortest", "bfs"):
        label(500, 45, "Directed weighted graph" if kind == "shortest" else "Directed graph (unit edges)")
        label(500, 88, "Arrowheads determine direction; repeated labels refer to the same vertex.")
        positions = {name: (110 + (3 - at % 4 if shift else at % 4) * 260,
                            240 + (at // 4) * (320 if shift else 300))
                     for name, at in sorted(world["positions"].items())}
        for a, b, weight in world["edges"]:
            start, end = positions[a], positions[b]
            arrow(start, end)
            if kind == "shortest":
                x, y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
                draw.rectangle((x - 19, y - 18, x + 19, y + 18), fill=image.getpixel((0, 0)))
                label(x, y, str(weight))
        for name, (x, y) in positions.items():
            node(x, y, name)
    elif kind == "dfa":
        label(500, 42, "DFA transition diagram in tabular form (alphabet: 0, 1)")
        label(500, 82, "Each row gives the outgoing state for the indicated input symbol.")
        label(500, 122, "Double-circle states accept; the arrow marks the start state.")
        for x, heading in ((210, "State"), (500, "0"), (790, "1")):
            label(x, 205, heading)
        states = sorted(world["trans"], reverse=shift)
        for row, state in enumerate(states):
            y = 300 + row * 105
            if state == world["start"]:
                arrow((70, y), (210, y))
            node(210, y, state, state in world["accepting"])
            label(500, y, world["trans"][state]["0"])
            label(790, y, world["trans"][state]["1"])
            draw.line((300, y + 48, 900, y + 48), fill=ink, width=1)
    else:
        label(500, 42, "Binary tree: L = left child, R = right child")
        order = _tree_order(world, "inorder", iterative=True)
        positions = {}
        def locate(name, depth):
            positions[name] = (100 + order.index(name) * (125 if shift else 130), 160 + depth * 125)
            for child in world["children"].get(name, {}).values():
                locate(child, depth + 1)
        locate(world["root"], 0)
        for parent, children in sorted(world["children"].items()):
            for side, child in sorted(children.items()):
                a, b = positions[parent], positions[child]
                arrow(a, b)
                label((a[0] + b[0]) / 2 + (14 if side == "R" else -14), (a[1] + b[1]) / 2, side)
        for name, (x, y) in sorted(positions.items()):
            node(x, y, name)
    return png_bytes(image, dpi=100), tuple(printed)


def generate(index: int, seed: int, split: Split, mode: Mode, style: str = "default") -> GeneratedProblem:
    if type(index) is not int or index < 0 or type(seed) is not int or seed < 0:
        raise ValueError("graph: nonnegative integer index and seed required")
    matches_split("0" * 16, split, style)
    if mode not in ("obs", "plain"):
        raise ValueError("graph: invalid mode")
    kind = _KINDS[index % 4]
    traversal = ("preorder", "inorder", "postorder")[(index // 4) % 3]
    rng = random.Random(seed * 1_000_003 + index)
    for _ in range(1000):
        world = _sample(kind, rng)
        if world is None:
            continue
        _validate_world(world)
        if _shortcut_reason(world, traversal):
            continue
        # Layout is a rendering choice, never a split key.
        structure = {k: v for k, v in world.items() if k != "positions"}
        if not matches_split(structure_hash(structure), split, style):
            continue
        gold, independent = _answers(world, traversal)
        if gold != independent:
            raise ValueError("graph: independent solver disagreement")
        raw, printed = _render(world, style)
        slots = _facts(world)
        # Grid placement is reconstructible from node positions, held privately for self-check only.
        accept = {"ordered": True, "element_type": "short_text"} if kind in ("bfs", "tree") else {}
        accept["graph_world"] = world
        p = build(source=GENERATOR_VERSION, family=FAMILY, index=index, split=split, mode=mode,
                  question=_question(kind, traversal), choices=(),
                  answer_type="int" if kind == "shortest" else "short_text" if kind == "dfa" else "list",
                  gold=gold, accept=accept, slots=slots,
                  slot_types=_slot_types(slots),
                  required_slots=tuple(slots), oracle_fact_lines=tuple(f"{k} = {v}" for k, v in slots.items()),
                  printed_text=printed, structure=structure,
                  template_id=f"graph-{kind}" + (f"-{traversal}" if kind == "tree" else ""),
                  image_bytes=(raw,), generator_version=GENERATOR_VERSION, seed=seed, style=style)
        return p
    raise RuntimeError("graph: exhausted 1000 candidate structures")


def self_check(p: GeneratedProblem) -> None:
    try:
        for record in (p.actor, p.verifier, p.lineage):
            to_json(record)
        world = p.verifier.accept["graph_world"]
        _validate_world(world)
        template = p.lineage.template_id
        traversal = template.rsplit("-", 1)[-1] if world["kind"] == "tree" else "postorder"
        style = "shift" if p.lineage.split == "style_sealed" else "default"
        structure = {k: v for k, v in world.items() if k != "positions"}
        if (structure_hash(structure) != p.lineage.structure_hash
                or not matches_split(p.lineage.structure_hash, p.lineage.split, style)
                or _shortcut_reason(world, traversal)):
            raise ValueError("structure, split, or shortcut mismatch")
        gold, independent = _answers(world, traversal)
        raw, printed = _render(world, style)
        accept_keys = {"graph_world", "ordered", "element_type"} if world["kind"] in ("bfs", "tree") else {"graph_world"}
        if (set(p.verifier.accept) != accept_keys or (world["kind"] in ("bfs", "tree") and (
                p.verifier.accept["ordered"] is not True
                or p.verifier.accept["element_type"] != "short_text"))):
            raise ValueError("unapproved answer comparison policy")
        expected_type = "int" if world["kind"] == "shortest" else "short_text" if world["kind"] == "dfa" else "list"
        if (p.actor.source != GENERATOR_VERSION or p.actor.family != FAMILY
                or p.lineage.source != GENERATOR_VERSION or p.lineage.generator_version != GENERATOR_VERSION
                or p.lineage.source_revision != GENERATOR_VERSION
                or p.lineage.image_provenance != f"rendered:{GENERATOR_VERSION}:{style}"
                or p.actor.problem_id != p.verifier.problem_id or p.actor.problem_id != p.lineage.problem_id
                or p.lineage.benchmark_derived or p.lineage.model_touched
                or p.lineage.label_provenance != "PROGRAM"
                or p.actor.answer_format != expected_type or p.verifier.answer_type != expected_type
                or p.actor.choices or p.verifier.choices
                or template != f"graph-{world['kind']}" + (f"-{traversal}" if world["kind"] == "tree" else "")
                or gold != independent or gold != p.verifier.gold or _facts(world) != p.verifier.slots
                or _slot_types(p.verifier.slots) != p.verifier.slot_types
                or tuple(p.verifier.slots) != p.verifier.required_slots
                or p.verifier.oracle_fact_lines != tuple(f"{k} = {v}" for k, v in p.verifier.slots.items())
                or printed != p.verifier.printed_text or p.image_bytes != (raw,)
                or p.actor.question != _question(world["kind"], traversal)):
            raise ValueError("answer, facts, or rendering mismatch")
        if p.actor.image_sha256 != (hashlib.sha256(raw).hexdigest(),):
            raise ValueError("image hash mismatch")
    except (AttributeError, IndexError, KeyError, TypeError, nx.NetworkXException,
            RecursionError, ValueError) as exc:
        raise ValueError("graph: self-check failed") from exc
