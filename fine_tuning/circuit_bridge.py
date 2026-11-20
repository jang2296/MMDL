"""Quarantined independent DC circuit A0-A3 prototype, not full professional G7.

Accepts procedural counts/seeds only, never benchmark or external data inputs.
No model/GPU/cloud operation or automatic training approval occurs here.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
from fractions import Fraction
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

from . import nvw
from .ablation import ARMS, FLAGS, VIEWS
from .graph_bridge import font_sha256

VERSION = "independent-dc-circuit-bridge-v2"
# v2 repairs choice construction only; preserve the original worlds and glyph mapping.
MAPPING_VERSION = "independent-dc-circuit-bridge-v1"
SOURCE_ID = VERSION
FIXED_MAPPING = {"RESISTOR": 0, "DC_SOURCE": 1, "SWITCH": 2}
SCOPE = "BOUNDED_SYNTHETIC_SERIES_PARALLEL_DC_CIRCUIT_BRIDGE"


def validate_world(world):
    if (set(world) != {"voltage_V", "resistance_ohm", "switch_closed"}
            or type(world["voltage_V"]) is not int or not 1 <= world["voltage_V"] <= 48
            or not isinstance(world["resistance_ohm"], list) or len(world["resistance_ohm"]) != 3
            or any(type(r) is not int or not 1 <= r <= 48 for r in world["resistance_ohm"])
            or type(world["switch_closed"]) is not bool):
        raise ValueError("Unsupported bounded DC component/value/unit contract")


def solve_nodal(world):
    """Exact KCL elimination; an ideal closed switch identifies nodes N2 and N3."""
    validate_world(world)
    voltage = Fraction(world["voltage_V"])
    known = {0: Fraction(0), 1: voltage}
    unknown = [2] if world["switch_closed"] else [2, 3]
    conductance = [[Fraction(0) for _ in unknown] for _ in unknown]
    rhs = [Fraction(0) for _ in unknown]
    resistors = [(1, 2), (2, 0), (3, 0)]
    for (a, b), resistance in zip(resistors, world["resistance_ohm"], strict=True):
        if world["switch_closed"]:
            a, b = (2 if n == 3 else n for n in (a, b))
        for node, other in ((a, b), (b, a)):
            if node in known:
                continue
            i = unknown.index(node)
            g = Fraction(1, resistance)
            conductance[i][i] += g
            if other in known:
                rhs[i] += g * known[other]
            else:
                conductance[i][unknown.index(other)] -= g
    for i in range(len(unknown)):
        pivot = conductance[i][i]
        if not pivot:
            raise ValueError("Singular circuit")
        conductance[i] = [v/pivot for v in conductance[i]]
        rhs[i] /= pivot
        for j in range(len(unknown)):
            if i == j:
                continue
            factor = conductance[j][i]
            conductance[j] = [v-factor*w for v,w in zip(conductance[j], conductance[i], strict=True)]
            rhs[j] -= factor*rhs[i]
    nodes = {**known, **dict(zip(unknown, rhs, strict=True))}
    if world["switch_closed"]:
        nodes[3] = nodes[2]
    currents = [(nodes[a]-nodes[b])/r for (a,b),r in zip(resistors, world["resistance_ohm"], strict=True)]
    if currents[0] != currents[1]+currents[2]:
        raise ValueError("KCL verification failed")
    return {"source_current_A": str(currents[0]),
            "node_voltages_V": {f"N{i}": str(nodes[i]) for i in range(4)},
            "resistor_currents_A": {f"C{i}": str(current) for i,current in enumerate(currents)}}


def verify_series_parallel(world):
    """Separate circuit reduction, without constructing or solving nodal equations."""
    validate_world(world)
    r0, r1, r2 = map(Fraction, world["resistance_ohm"])
    parallel = r1*r2/(r1+r2) if world["switch_closed"] else r1
    total = Fraction(world["voltage_V"])/(r0+parallel)
    branch = total*parallel
    second = branch/r2 if world["switch_closed"] else Fraction(0)
    return {"source_current_A": str(total), "node_voltages_V": {
        "N0": "0", "N1": str(world["voltage_V"]), "N2": str(branch),
        "N3": str(branch if world["switch_closed"] else Fraction(0))},
        "resistor_currents_A": {"C0": str(total), "C1": str(branch/r1), "C2": str(second)}}


def canonical_structure(world):
    validate_world(world)
    components = [{"id": f"C{i}", "type": "RESISTOR", "pins": [f"N{a}", f"N{b}"],
                   "value": resistance, "unit": "ohm"}
                  for i,((a,b),resistance) in enumerate(zip(((1,2),(2,0),(3,0)), world["resistance_ohm"], strict=True))]
    components += [{"id": "C3", "type": "DC_SOURCE", "pins": ["N1", "N0"],
                    "positive_pin": "N1", "value": world["voltage_V"], "unit": "V"},
                   {"id": "C4", "type": "SWITCH", "pins": ["N2", "N3"],
                    "state": "CLOSED" if world["switch_closed"] else "OPEN"}]
    return {"nodes": [f"N{i}" for i in range(4)], "reference_node": "N0", "components": components}


def root_episode(index, seed):
    rng = random.Random(nvw.digest([MAPPING_VERSION, seed, index, "world"]))
    world = {"voltage_V": rng.choice([6, 9, 12, 18, 24]),
             "resistance_ohm": rng.sample([2, 3, 4, 6, 8, 10, 12, 16, 20, 24], 3),
             "switch_closed": bool(rng.randrange(2))}
    solution = solve_nodal(world)
    if solution != verify_series_parallel(world):
        raise ValueError("Independent solver disagreement")
    divisor = math.gcd(*world["resistance_ohm"])
    family = "dc-circuit-"+nvw.digest([world["switch_closed"], [r//divisor for r in world["resistance_ohm"]]])
    root_id = nvw.digest([family, world])
    answer = Fraction(solution["source_current_A"])
    r0, r1, r2 = map(Fraction, world["resistance_ohm"])
    voltage = Fraction(world["voltage_V"])
    parallel = r1*r2/(r1+r2) if world["switch_closed"] else r1
    low = sorted(v for v in {voltage/(r0+r1+r2),voltage/(2*r0+r1+r2),
                  voltage/(2*(r0+r1+r2)),*(Fraction(v) for v in solution["resistor_currents_A"].values())} if 0 < v < answer)
    high = sorted({voltage/r0,voltage*(1/r0+1/r1+1/r2),
                   voltage/(r0/2+parallel),voltage/(r0/3+parallel)} - {answer})
    if len(low) < 3 or len(high) < 3 or any(v >= answer for v in low) or any(v <= answer for v in high):
        raise ValueError("Cannot build independent distractors on both sides of the current")
    option_rng = random.Random(nvw.digest([VERSION, seed, root_id, "options"]))
    rank = option_rng.randrange(4)
    options = [str(v) for v in [answer,*option_rng.sample(low,rank),*option_rng.sample(high,3-rank)]]
    option_rng.shuffle(options)
    return {"id": f"{VERSION}-{seed}-{index}", "root_id": root_id, "family_id": family,
            "split": nvw.family_split(family), "world": world, "solution": solution,
            "options": options, "answer_letter": chr(65+options.index(str(answer))),
            "source_id": SOURCE_ID, "source_kind": "INDEPENDENT_SYNTHETIC", "parents": [],
            "lineage": [SOURCE_ID], "all_parents_known": True, **FLAGS}


def render(episode, representation):
    if representation not in ("novel", "standard", "standard_relayout"):
        raise ValueError("Unknown circuit representation")
    mapping = episode["component_glyph_mapping"]
    if set(mapping) != set(FIXED_MAPPING) or sorted(mapping.values()) != [0, 1, 2]:
        raise ValueError("Component legend is not a bijection")
    structure = canonical_structure(episode["world"])
    positions = {"N0": (600, 800), "N1": (150, 420), "N2": (600, 420), "N3": (1000, 420)}
    if representation == "standard_relayout":
        positions = {n: (nvw.WIDTH-x, y) for n,(x,y) in positions.items()}
    image = Image.new("RGB", (nvw.WIDTH, nvw.HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    printed, texts, symbols = [], [], []
    def label(xy, value, small=False):
        bbox = nvw.text(draw, xy, value, small=small)
        printed.append(value)
        texts.append({"value": value, "bbox": bbox})
    label((30, 20), "DC resistive circuit: ideal source, resistors and switch")
    draw.rectangle((20, 65, 1180, 320), outline="black", width=2)
    label((35, 80), "All wires and a CLOSED switch have zero resistance. An OPEN switch is disconnected.", True)
    label((35, 114), "Source polarity is explicit: + at its upper pin, - at its lower pin. N0 is the reference.", True)
    for i,kind in enumerate(FIXED_MAPPING):
        x = 70+i*365
        if representation == "novel":
            draw.rectangle((x-30,150,x+30,210),outline="black",width=2)
            nvw.glyph(draw, x, 180, mapping[kind], radius=20)
        label((x+36, 169), kind if representation == "novel" else "Standard "+kind, True)
    label((35, 238), "Values use V and ohm. Junctions are filled dots; no wire crossings occur.", True)
    label((35, 279), "Use the circuit as drawn; C0-C4 are component IDs, not their types.", True)
    draw.rectangle((20, 350, 1180, 860), outline="black", width=2)
    for component in structure["components"]:
        a, b = (positions[n] for n in component["pins"])
        if component["id"] in {"C1", "C2", "C3"}:
            b = (a[0], positions["N0"][1])
            draw.line((b, positions["N0"]), fill="black", width=3)
        cx, cy = ((a[i]+b[i])//2 for i in (0,1))
        ux, uy = (1 if b[0] > a[0] else -1, 0) if a[1] == b[1] else (0, 1)
        p, q = (cx-ux*35, cy-uy*35), (cx+ux*35, cy+uy*35)
        draw.line((a,p), fill="black", width=3)
        draw.line((q,b), fill="black", width=3)
        bbox = [cx-30,cy-35,cx+30,cy+35]
        if representation == "novel":
            draw.rectangle((cx-30,cy-30,cx+30,cy+30),outline="black",width=2)
            nvw.glyph(draw,cx,cy,mapping[component["type"]],radius=20)
            draw.line((p,(cx-ux*30,cy-uy*30)),fill="black",width=3)
            draw.line(((cx+ux*30,cy+uy*30),q),fill="black",width=3)
        elif component["type"] == "RESISTOR":
            body = [cx-30,cy-12,cx+30,cy+12] if uy == 0 else [cx-12,cy-30,cx+12,cy+30]
            draw.rectangle(body,outline="black",width=3)
            draw.line((p,(cx-ux*30,cy-uy*30)),fill="black",width=3)
            draw.line(((cx+ux*30,cy+uy*30),q),fill="black",width=3)
        elif component["type"] == "DC_SOURCE":
            draw.ellipse((cx-30,cy-30,cx+30,cy+30),outline="black",width=3)
            draw.line((p,(cx,cy-30)),fill="black",width=3)
            draw.line(((cx,cy+30),q),fill="black",width=3)
        else:
            endpoint = q if episode["world"]["switch_closed"] else (q[0],q[1]-28)
            draw.line((p,endpoint),fill="black",width=3)
            for x,y in (p,q):
                draw.ellipse((x-4,y-4,x+4,y+4),fill="white",outline="black",width=2)
        if component["type"] == "DC_SOURCE":
            label((cx-50,cy-30),"+",True)
            label((cx-50,cy+8),"-",True)
        value = component.get("state", str(component.get("value", ""))+" "+component.get("unit", ""))
        label((cx-55,cy+45) if uy == 0 else (cx+35,cy-8), component["id"]+" "+value, True)
        symbols.append({"id":component["id"], "bbox":bbox, "wire_pins":[list(a),list(b)],
                        "glyph_code":mapping[component["type"]] if representation == "novel" else None})
    for node,(x,y) in positions.items():
        draw.ellipse((x-5,y-5,x+5,y+5),fill="black")
        label((x+10,y-27),node,True)
    label((35,900),"What current does the source deliver, in A? Choose one option.")
    label((35,946),"    ".join(f"{chr(65+i)}. {value} A" for i,value in enumerate(episode["options"])),True)
    if any(not (20 <= v["bbox"][0] < v["bbox"][2] <= 1180 and 350 <= v["bbox"][1] < v["bbox"][3] <= 860) for v in symbols):
        raise ValueError("Component outside circuit panel")
    for i,text in enumerate(texts):
        x0,y0,x1,y1 = text["bbox"]
        for other in texts[i+1:]:
            a,b,c,d = other["bbox"]
            if max(x0,a) < min(x1,c) and max(y0,b) < min(y1,d):
                raise ValueError("Printed labels overlap")
    return image, {"representation":representation, "image_size":[nvw.WIDTH,nvw.HEIGHT],
                   "canonical_structure":structure, "node_positions":{n:list(p) for n,p in positions.items()},
                   "symbols":symbols, "printed_text":printed, "text_boxes":texts,
                   "legend":dict(mapping) if representation == "novel" else None}


def record(episode, paths, view):
    if view not in VIEWS:
        raise ValueError("Unknown circuit view")
    selected = paths if view == "grounded" else paths[:1]
    instruction = {"qa":"Choose the source-current option. Give only its letter.",
        "structure":"Transcribe nodes, reference, component types, pins, values, units, source polarity and switch state as JSON; do not solve currents.",
        "grounded":"The images represent the same circuit. Give exact node voltages, resistor currents, source current and option letter as JSON."}[view]
    text = "<image>\n"*len(selected)+instruction
    target = (episode["answer_letter"] if view == "qa" else json.dumps(
        canonical_structure(episode["world"]) if view == "structure" else
        {**episode["solution"],"option":episode["answer_letter"]},sort_keys=True))
    return {"id":episode["id"]+"-"+episode["condition"]+"-"+view,
            "root_id":episode["root_id"], "family_id":episode["family_id"], "split":episode["split"],
            "condition":episode["condition"],"view":view,"source_id":SOURCE_ID,"source_kind":"INDEPENDENT_SYNTHETIC",
            "parents":[],"lineage":[SOURCE_ID],"all_parents_known":True,
            "quality_status":"DERIVED_VERIFIED","quality_scope":"canonical_and_exact_solver_only; render_and_processor_review_pending",
            "answer_identifiable":True,"supervised_target_identifiable":True,
            "user_text":text,"target_text":target,"image_paths":selected,"image":selected,
            "conversations":[{"from":"human","value":text},{"from":"gpt","value":target}], **FLAGS}


def bundle(count, seed, start_index):
    if (type(count) is not int or not 1 <= count <= 100 or type(seed) is not int or seed < 0
            or type(start_index) is not int or start_index < 1_000_000):
        raise ValueError("Prototype requires count1..100, nonnegative seed, index>=1000000")
    files = {"sources/"+name:Path(__file__).with_name(name).read_bytes() for name in
             ("circuit_bridge.py","circuit_clean_spec_v1.json","nvw.py","ablation.py","graph_bridge.py")}
    source_hashes = {n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
    episodes,pairings,reviews,seen = [],[],[],set()
    records = {arm:[] for arm in ARMS}
    index = start_index
    while len(pairings) < count:
        if index-start_index >= count*100:
            raise ValueError("Bounded independent root search exhausted")
        root = root_episode(index,seed)
        index += 1
        if root["root_id"] in seen:
            continue
        seen.add(root["root_id"])
        paired = {}
        for arm,(randomized,enabled) in ARMS.items():
            episode = copy.deepcopy(root)
            codes = list(FIXED_MAPPING.values())
            if randomized:
                random.Random(nvw.digest([MAPPING_VERSION,seed,root["root_id"],"component_mapping"])).shuffle(codes)
            episode.update(condition=arm,bridge_enabled=enabled,component_glyph_mapping=dict(zip(FIXED_MAPPING,codes,strict=True)))
            kinds = ["novel","standard","standard_relayout"] if enabled else ["novel"]*3
            paths,scenes = [],[]
            for slot,kind in enumerate(kinds):
                image,scene = render(episode,kind)
                path = f"images/{root['id']}-{arm}-{slot}.png"
                stream = io.BytesIO()
                image.save(stream,format="PNG")
                files[path] = stream.getvalue()
                paths.append(path)
                scenes.append(scene)
            episode.update(image_paths=paths,representations=kinds,representation_scenes=scenes,
                image_hashes={p:hashlib.sha256(files[p]).hexdigest() for p in paths},source_files_sha256=source_hashes)
            episodes.append(episode)
            rows = [record(episode,paths,view) for view in VIEWS]
            records[arm].extend(rows)
            paired[arm] = {"record_ids":[r["id"] for r in rows],"images":paths,"component_glyph_mapping":episode["component_glyph_mapping"]}
            reviews.append({"root_id":root["root_id"],"condition":arm,"images":paths,
                            "status":"PENDING_RENDER_RIGHTS_CONTAMINATION_AND_ACTUAL_PROCESSOR_REVIEW",**FLAGS})
        pairings.append({"root_id":root["root_id"],"family_id":root["family_id"],"split":root["split"],
                         "semantic_sha256":nvw.digest([root["world"],root["solution"],root["options"]]),"conditions":paired,**FLAGS})
    for name,rows in (("canonical_private.jsonl",episodes),("pairing.jsonl",pairings),("review_queue.jsonl",reviews),
                      *((arm+".jsonl",rows) for arm,rows in records.items())):
        files[name] = "".join(json.dumps(r,sort_keys=True)+"\n" for r in rows).encode()
    # ponytail: one series/parallel DC topology; additional topologies need a new verified solver/render contract.
    manifest = {"version":VERSION,"mapping_version":MAPPING_VERSION,"scope":SCOPE,"source_id":SOURCE_ID,"count":count,"seed":seed,
        "start_index":start_index,"next_index":index,"root_count":count,"records_per_arm":3*count,
        "image_files_per_arm":3*count,"image_references_per_arm":5*count,
        "source_files_sha256":source_hashes,"family_splits":{p["family_id"]:p["split"] for p in pairings},
        "source_ancestry":{SOURCE_ID:{"parents":[],"rights_approved":False,"benchmark_derived":False}},
        "render_assets":{"pillow_version":PILLOW_VERSION,"font_module_sha256":hashlib.sha256(Path(ImageFont.__file__).read_bytes()).hexdigest(),
            "font_ttf_sha256":font_sha256(nvw.FONT),"small_font_ttf_sha256":font_sha256(nvw.SMALL)},
        "solver_policy":"Exact rational nodal KCL vs independent series/parallel reduction",
        "choice_policy":"Hash-seeded varied correct sorted rank; exact rational alternate circuit reductions, then shuffled labels",
        "gate":"UNAPPROVED_NOT_FOR_TRAINING_OR_PUBLICATION",
        "limitations":["One bounded ideal DC resistive topology only; no AC/nonlinear/transient circuits",
            "Synthetic representation comparison, not full professional G7 or actual source transfer",
            "Values/units can disclose component roles; mapping-only causality or shortcut immunity not claimed",
            "Program geometry checks are not independent render or pinned processor approval",
            "Matched records/image references are not matched token exposure or GPU time"],
        "files":{n:{"sha256":hashlib.sha256(v).hexdigest(),"bytes":len(v)} for n,v in files.items()},**FLAGS}
    files["manifest.json"] = (json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    return files


def generate(output, count=4, seed=73041, start_index=1_000_000):
    if output.exists() or output.is_symlink():
        raise FileExistsError("Refuse to overwrite existing output")
    files = bundle(count,seed,start_index)
    output.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        path = output/name
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
    return json.loads(files["manifest.json"])


def verify(output):
    if output.is_symlink() or any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("Symlinks are not allowed")
    manifest = json.loads((output/"manifest.json").read_bytes())
    expected = bundle(manifest["count"],manifest["seed"],manifest["start_index"])
    if {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()} != set(expected):
        raise ValueError("Artifact inventory mismatch")
    if any((output/name).read_bytes() != data for name,data in expected.items()):
        raise ValueError("Artifact differs from deterministic generation")
    return {"status":"CPU_REPRODUCED_NOT_APPROVED","root_count":manifest["root_count"],"scope":SCOPE,**FLAGS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--count",type=int,default=4)
    parser.add_argument("--seed",type=int,default=73041)
    parser.add_argument("--start-index",type=int,default=1_000_000)
    parser.add_argument("--verify",action="store_true")
    args = parser.parse_args()
    result = verify(args.output) if args.verify else generate(args.output,args.count,args.seed,args.start_index)
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
