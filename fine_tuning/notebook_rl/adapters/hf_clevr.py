"""Original CLEVR train spatial QA, scenes and program labels; no teacher traces.

The pinned loader points to the original release; acquired train files are hash-journaled. One spatial question
per image is chosen by candidate_order(seed3407), independently of its answer.
Native functional programs must reproduce the original gold; failed labels are
omitted, never repaired. Programs and scene annotations never enter actor text.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from ..schema import AnswerType
from .hf_common import Candidate, SourceSpec, candidate_order

SPECS = {"hf-clevr": SourceSpec(
    "hf-clevr", "HuggingFaceM4/clevr", "d2a0cc5509d56204238d34969b0b3657f8d3bf32",
    "CC-BY-4.0", "PROGRAM", "Original CLEVR v1.0 rendered scenes and functional-program answers (Johnson et al. 2017)",
    "spatial_scene",
)}
_ATTRIBUTES = ("color", "shape", "size", "material")
_IMAGE = re.compile(r"CLEVR_train_\d{6}\.png\Z")


def program_answer(scene: dict, program: list[dict]) -> str:
    """Execute the published CLEVR operations on their native ground-truth scene."""
    objects = scene["objects"]
    outputs: list = []
    for node in program:
        indices, function, values = node["inputs"], node["function"], node["value_inputs"]
        if any(type(i) is not int or not 0 <= i < len(outputs) for i in indices):
            raise ValueError("CLEVR: invalid program input index")
        args = [outputs[i] for i in indices]
        if function == "scene":
            result: object = list(range(len(objects)))
        elif function == "unique":
            if len(args[0]) != 1:
                raise ValueError("CLEVR: nonunique object reference")
            result = args[0][0]
        elif function == "relate":
            result = list(scene["relationships"][values[0]][args[0]])
        elif function == "count":
            result = len(args[0])
        elif function == "exist":
            result = bool(args[0])
        elif function in ("union", "intersect"):
            left, right = set(args[0]), set(args[1])
            result = sorted(left | right if function == "union" else left & right)
        elif function in ("equal_integer", "less_than", "greater_than"):
            result = (args[0] == args[1] if function == "equal_integer" else
                      args[0] < args[1] if function == "less_than" else args[0] > args[1])
        else:
            operation, _, attribute = function.partition("_")
            if attribute not in _ATTRIBUTES:
                raise ValueError(f"CLEVR: unsupported program function {function}")
            if operation == "filter":
                result = [i for i in args[0] if objects[i][attribute] == values[0]]
            elif operation == "query":
                result = objects[args[0]][attribute]
            elif operation == "same":
                result = [i for i, obj in enumerate(objects)
                          if i != args[0] and obj[attribute] == objects[args[0]][attribute]]
            elif operation == "equal":
                result = args[0] == args[1]
            else:
                raise ValueError(f"CLEVR: unsupported program function {function}")
        outputs.append(result)
    if not outputs or type(outputs[-1]) not in (str, int, bool):
        raise ValueError("CLEVR: program has no scalar answer")
    answer = outputs[-1]
    return ("yes" if answer else "no") if type(answer) is bool else str(answer)


def iter_clevr(repo_dir: Path) -> Iterator[Candidate]:
    root = Path(repo_dir)
    questions = json.loads((root / "questions/CLEVR_train_questions.json").read_text())["questions"]
    scenes = {row["image_index"]: row for row in
              json.loads((root / "scenes/CLEVR_train_scenes.json").read_text())["scenes"] if row["split"] == "train"}
    best: dict[str, tuple[str, Candidate]] = {}
    images: dict[str, bytes] = {}
    for question in questions:
        filename, program = question["image_filename"], question["program"]
        if (question["split"] != "train" or _IMAGE.fullmatch(filename) is None or len(program) < 8
                or not any(node["function"] == "relate" for node in program)):
            continue
        scene = scenes.get(question["image_index"])
        if scene is None or scene["image_filename"] != filename:
            raise ValueError("CLEVR: question/image/scene binding mismatch")
        try:
            answer = program_answer(scene, program)
        except ValueError:
            continue
        if answer != question["answer"] or not question["question"].strip():
            continue
        if filename not in images:
            images[filename] = (root / "images/train" / filename).read_bytes()
        choices: tuple[str, ...] = ()
        kind: AnswerType = "short_text"
        gold = answer
        if answer in ("yes", "no"):
            choices, kind, gold = ("Yes", "No"), "mcq", "A" if answer == "yes" else "B"
        elif re.fullmatch(r"\d+", answer):
            kind = "int"
        candidate = Candidate(
            "hf-clevr", f"train:{question['question_index']}", "Math", question["question"].strip(),
            choices, kind, gold, images=(images[filename],), family_key=filename,
        )
        key = candidate_order(candidate, 3407)
        if filename not in best or key < best[filename][0]:
            best[filename] = key, candidate
    for filename in sorted(best):
        yield best[filename][1]


ITERATORS = {"hf-clevr": iter_clevr}
