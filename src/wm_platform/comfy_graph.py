"""Transformations on ComfyUI API-format prompts (dict of node_id -> {class_type, inputs})."""

from __future__ import annotations

import copy

LOADER_NODE_TYPES = frozenset({"VHS_LoadVideoPath", "VHS_LoadVideo"})
ROI_ENTRY_NODE_TYPES = frozenset({"ImageCrop", "CropMask"})
ROI_EXIT_NODE_TYPE = "ImageCompositeMasked"


def find_node_inputs(prompt: dict[str, dict[str, object]], class_type: str) -> dict[str, object] | None:
    for node in prompt.values():
        if not isinstance(node, dict) or node.get("class_type") != class_type:
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            return inputs
    return None


def replace_placeholders(payload: object, replacements: dict[str, object]) -> object:
    if isinstance(payload, str):
        return replacements.get(payload, payload)
    if isinstance(payload, list):
        return [replace_placeholders(item, replacements) for item in payload]
    if isinstance(payload, dict):
        return {key: replace_placeholders(value, replacements) for key, value in payload.items()}
    return payload


def set_frame_window(
    prompt: dict[str, dict[str, object]],
    start: int,
    count: int,
    *,
    static_mask_video: str | None = None,
) -> dict[str, dict[str, object]]:
    """Point every video loader (source and mask) at the same frame window.

    When `static_mask_video` is provided, loaders reading that static mask file keep
    `skip_first_frames = 0` so a single chunk-sized mask clip serves the whole video.
    """
    for node in prompt.values():
        if isinstance(node, dict) and node.get("class_type") in LOADER_NODE_TYPES:
            inputs = node.setdefault("inputs", {})
            if isinstance(inputs, dict):
                is_static_mask = (
                    static_mask_video is not None and str(inputs.get("video", "")) == static_mask_video
                )
                inputs["skip_first_frames"] = 0 if is_static_mask else int(start)
                inputs["frame_load_cap"] = int(count)
    return prompt


def _links(inputs: object) -> list[str]:
    if not isinstance(inputs, dict):
        return []
    return [
        str(value[0])
        for value in inputs.values()
        if isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)
    ]


def _roi_subgraph(prompt: dict[str, dict[str, object]]) -> tuple[list[str], str] | None:
    """Nodes between the crop entry nodes and the composite exit node (inclusive)."""
    exits = [key for key, node in prompt.items() if node.get("class_type") == ROI_EXIT_NODE_TYPE]
    if len(exits) != 1:
        return None
    exit_id = exits[0]
    ancestors: set[str] = set()
    stack = [exit_id]
    while stack:
        current = stack.pop()
        if current in ancestors or current not in prompt:
            continue
        ancestors.add(current)
        stack.extend(_links(prompt[current].get("inputs")))
    entries = {key for key in ancestors if prompt[key].get("class_type") in ROI_ENTRY_NODE_TYPES}
    if not entries:
        return None
    downstream: set[str] = set(entries)
    changed = True
    while changed:
        changed = False
        for key in ancestors - downstream:
            if set(_links(prompt[key].get("inputs"))) & downstream:
                downstream.add(key)
                changed = True
    return sorted(downstream), exit_id


def expand_roi_chain(
    prompt: dict[str, dict[str, object]],
    per_roi_replacements: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    """Repeat the crop -> inpaint -> composite section once per ROI, chained in series.

    The template holds one ROI section whose crop reads the source frames and whose
    composite pastes onto them. Section i reads and pastes onto the output of section
    i-1; whatever consumed the template composite consumes the last one. Each section
    gets its own ROI placeholder values.
    """
    if not per_roi_replacements:
        raise ValueError("at least one ROI is required")
    found = _roi_subgraph(prompt)
    if found is None:
        raise ValueError("workflow has no crop -> composite ROI section")
    section, exit_id = found
    section_set = set(section)
    template = {key: copy.deepcopy(prompt[key]) for key in section}
    result = {key: value for key, value in prompt.items() if key not in section_set}
    consumers = [key for key in result if exit_id in _links(result[key].get("inputs"))]

    previous_exit: str | None = None
    for index, replacements in enumerate(per_roi_replacements):
        suffix = "" if index == 0 else f"_roi{index}"
        id_map = {key: f"{key}{suffix}" for key in section}
        for key, node in template.items():
            clone = replace_placeholders(copy.deepcopy(node), replacements)
            assert isinstance(clone, dict)
            inputs = clone.get("inputs", {})
            for name, value in list(inputs.items()):
                if not (isinstance(value, list) and len(value) == 2 and isinstance(value[1], int)):
                    continue
                source = str(value[0])
                if source in id_map:
                    inputs[name] = [id_map[source], value[1]]
                elif previous_exit is not None and _is_frame_source(clone, name):
                    inputs[name] = [previous_exit, 0]
            result[id_map[key]] = clone
        previous_exit = id_map[exit_id]

    assert previous_exit is not None
    for key in consumers:
        inputs = result[key].get("inputs", {})
        for name, value in list(inputs.items()):
            if isinstance(value, list) and len(value) == 2 and str(value[0]) == exit_id:
                inputs[name] = [previous_exit, value[1]]
    return result


def _is_frame_source(node: dict[str, object], input_name: str) -> bool:
    """Inputs that carry the full frames being edited (not masks or models)."""
    class_type = node.get("class_type")
    return (class_type == "ImageCrop" and input_name == "image") or (
        class_type == ROI_EXIT_NODE_TYPE and input_name == "destination"
    )
