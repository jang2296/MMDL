"""Placement budgets exclude runtime headroom; no disk or precision fallback."""


def placement_kwargs(hw, free_gpu_bytes, available_ram_bytes):
    gib = 1024**3
    if free_gpu_bytes < hw["min_free_gpu_gib"] * gib:
        raise RuntimeError("Insufficient free GPU memory for selected profile")
    if hw["placement"] == "gpu_only":
        return {"device_map": {"": hw["gpu_index"]}}
    gpu = min(int(hw["gpu_weight_cap_gib"] * gib),
              int(free_gpu_bytes - hw["gpu_reserve_gib"] * gib))
    cpu = min(int(hw["cpu_weight_cap_gib"] * gib),
              int(available_ram_bytes * hw["cpu_available_fraction"]))
    if min(gpu, cpu) < gib:
        raise RuntimeError("Insufficient GPU/RAM weight budget")
    return {"device_map": "auto", "max_memory": {hw["gpu_index"]: gpu, "cpu": cpu}}
