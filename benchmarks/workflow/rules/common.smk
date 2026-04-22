import json
from pathlib import Path


def _summary_payload(task_name):
    task_cfg = config["tasks"].get(task_name, {})
    return {
        "task": task_name,
        "synthetic": task_cfg.get("synthetic", []),
        "literature": task_cfg.get("literature", []),
        "methods": config.get("methods", []),
        "runtime_target_minutes": config.get("runtime", {}).get("target_minutes", 45),
    }


rule emit_summary:
    output:
        "results/{task}/summary.json"
    run:
        task_name = wildcards.task
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        with open(output[0], "w", encoding="ascii") as fh:
            json.dump(_summary_payload(task_name), fh, indent=2)
            fh.write("\n")


rule emit_smoke_summary:
    output:
        "results/smoke/summary.json"
    run:
        Path(output[0]).parent.mkdir(parents=True, exist_ok=True)
        payload = _summary_payload("smoke")
        with open(output[0], "w", encoding="ascii") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
