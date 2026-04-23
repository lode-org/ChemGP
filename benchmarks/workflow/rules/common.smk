from pathlib import Path


def task_cases(task_name):
    return config["tasks"].get(task_name, {}).get("cases", [])


def task_run_meta(task_name):
    targets = []
    for case in task_cases(task_name):
        for variant in case.get("variants", []):
            targets.append(f"results/{task_name}/{case['id']}/{variant}/run_meta.json")
    return targets


def task_run_jsonl(task_name):
    targets = []
    for case in task_cases(task_name):
        for variant in case.get("variants", []):
            targets.append(f"results/{task_name}/{case['id']}/{variant}/raw.jsonl")
    return targets


rule run_case:
    input:
        cfg="config/benchmarks.yaml",
        runner="scripts/run_case.py",
    output:
        meta="results/{task}/{case}/{variant}/run_meta.json",
        jsonl="results/{task}/{case}/{variant}/raw.jsonl",
    shell:
        (
            "python {input.runner} "
            "--config {input.cfg} "
            "--task {wildcards.task} "
            "--case {wildcards.case} "
            "--variant {wildcards.variant} "
            "--output-jsonl {output.jsonl} "
            "--output-meta {output.meta}"
        )


rule task_summary:
    input:
        lambda wc: task_run_meta(wc.task),
    output:
        "results/{task}/summary.json",
    params:
        task=lambda wc: wc.task,
    shell:
        (
            "python scripts/task_summary.py "
            "--task {params.task} "
            "--output {output} "
            "{input}"
        )
