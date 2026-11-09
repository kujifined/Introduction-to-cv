"""Execute the live notebook's actual cells in a fresh IPython process offline."""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path


def block_network():
    attempts = []
    socket_operations = {
        "socket.bind",
        "socket.connect",
        "socket.getaddrinfo",
        "socket.gethostbyaddr",
        "socket.gethostbyname",
        "socket.getnameinfo",
        "socket.sendto",
        "socket.sendmsg",
    }
    process_operations = {"subprocess.Popen", "os.exec", "os.fork", "os.posix_spawn", "os.system"}

    def guard(event, args):
        if (
            event in socket_operations
            or event in process_operations
            or (event == "socket.__new__" and args[1] in (socket.AF_INET, socket.AF_INET6))
        ):
            attempts.append(event)
            raise PermissionError(f"Offline smoke test blocked {event}")

    sys.addaudithook(guard)
    probes = []
    for family in (socket.AF_INET, socket.AF_INET6):
        try:
            with socket.socket(family):
                raise RuntimeError(f"Network guard did not block {family.name}")
        except PermissionError as error:
            if str(error) != "Offline smoke test blocked socket.__new__":
                raise
            probes.append(family.name)
    attempts.clear()
    return attempts, probes


def execute_notebook(directory):
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from traitlets.config import Config

    notebook = json.loads((directory / "09_live_demo.ipynb").read_text(encoding="utf-8"))
    config = Config()
    config.HistoryManager.enabled = False
    shell = InteractiveShell.instance(config=config)
    timings, html_outputs = {}, {}
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        started = time.perf_counter()
        with capture_output() as capture:
            run = shell.run_cell("".join(cell["source"]), store_history=False)
        if not run.success:
            raise RuntimeError(
                f"Notebook cell {cell['id']} failed:\n{capture.stdout}{capture.stderr}"
            ) from (run.error_before_exec or run.error_in_exec)
        timings[cell["id"]] = time.perf_counter() - started
        html_outputs[cell["id"]] = [
            out.data["text/html"] for out in capture.outputs if "text/html" in out.data
        ]
    return shell.user_ns, timings, html_outputs


def verify_references(directory, namespace, html_outputs):
    with (directory / "demo_assets/demo_reference_outputs.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        expected = {(row["case_id"], row["configuration"]): row for row in csv.DictReader(stream)}
    demo = namespace["demo"]
    checks = []
    for case_id in demo.cases:
        result = namespace["demo_results"][case_id]
        if len(html_outputs[case_id]) != 1:
            raise ValueError(f"Expected one rendered case output for {case_id}")
        html = html_outputs[case_id][0]
        expected_images = 1 if result["case"]["corruption"] == "clean" else 2
        if html.count("data:image/png;base64,") != expected_images or html.count("<tr>") != 5:
            raise ValueError(f"Missing case images or prediction rows for {case_id}")
        for name, prediction in result["predictions"].items():
            reference = expected.pop((case_id, name))
            p = prediction["probabilities"]
            if (
                int(reference["seed"]) != demo.seed
                or any(
                    str(result["case"][field]) != reference[field]
                    for field in ("path", "true_class", "true_index", "corruption", "severity")
                )
                or (prediction["top1"], prediction["top2"])
                != (reference["predicted_class"], reference["top2_class"])
            ):
                raise ValueError(f"Metadata or ranking mismatch for {case_id}/{name}: {prediction}")
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()) or not math.isclose(
                sum(p.values()), 1.0, abs_tol=1e-6
            ):
                raise ValueError(f"Invalid probabilities for {case_id}/{name}: {p}")
            reference_p = {label: float(reference[f"prob_{label}"]) for label in p}
            max_delta = max(abs(v - reference_p[label]) for label, v in p.items())
            if max_delta > 0.0005:
                raise ValueError(
                    f"Probability difference > 0.0005 for {case_id}/{name}: {max_delta}"
                )
            checks.append(
                {
                    "case_id": case_id,
                    "configuration": name,
                    "true_class": reference["true_class"],
                    **prediction,
                    "reference_probabilities": reference_p,
                    "max_abs_delta": max_delta,
                }
            )
    if expected or len(checks) != 12:
        raise ValueError("Expected all 12 frozen case/model comparisons.")
    return checks


def main():
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=Path, help="Local final-results ZIP or extracted directory"
    )
    parser.add_argument(
        "--output", type=Path, default=directory.parent / "validation/live_demo_smoke_test.json"
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.weights:
        os.environ["ICV_DEMO_WEIGHTS"] = str(args.weights.resolve())
    os.chdir(directory)
    attempts, probes = block_network()
    with tempfile.TemporaryDirectory(prefix="icv-demo-ipython-") as tmp:
        os.environ["IPYTHONDIR"] = tmp
        namespace, timings, html_outputs = execute_notebook(directory)
    comparisons = verify_references(directory, namespace, html_outputs)
    if attempts:
        raise RuntimeError(f"Notebook attempted to use network or external processes: {attempts}")
    demo = namespace["demo"]
    max_delta = max(row["max_abs_delta"] for row in comparisons)
    result = {
        "status": "pass",
        "execution": "all notebook code cells in a fresh in-process IPython session",
        "offline_guard": (
            "Python audit hook blocks IP sockets, DNS, socket IO and external processes"
        ),
        "blocked_probe_families": probes,
        "network_or_process_attempts_during_execution": attempts,
        "environment": {
            "python": sys.version.split()[0],
            "os": platform.system(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "device": "cpu",
            "packages": {
                p: version(p) for p in ("torch", "torchvision", "Pillow", "IPython", "ipykernel")
            },
        },
        "notebook_sha256": hashlib.sha256(
            (directory / "09_live_demo.ipynb").read_bytes()
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            (directory / "live_demo.py").read_bytes()
        ).hexdigest(),
        "seed": demo.seed,
        "weights_source": "zip" if namespace["weights_source"].is_file() else "directory",
        "verified_checkpoints": demo.spec["checkpoints"],
        "verified_input_hashes": {case["path"]: case["raw_sha256"] for case in demo.cases.values()},
        "cell_elapsed_seconds": timings,
        "max_abs_probability_delta_vs_kaggle": max_delta,
        "probability_tolerance": 0.0005,
        "comparisons": comparisons,
    }
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASS: 3 cases x 4 models, all top-1/top-2 match; max probability delta {max_delta:.3g}."
    )
    print(
        f"Offline probes blocked: {', '.join(probes)}. "
        f"Notebook network/process attempts: {len(attempts)}."
    )
    print("Cell seconds:", {k: round(v, 3) for k, v in timings.items()})
    print(output)


if __name__ == "__main__":
    main()
