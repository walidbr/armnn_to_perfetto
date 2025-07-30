#!/usr/bin/env python3
"""
Convert ArmNN trace (with OpenClKernelTimer entries) to a Perfetto-compatible JSON trace.

Features:
- Extracts top-level wall-clock spans and nested kernel timers
- Converts kernel names to clean names, with full info in args
- Optionally emits flows between parent ArmNN spans and child OpenCL kernels
- Supports input JSONs with text before/after actual JSON data
- Produces thread metadata for cleaner track display in Perfetto

Usage:
    python armnn_to_perfetto.py --input out.json --output perfetto_trace.json --flows
"""

import json
import re
import argparse
import os
from typing import List, Dict, Any


def clean_json_text(file_path: str) -> Any:
    """
    Extract and parse the JSON content from a text file that may contain
    non-JSON preamble or postamble.
    """
    with open(file_path, 'r') as f:
        raw = f.read()

    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    start = raw.find('[')
    end = raw.rfind(']')
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start:end + 1])

    raise ValueError("No valid JSON object or array found in the file.")


def is_wall_clock_span(d: Dict) -> bool:
    return isinstance(d, dict) and any(k.startswith("Wall clock time (Start)") for k in d) and any(k.startswith("Wall clock time (Stop)") for k in d)


def is_kernel_measurement(k: str, v: Dict) -> bool:
    return (
        isinstance(v, dict) and
        k.startswith("OpenClKernelTimer/") and
        "raw" in v and
        isinstance(v["raw"], list) and
        isinstance(v["raw"][0], (int, float))
    )


def extract_events(obj, trace_events: List[Dict], emit_flows: bool = False, pid: int = 0, tid_base: int = 0, stack: List[str] = []):
    """
    Recursively walk through the JSON structure to extract slices and optional flow events.
    """
    global flow_id
    if isinstance(obj, dict):
        for k, v in obj.items():
            if is_wall_clock_span(v):
                name = k
                start_us = v[next(x for x in v if x.startswith("Wall clock time (Start)"))]["raw"][0]
                stop_us = v[next(x for x in v if x.startswith("Wall clock time (Stop)"))]["raw"][0]
                dur_us = stop_us - start_us

                # Parent ArmNN-level event
                trace_events.append({
                    "name": name,
                    "cat": "armnn",
                    "ph": "X",
                    "ts": start_us,
                    "dur": dur_us,
                    "pid": pid,
                    "tid": tid_base,
                    "args": {}
                })

                kernel_start = start_us
                for sub_k, sub_v in v.items():
                    if is_kernel_measurement(sub_k, sub_v):
                        kernel_name_full = sub_k
                        duration = sub_v["raw"][0]

                        # Extract cleaned kernel name
                        match = re.search(r":\s*([^\s]+)", kernel_name_full)
                        kernel_name = match.group(1) if match else kernel_name_full

                        # Parse GWS/LWS/#id into args
                        args = {
                            "parent": name,
                            "raw_name": kernel_name_full
                        }
                        gws_match = re.search(r"GWS\[(.*?)\]", kernel_name_full)
                        lws_match = re.search(r"LWS\[(.*?)\]", kernel_name_full)
                        id_match = re.search(r"(#[0-9]+)", kernel_name_full)
                        if gws_match: args["gws"] = gws_match.group(1)
                        if lws_match: args["lws"] = lws_match.group(1)
                        if id_match: args["kernel_id"] = id_match.group(1)

                        cat = "opencl.gemm_mm" if "gemm_mm_reshaped_only_rhs_nt_mmul_texture" in kernel_name else "opencl.kernel"
                        tid = 1 if cat == "opencl.gemm_mm" else 2

                        # Child event
                        trace_events.append({
                            "name": kernel_name,
                            "cat": cat,
                            "ph": "X",
                            "ts": kernel_start,
                            "dur": duration,
                            "pid": pid,
                            "tid": tid,
                            "args": args
                        })

                        # Flow arrows
                        if emit_flows:
                            trace_events.append({
                                "ph": "s",
                                "id": flow_id,
                                "ts": start_us,
                                "tid": tid_base,
                                "pid": pid,
                                "cat": "flow"
                            })
                            trace_events.append({
                                "ph": "f",
                                "id": flow_id,
                                "ts": kernel_start,
                                "tid": tid,
                                "pid": pid,
                                "cat": "flow"
                            })

                        kernel_start += duration
                        flow_id += 1

                extract_events(v, trace_events, emit_flows, pid, tid_base, stack + [k])
            else:
                extract_events(v, trace_events, emit_flows, pid, tid_base, stack + [k])
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            extract_events(item, trace_events, emit_flows, pid, tid_base, stack + [str(i)])


def generate_perfetto_trace(input_path: str, output_path: str, emit_flows: bool = False):
    """
    Loads the input ArmNN JSON trace and writes a Perfetto-compatible trace JSON.
    """
    data = clean_json_text(input_path)
    trace_events = []

    extract_events(data, trace_events, emit_flows=emit_flows)

    thread_metadata = [
        { "ph": "M", "pid": 0, "name": "process_name", "args": { "name": "ArmNN Trace" } },
        { "ph": "M", "pid": 0, "tid": 0, "name": "thread_name", "args": { "name": "ArmNN Top-Level" } },
        { "ph": "M", "pid": 0, "tid": 1, "name": "thread_name", "args": { "name": "GEMM MM Kernels" } },
        { "ph": "M", "pid": 0, "tid": 2, "name": "thread_name", "args": { "name": "Other OpenCL Kernels" } }
    ]

    perfetto_trace = {
        "traceEvents": thread_metadata + trace_events,
        "displayTimeUnit": "ns"
    }

    with open(output_path, 'w') as f:
        json.dump(perfetto_trace, f, indent=2)

    print(f"✅ Perfetto trace written to: {output_path} ({len(trace_events)} events)")


# === Script Entry Point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ArmNN trace JSON to Perfetto trace format.")
    parser.add_argument('--input', '-i', required=True, help='Path to input ArmNN JSON trace')
    parser.add_argument('--output', '-o', required=True, help='Path to output Perfetto JSON trace')
    parser.add_argument('--flows', action='store_true', help='Enable flow arrows between parent and kernel slices')
    args = parser.parse_args()

    # Global flow ID counter
    flow_id = 1

    if not os.path.exists(args.input):
        print(f"❌ Input file does not exist: {args.input}")
        exit(1)

    generate_perfetto_trace(args.input, args.output, emit_flows=args.flows)