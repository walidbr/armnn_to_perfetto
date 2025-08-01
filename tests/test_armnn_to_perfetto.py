import json
import os
import tempfile
import pytest

from armnn_to_perfetto import generate_perfetto_trace

# Path to the sample input trace in the repo
TEST_TRACE = os.path.join(os.path.dirname(__file__), '..', 'test_trace.json')

def load_output(tmp_path, **kwargs):
    out_file = tmp_path / 'out.json'
    # Call the converter
    generate_perfetto_trace(
        input_path=TEST_TRACE,
        output_path=str(out_file),
        **kwargs
    )
    # Load and return parsed JSON
    with open(out_file, 'r') as f:
        return json.load(f)

def test_default_begin_end_events(tmp_path):
    data = load_output(tmp_path, emit_flows=False, use_beg_end=True)
    # 3 metadata + 2 parent (B/E) + 2 kernel (B/E) = 7 events
    events = data.get('traceEvents', [])
    assert data.get('displayTimeUnit') == 'ns'
    assert len(events) == 7
    # Find parent begin event
    parent_b = [e for e in events if e.get('name') == 'Foo' and e.get('ph') == 'B']
    assert parent_b, "Missing parent begin event"
    assert parent_b[0]['ts'] == 100
    # Find kernel begin event
    kernel_b = [e for e in events if e.get('name') == 'baz' and e.get('ph') == 'B']
    assert kernel_b, "Missing kernel begin event"
    assert kernel_b[0]['ts'] == 100

def test_complete_events(tmp_path):
    data = load_output(tmp_path, emit_flows=False, use_beg_end=False)
    # 3 metadata + 1 parent X + 1 kernel X = 5 events
    events = data.get('traceEvents', [])
    assert len(events) == 5
    # Parent complete event
    parent_x = [e for e in events if e.get('name') == 'Foo' and e.get('ph') == 'X']
    assert parent_x, "Missing parent complete event"
    assert parent_x[0]['dur'] == 100
    # Kernel complete event
    kernel_x = [e for e in events if e.get('name') == 'baz' and e.get('ph') == 'X']
    assert kernel_x, "Missing kernel complete event"
    assert kernel_x[0]['dur'] == 50

def test_flows_enabled(tmp_path):
    data = load_output(tmp_path, emit_flows=True, use_beg_end=True)
    events = data.get('traceEvents', [])
    # Flows insert 2 events per kernel: s and f
    # total = 3 metadata + 2 parent + 2 kernel + 2 flow = 9
    assert len(events) == 9
    # Check presence of flow start (ph='s') and flow end (ph='f')
    phases = [e.get('ph') for e in events]
    assert 's' in phases
    assert 'f' in phases
