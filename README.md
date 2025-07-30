# ArmNN to Perfetto Trace Converter

This Python tool converts ArmNN profiling JSON traces into a [Perfetto](https://ui.perfetto.dev) compatible trace format.

It extracts:
- Top-level ArmNN events using `"Wall clock time (Start/Stop)"`
- OpenCL kernel timers (e.g., `OpenClKernelTimer/...`) as child spans
- Clean kernel names (e.g., `gemm_mm_reshaped_only_rhs_nt_mmul_texture_fp16`)
- Optional thread-to-thread flow arrows between parent and kernel spans

---

## 🔧 Usage

```bash
python armnn_to_perfetto.py --input out.json --output perfetto_trace.json [--flows]
```

### Arguments:

| Flag       | Description                                     |
|------------|-------------------------------------------------|
| `--input`  | Path to the input JSON file (from ArmNN)        |
| `--output` | Path to the generated Perfetto trace JSON       |
| `--flows`  | _(Optional)_ Add arrows between parent and child spans (experimental) |

---

## 🧠 Features

- Supports messy input JSON files (extra text before/after actual JSON)
- Normalizes OpenCL kernel names:
  - Span name: `gemm_mm_reshaped_only_rhs_nt_mmul_texture_fp16`
  - Extra info (e.g., `GWS[...]`, `#ID`) is moved to `args`
- Produces well-organized tracks:
  - `tid=0`: ArmNN high-level spans
  - `tid=1`: GEMM MM kernels
  - `tid=2`: Other OpenCL kernels
- Named threads and process for better visualization in Perfetto

---

## ⚠️ Flow Arrows (`--flows`) Limitation

The `--flows` flag emits visual arrows from parent events to kernel spans using Perfetto's `flow` protocol.

**⚠️ This is not fully accurate.**  
Arrows are generated with a basic counter and may misalign when timestamps overlap or multiple flows exist within one span. You may disable it if visual clarity is impacted.

---

## 🖼 Example Output

```json
{
  "name": "gemm_mm_reshaped_only_rhs_nt_mmul_texture_fp16",
  "cat": "opencl.gemm_mm",
  "ph": "X",
  "ts": 123456789,
  "dur": 12345,
  "pid": 0,
  "tid": 1,
  "args": {
    "parent": "fused-Conv2D:...",
    "gws": "32,64,16",
    "lws": "4,4,1",
    "kernel_id": "#1234",
    "raw_name": "OpenClKernelTimer/3: gemm_mm_reshaped_only_rhs_nt_mmul_texture_fp16 GWS[32,64,16] LWS[4,4,1]_#1234"
  }
}
```

---

## 🧪 Visualizing in Perfetto

1. Run the script to generate your `.json` trace.
2. Open [https://ui.perfetto.dev](https://ui.perfetto.dev).
3. Drag and drop the output file.
4. Search for kernel slices using:
```sql
SELECT * FROM slice WHERE cat = 'opencl.gemm_mm';
```

---

## 📄 License

MIT License © 2024
