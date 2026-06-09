# Ultralytics → DeepX Export Reference

> One-shot `format=deepx` exporter built into Ultralytics YOLO. Converts a YOLO
> `.pt` to a deployable DeepX NPU model (`.dxnn`) in a single command —
> internally running ONNX export → INT8 EMA calibration → `dx_com` compilation →
> packaging. Use this BEFORE hand-rolling the PT→ONNX→`dxcom` pipeline for
> Ultralytics YOLO **detection** models.

## When to use this path (decision matrix)

| Situation | Path |
|---|---|
| Ultralytics YOLO **detection** model (`.pt`) → DeepX NPU | **`format=deepx` one-shot** (this doc) |
| Non-detection task (seg/pose/cls/obb), or export currently unsupported | Manual PT→ONNX (`dx-agentic-compiler-convert`) → `dxcom` (`dxcom-cli.md`) |
| Arbitrary / non-YOLO ONNX, custom graph, or fine control over config.json | Direct `dxcom` / `dx_com.compile()` (`dxcom-api.md`, `config-schema.md`) |
| Already have a clean `.onnx` and just need `.dxnn` | Direct `dxcom` |

The one-shot path is preferred for the YOLO-detection→DeepX case because it
handles ONNX export quirks (single-output graph), INT8 calibration, and `dx_com`
invocation automatically — eliminating the most common manual-pipeline errors.

## Platform constraints (HARD)

- **x86-64 Linux only** for the export/compile step — `dx_com` does **not**
  support ARM64/aarch64. Always export on an x86-64 Linux host.
- **Detection models only** (current release). Other tasks may be added later.
- **INT8 is enforced** — `int8=True` is set automatically; passing `int8=False`
  is overridden with a warning. There is no FP16/FP32 DeepX output.
- Target NPU is **DX-M1** (`dx_m1`), consistent with the rest of dx-compiler.

## Installation

```bash
pip install ultralytics
# dx_com is installed automatically from the DeepX SDK on first `format=deepx` export.
# dx_engine (inference runtime) is installed automatically on first deployment use.
```

## Export — API and CLI

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")            # load (auto-downloads from Ultralytics if absent)
model.export(format="deepx")          # creates 'yolo26n_deepx_model/' (int8=True enforced)
```

```bash
# CLI equivalent
yolo export model=yolo26n.pt format=deepx     # creates 'yolo26n_deepx_model/'
```

### Export arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `format` | str | `'deepx'` | Target format — DeepX NPU. |
| `imgsz` | int \| tuple | `640` | Input size; int (square) or `(h, w)`. |
| `batch` | int | `1` | Export batch size. Keep `1` for DX-M1 (NPU is batch=1). |
| `int8` | bool | `True` | INT8 quantization — **enforced**; `False` is overridden with a warning. |
| `data` | str | `'coco128.yaml'` | Calibration dataset config (image source for INT8 calibration). |
| `fraction` | float | `1.0` | Fraction of calibration data to use. 100–400 images is typically enough. |
| `device` | str | `None` | Export device: `0` (GPU) or `cpu`. |

Calibration uses **EMA** with a default of **100 images**; more than a few
hundred rarely improves accuracy. Tune via `data` / `fraction`.

## Output structure

```
yolo26n_deepx_model/
├── yolo26n.dxnn     # Compiled NPU binary — loaded directly by dx_engine
├── config.json      # Calibration + preprocessing configuration
└── metadata.yaml    # Class names, image size, task, etc.
```

The output is a **directory** (`*_deepx_model/`), not a bare `.dxnn`. The
`metadata.yaml` is what lets the Ultralytics inference pipeline reattach class
names and the postprocessor.

## Deploy — run inference on the exported model

```python
from ultralytics import YOLO

model = YOLO("yolo26n_deepx_model")              # load the exported DeepX model dir
results = model("https://ultralytics.com/images/bus.jpg")
for r in results:
    print(f"Detected {len(r.boxes)} objects")
    r.show()
```

```bash
yolo predict model='yolo26n_deepx_model' source='https://ultralytics.com/images/bus.jpg'
```

The DeepX backend converts each input from normalized-float BCHW `[0, 1]` to
uint8 HWC `[0, 255]` before handing it to the `dx_engine` runtime, as required by
the inference contract. Inference with the exported `.dxnn` runs on any platform
the `dx_engine` runtime supports (the ARM64 restriction applies only to the
export/compile step).

### Advanced: deploy through the dx_app IFactory pattern

For integration into the dx-runtime app framework (multi-model pipelines,
SyncRunner/AsyncRunner, custom visualizers), the `.dxnn` inside `*_deepx_model/`
can be consumed by a dx_app IFactory app instead of the Ultralytics backend. Use
this only when the app framework is required; the `YOLO(...)` backend above is
sufficient for standalone inference. See `dx-runtime/dx_app/CLAUDE.md`.

## FAQ

- **Why a directory instead of one `.dxnn`?** The `config.json` and
  `metadata.yaml` carry calibration/preprocessing settings and class metadata the
  runtime needs; bundling keeps deployment self-contained.
- **Export fails on my ARM board.** Expected — run the export on an x86-64 Linux
  host. Only the export/compile step is x86-64-restricted.
- **My model is segmentation/pose.** The one-shot path is detection-only for now;
  fall back to manual PT→ONNX→`dxcom` (`dx-agentic-compiler-convert` + `dxcom-cli.md`).
- **Can I deploy custom-trained YOLO?** Yes — any detection model trained with
  Ultralytics Train Mode and exported with `format="deepx"` deploys on DX-M1.

## References

- Authoritative integration doc: `ultralytics/docs/en/integrations/deepx.md`
- Direct compiler paths: `dxcom-cli.md`, `dxcom-api.md`, `config-schema.md`
- Conversion fallback skill: `dx-agentic-compiler-convert`
