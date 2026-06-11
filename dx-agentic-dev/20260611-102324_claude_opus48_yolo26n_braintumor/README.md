# YOLO26n Brain-Tumor Retrain → DeepX · 4-Way Evaluation

Adapt the COCO-pretrained `yolo26n` general detector into a **brain-tumor screening
model** for a medical edge device (MRI/CT brain scans), then compare accuracy
(mAP50-95) and speed (FPS) across four configurations:

| # | Model | Form | Device |
|---|-------|------|--------|
| 1 | base `yolo26n` | PyTorch fp32 | RTX 5060 Ti GPU |
| 2 | base `yolo26n` | DeepX `.dxnn` INT8 | DX-M1 NPU |
| 3 | retrained | PyTorch fp32 | RTX 5060 Ti GPU |
| 4 | retrained | DeepX `.dxnn` INT8 | DX-M1 NPU |

Dataset: Ultralytics `brain-tumor` (893 train / 223 val), classes `negative` /
`positive`. Fine-tune = 40 epochs, imgsz 640, batch 16, seed 0, on the local GPU.
DeepX export via the one-shot `format=deepx` path (INT8 EMA calibration → `dx_com`).

## Quick start

```bash
./setup.sh            # sanity-check dx_rt, resolve venv, verify imports
./run.sh              # run the full train → eval → export → eval pipeline
python verify.py      # fp32 PyTorch vs INT8 DeepX agreement on a val image
```

`run.sh` reuses `dx-runtime/venv-dx-runtime` (ultralytics 8.4.63, dx_com 2.3.0-rc.5,
dx_engine 3.3.2, torch 2.12+cu130). The pipeline takes ~25–40 min (40-epoch training
dominates); progress is mirrored to `session.log`.

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Driver: base fp32 eval → train 40 ep → retrained fp32 eval → export both → both NPU INT8 evals → sample image |
| `setup.sh` | Env setup (SUITE_ROOT autodetect, dx_rt sanity, venv reuse, import check) |
| `run.sh` | One-command pipeline launcher (venv activate + `tee session.log`) |
| `verify.py` | fp32 PyTorch vs INT8 DeepX numerical agreement → `RESULT: PASS/FAIL` |
| `metrics.json` | All four measured points (mAP50-95 / mAP50 / FPS / latency / per-class) |
| `report.md` | 4-way comparison table + analysis (accuracy gain + INT8 quantization effect) |
| `sample_detect.jpg` | Retrained model on a representative val image (boxes + class labels) |
| `session.log` | Real captured stdout/stderr of the pipeline run |
| `yolo26n.pt` | Base COCO weights (auto-downloaded) |
| `yolo26n_braintumor.pt` | Retrained best weights (copied from training) |
| `yolo26n_deepx_model/` | Base DeepX export (`yolo26n.dxnn`, `config.json`, `metadata.yaml`) |
| `yolo26n_braintumor_deepx_model/` | Retrained DeepX export (`.dxnn` + config + metadata) |
| `runs/train_braintumor/` | Ultralytics training run (weights, curves, plots) |

## Knowledge base

Built per `dx-compiler/.deepx/toolsets/ultralytics-train-eval.md` and
`ultralytics-deepx-export.md`. Export is x86-64-Linux-only, detection-only, INT8-enforced;
NPU INT8 eval requires the `dx_rt` runtime (`dx_engine`), confirmed by the sanity check.
