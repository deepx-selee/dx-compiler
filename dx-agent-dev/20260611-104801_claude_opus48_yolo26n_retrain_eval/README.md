# YOLO26n African-Wildlife Domain Retrain + DeepX 4-Way Benchmark

Fine-tune the stock COCO `yolo26n` detector for a **wildlife-monitoring / safari-camera**
scenario on the Ultralytics **african-wildlife** dataset (nc=4: buffalo, elephant, rhino,
zebra), then benchmark **accuracy (mAP50-95)** and **speed (FPS)** across four points:

| Model | Form | Device |
|-------|------|--------|
| base `yolo26n` | `.pt` fp32 | GPU (RTX 5060 Ti) |
| base `yolo26n` | `.dxnn` INT8 | DX-M1 NPU |
| retrained | `.pt` fp32 | GPU |
| retrained | `.dxnn` INT8 | DX-M1 NPU |

## Quick start

```bash
bash setup.sh     # verifies dx-runtime/venv-dx-runtime full stack (ultralytics+dx_com+dx_engine+torch)
bash run.sh       # train (40ep) -> export both -> eval 4 points -> sample -> report (tees session.log)
python verify.py  # acceptance check (RESULT: PASS)
```

`run.sh` activates `dx-runtime/venv-dx-runtime` and runs `pipeline.py`, which:
1. trains `yolo26n.pt` on `african-wildlife.yaml` (40 epochs, imgsz=640, batch=16, GPU);
2. exports base `yolo26n.pt` and the retrained `wildlife_yolo26n.pt` with
   `format=deepx` (INT8 enforced, EMA calibration on domain images);
3. evaluates all four points with `model.val()` (FPS = `1000 / speed['inference']`, batch=1);
4. saves an annotated detection sample (`sample_detect.jpg`) from the retrained model.

## Files

| File | Purpose |
|------|---------|
| `setup.sh` | Environment verification (full-stack venv + dx_rt sanity check) |
| `run.sh` | One-command launcher (activates venv, runs pipeline, tees `session.log`) |
| `pipeline.py` | Train + export×2 + eval×4 + sample driver |
| `verify.py` | Acceptance check (.dxnn present, 4 measured points, retrained>base, sample) |
| `results.json` | Measured mAP50-95 / mAP50 / FPS for all four points |
| `report.md` | 4-way comparison table + analysis (accuracy gain, INT8 effect) |
| `sample_detect.jpg` | Retrained model on a validation image (boxes + class labels) |
| `yolo26n_deepx_model/` | Base DeepX export (`yolo26n.dxnn`, `config.json`, `metadata.yaml`) |
| `wildlife_yolo26n_deepx_model/` | Retrained DeepX export (`wildlife_yolo26n.dxnn`, ...) |
| `yolo26n.pt` / `wildlife_yolo26n.pt` | Base + retrained PyTorch weights |
| `runs/train/` | Ultralytics training run (best.pt, curves, confusion matrix) |
| `session.log` | Real captured pipeline output |

## Environment

- `dx-runtime/venv-dx-runtime`: ultralytics 8.4.63, dx_com 2.3.0-rc.5, dx_engine 3.3.2,
  torch 2.12.0+cu130 (CUDA, RTX 5060 Ti). DX-M1 NPU, dx_rt sanity check PASSED.
- DeepX export/compile is x86-64 Linux only, detection-only, INT8-enforced (per the
  `ultralytics-deepx-export.md` KB).

## Notes

- The base COCO model scores near-zero mAP on african-wildlife (its 80 COCO class
  indices don't match the 4 domain classes) — this is exactly why domain fine-tuning is
  needed. See `report.md` for the measured numbers and analysis.
