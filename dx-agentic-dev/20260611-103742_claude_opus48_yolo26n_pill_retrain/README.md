# YOLO26n Pill Detector — Domain Retrain + DeepX 4-Way Evaluation

Adapts the COCO-pretrained `yolo26n` general detector into a single-class **`pill`**
detector for a pharmaceutical pill identification / counting station, then fairly
compares accuracy (mAP50-95) and speed (FPS) across four points:

| # | Model | Form | Device |
|---|-------|------|--------|
| 1 | base `yolo26n` | PyTorch fp32 | RTX 5060 Ti GPU |
| 2 | base `yolo26n` | DeepX INT8 `.dxnn` | DX-M1 NPU |
| 3 | retrained | PyTorch fp32 | RTX 5060 Ti GPU |
| 4 | retrained | DeepX INT8 `.dxnn` | DX-M1 NPU |

Follows the dx-compiler KB toolsets `ultralytics-train-eval.md` (retrain + fair eval)
and `ultralytics-deepx-export.md` (`format=deepx` one-shot INT8 export to DX-M1).

## Quick start

```bash
./setup.sh          # reuse dx-runtime/venv-dx-runtime (ultralytics+dx_com+dx_engine+torch/cuda)
./run.sh            # train 40ep + 4-way eval + DeepX export + sample image, then build report.md
python verify.py    # fp32 vs INT8 (.dxnn) agreement on a val image -> RESULT: PASS
```

## Pipeline stages (`pipeline.py`)

1. base `yolo26n` fp32 eval on GPU (expected mAP ≈ 0 — never saw the pill class)
2. fine-tune on `medical-pills` (40 epochs, imgsz 640, batch 16, GPU)
3. retrained fp32 eval on GPU
4. export base `yolo26n.pt` → `yolo26n_deepx_model/` (`format=deepx`, INT8)
5. export retrained `yolo26n_pill.pt` → `yolo26n_pill_deepx_model/`
6. base INT8 `.dxnn` eval on DX-M1 NPU
7. retrained INT8 `.dxnn` eval on DX-M1 NPU
8. annotated `sample_detect.jpg` (retrained model on a representative val image)

`metrics.json` is rewritten after every stage; `make_report.py` turns it into `report.md`.

## Files

| File | Purpose |
|------|---------|
| `pipeline.py` | 8-stage retrain + 4-way eval + export + sample driver |
| `make_report.py` | builds `report.md` from `metrics.json` |
| `setup.sh` / `run.sh` | environment setup / one-command re-run |
| `verify.py` | fp32-vs-INT8 numerical agreement gate |
| `report.md` | 4-way comparison + accuracy-gain + INT8 analysis |
| `metrics.json` | all four measured points (mAP + FPS) |
| `sample_detect.jpg` | annotated retrained detection |
| `session.log` | captured pipeline output |
| `yolo26n_deepx_model/`, `yolo26n_pill_deepx_model/` | base / retrained `.dxnn` exports |
| `yolo26n_pill.pt` | retrained fp32 weights |
