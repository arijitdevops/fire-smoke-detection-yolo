# Training

Scripts in this directory prepare the dataset, fine-tune a YOLO detector on it,
evaluate the result and export deployment artefacts. They are ordinary CLI
programs -- nothing here imports the backend, and the backend never imports
these.

| Script | Purpose |
| --- | --- |
| `prepare_data.py` | Validate the dataset and regenerate `data.generated.yaml`. |
| `train.py` | Fine-tune `yolo11n/s/m` or `yolov8n/s/m` on the generated descriptor. |
| `validate.py` | Run `model.val()` and write `reports/metrics.json`. |
| `export.py` | Export `best.pt` to ONNX / TensorRT / TorchScript / OpenVINO. |
| `configs/` | Hyper-parameter override files passed via `--config`. |

Relative paths (`../_datasets/smoke_fire_detection`, `runs/detect/...`) are
resolved against the repository root, so the defaults work from any working
directory. Keys in a `--config` file take precedence over command-line flags.

```bash
# Windows example with the dataset outside the default location
python training/prepare_data.py --data-dir D:\sample_projects\_datasets\smoke_fire_detection
python training/train.py --data-dir D:\sample_projects\_datasets\smoke_fire_detection --device 0
python training/validate.py --split test
```

Trained weights land in `runs/detect/fire_smoke/weights/best.pt`, which is the
API's default `MODEL_PATH`. See the root `README.md` for the full walkthrough.
