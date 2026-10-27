# SensorLine guide

The [README](../README.md) covers the problem, the results and the quick start.
This guide covers what the README leaves out: the data format, the config,
the training modes, the dashboard and counting in long recordings.

Paths inside a YAML or JSON file are relative to that file; paths on the command
line are relative to the current folder. Every config in `configs/` uses
`device: auto` (CUDA when PyTorch sees it, else CPU); `--device` overrides it.

## Data

A split is a folder with `images/` (grayscale PNG or TIFF, one frame each) and
`labels/` (YOLO text files, class 0 only). A frame is `sensors × T`: the sensors
lie along the image axis named by `sensor_axis`, time along the other one, and
`T` may differ between frames. A label file may be empty; a missing one is an error.

A dataset folder can describe itself in a `data.yaml`, in the same form as the
`data` section of a model YAML:

```yaml
data:
  type: yolo_images
  train: train
  val: val
  sensor_axis: x     # the sensors run along x (image width)
  scale: 255.0       # signal = (pixel - offset) / scale
  offset: 0.0
```

`train.py --data <folder>` trains on such a dataset instead of the YAML's data.
`test.py` and the dashboard read a folder by its `data.yaml` when it has one, so a
float32 TIFF in signal units (`scale: 1`) and an 8-bit PNG (`scale: 255`) give a
model the same signal. Without `data.yaml` a folder is read as the checkpoint was trained.

## Config

A model YAML has five sections; unknown options and invalid combinations are
rejected on load. All options, their defaults and what Optuna may tune are listed
in one table, `SECTIONS` in `src/sensorline/config.py`.

- `data`: see above; `sensors` (default 40) is the number of sensors per frame.
- `model.backbone`: a list of blocks. `conv`, `residual`, `depthwise` and `tcn`
  take `kernel`, `stride` and `dilation` as `[sensor, time]` pairs, plus
  `activation` and `norm` (`batch`, `group`, `layer`, `none`). `eca` (channel
  attention) and `axial_attention` (attention along time and along sensors) keep
  the channels and the resolution.
- `model.neck`: `none`, `feature_fusion` (two layers of one resolution) or `fpn`
  (layers of different resolutions).
- `model.head`: `bbox`, or `bbox_decoupled` with separate box and objectness branches.
- `loss`: box loss `iou`, `giou`, `diou` or `ciou`; objectness `bce` or `focal`;
  weights and `hard_negative_ratio`. It is written in full: thresholds and loss
  decide what a run measures and learns, so they have no silent defaults.
- `training`: optimizer, learning rate, batch size, epochs, seed and the
  `augmentation` block (flips, rotation, shear, gain, noise, shifts; training
  frames only). `lr_schedule: cosine`, `warmup_epochs` and `ema_decay` are off by
  default; they did not help in a check on the control model (3 seeds).
- `evaluation`: `confidence`, `nms_iou`, `match_iou` and the candidate limits.

`python tools/model_info.py --config <yaml> --time 120` prints the layer shapes,
parameters and receptive field before any training.

## Training

`train.py` trains one model per call. The best epoch is chosen on validation by
count MAE, then mAP50, then loss. A run is repeatable bit for bit on the same
device for the same seed; `--seed` overrides the YAML to repeat a model with
several seeds. `--dry-run` validates everything and prints the plan.

**Optuna.** `--optuna configs/optuna.json` searches training, loss, augmentation
and threshold parameters (never the architecture) on validation; the JSON names
the objective metric and the ranges. Each trial is an ordinary run in
`trials/trial_NNNNN`; `best_params.json` holds the best one. `--resume --trials N`
adds trials to a study and refuses if the data, the config or the search changed.
`--best-params best_params.json` trains once with the result.

**Fine-tuning.** `--finetune base.pt` continues a checkpoint on new data, given by
the YAML or by `--data`. The architecture must match the checkpoint. Loss,
thresholds and optimizer settings come from the checkpoint (each replaced YAML
value is printed); the YAML gives the data, epochs, seed and device. Three
precautions protect what the model already does well:

| Option | Default | Effect |
|---|---|---|
| `--lr-scale` | 0.1 | multiplier of the base run's learning rate |
| `--replay-ratio` | 1 | base training frames replayed per new frame in each epoch; 0 turns replay off |
| `--max-f1-drop` | 0.005 | an epoch is accepted only if F1 on the base validation drops by at most this |
| `--max-count-mae-rise` | 0.01 | … and count MAE on it rises by at most this |

The base weights are the first candidate, so `best.pt` keeps them when no epoch is
accepted. The count MAE limit must be at least 1 / (frames of the base validation),
or a single miscounted frame rejects every epoch; the run warns about that.
`finetune.json` records the outcome. `python tools/finetune_demo.py demo` shows the
effect on synthetic data.

## Evaluation and comparison

`test.py --checkpoint best.pt` evaluates on the YAML's validation, on a labeled
folder (`--split test --data-dir <folder>`) or predicts counts for unlabeled frames
(`--unlabeled --data-dir <folder>`). It writes `metrics.json`, `samples.csv`
(per frame) and `predictions.json`; the dashboard's Inference section runs the same code.

`tools/compare_runs.py baseline other…` compares models evaluated on the same
frames by a paired bootstrap over frames (`--metric f1|precision|recall|count_mae`).
A folder may be a run, a `test.py` report or an Optuna study (its best trial).
"not distinguishable" means the 95% interval of the difference contains zero.

## Dashboard

```bash
python dashboard.py --config configs/toy.yaml
```

It opens in the browser on the first free port from 8008 (`--port`, or
`--no-browser` to only print the address). `--config` names the dataset whose
statistics are shown; the YAMLs next to it and the runs under `--runs-root`
(default `runs`) can be inspected. Sections:

- **Inference**: a checkpoint on its own splits or any folder; counts, metrics,
  a confidence sweep and every frame with its found, missed and false boxes.
- **Experiments**: the runs, their training curves and model summaries.
- **Model**: parameters, MACs, receptive field and layer table of a YAML or a
  checkpoint; speed and memory measured on request.
- **Dataset analysis** (Overview, Samples, Sensors, Spatial, Objects,
  Foreground / Background, Train vs Val, Noise): sensor statistics, correlation
  between sensors, object sizes and overlaps, train/val drift and duplicate frames.

`tools/analyze.py --config <yaml>` writes the same data statistics as JSON.

## Counting in long recordings

```bash
python track.py
```

opens a page (ports from 8010) where you choose a CSV recording, a checkpoint and
what the CSV holds. The recording is cut into overlapping windows of the model's
frame length, the model finds boxes in each window, and a tracker links them
across windows so that every object is counted once. Defaults live in `configs/tracking/default.yaml`;
`tools/track_run.py` runs one recording without the page and
`tools/track_folder.py` a folder of them.

With the toy model from the quick start:

```bash
python tools/frames_to_csv.py toy_data/val toy_data/recording --limit 30
python tools/track_run.py --csv toy_data/recording/recording.csv --model runs/toy/best.pt \
    --set data.input_kind=pixels --set data.sample_rate_hz=auto
```

`input_kind: pixels` ("Pixels" on the page) reads a recording stored like the
training images and works with any checkpoint. The default, `raw_adc`, reads raw
conveyor recordings and needs the transform from recordings to frames that
`tools/locate_frames.py` reconstructs from the private data.

## Tests

```bash
python -m unittest discover -s tests
```
