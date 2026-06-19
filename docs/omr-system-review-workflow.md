# OMR System Review Workflow

This workflow builds a human-reviewed dataset for reliable system detection in
music score PDFs. The review data is deliberately separate from YOLO training
data: model predictions are only used as a starting point, and only reviewed
pages should be exported for training.

## Goal

The first useful page cut is a reliable list of `system` boxes. Other layout
classes, especially `grand_staff`, can be useful as visual guides, but the
review target is currently system boxes.

## 1. Prepare Review Data

Render a PDF, run the current layout model, and create editable review files:

```bash
.venv-omr/bin/python scripts/omr_prepare_review_book.py \
  "/path/to/score-book.pdf" \
  --weights /path/to/best.pt \
  --out /tmp/omr-review/score-book \
  --device cpu
```

Useful options:

```text
--start-page N        first PDF page to process, 1-based
--max-pages N         limit pages per input PDF
--pdf-dpi N           render DPI, default 300
--conf N              YOLO confidence threshold, default 0.25
--iou N               YOLO NMS IoU threshold, default 0.65
--review-classes CSV  editable classes, default system
--helper-classes CSV  non-editable helper classes, default grand_staff
--force               replace an existing output directory
```

Example smoke run:

```bash
.venv-omr/bin/python scripts/omr_prepare_review_book.py \
  "/path/to/score-book.pdf" \
  --weights /path/to/layout-best.pt \
  --out /tmp/omr-review-smoke-score \
  --start-page 7 \
  --max-pages 2 \
  --device cpu \
  --force
```

## Campaign-Level Workflow

For the next iteration, do not hand-pick one score at a time. Create a review
campaign that records the corpus sample, negative-page mix, model settings,
review dirs, export command, and retrain commands in one `campaign.json`:

```bash
.venv-omr/bin/python scripts/omr_create_review_campaign.py \
  /path/to/score-pdf-corpus \
  --negative-input /path/to/text-only-or-newspaper-pdfs \
  --out /path/to/omr-campaigns/systems-v002 \
  --name systems-v002 \
  --weights /path/to/reviewed-system-yolo-best.pt \
  --base-model /path/to/base-layout-model.pt \
  --target-pages 3000 \
  --negative-ratio 0.15 \
  --pages-per-book 24 \
  --iou 0.5 \
  --device 0
```

By default this only plans the campaign. Add `--prepare` to actually create all
review directories by running `omr_prepare_review_book.py` for each planned
batch. The campaign keeps score pages and negative pages explicit:

```text
campaign-root/
  campaign.json
  review/
    batch-0001-score-...
    batch-0126-negative-...
  datasets/
  runs/
```

Negative inputs should include title pages, prefaces, indexes, blank pages,
library scans, text-only books, newspaper-like layouts, and anything else that
should produce zero `system` boxes. The exporter keeps reviewed pages with no
accepted boxes by default, so these become proper negative training examples.

## Review Directory Format

The preparation script writes:

```text
review-dir/
  manifest.json
  pages/
    page-0001.png
    page-0002.png
  predictions/
    page-0001.json
    page-0002.json
  reviewed/
    page-0001.json
    page-0002.json
```

`manifest.json` tracks source inputs, model settings, page order, and review
status:

```json
{
  "version": 1,
  "source_inputs": ["/path/to/score-book.pdf"],
  "render_dpi": 300,
  "model": {
    "path": "/path/to/best.pt",
    "confidence": 0.25,
    "iou": 0.65,
    "imgsz": 1536,
    "device": "cpu"
  },
  "review_classes": ["system"],
  "helper_classes": ["grand_staff"],
  "pages": [
    {
      "sequence": 1,
      "page": 7,
      "image": "pages/page-0001.png",
      "prediction": "predictions/page-0001.json",
      "review": "reviewed/page-0001.json",
      "status": "pending"
    }
  ]
}
```

`reviewed/page-XXXX.json` is the editable source of truth:

```json
{
  "version": 1,
  "page": 7,
  "sequence": 1,
  "source": "/path/to/score-book.pdf",
  "image": "pages/page-0001.png",
  "width": 2533,
  "height": 3349,
  "status": "pending",
  "boxes": [
    {
      "id": "p0007-box-0001",
      "class": "system",
      "bbox": [232.2, 1963.7, 2312.4, 2231.7],
      "source": "model",
      "confidence": 0.9357,
      "status": "pending"
    }
  ],
  "helpers": [
    {
      "id": "p0007-helper-0001",
      "class": "grand_staff",
      "bbox": [227.6, 2836.6, 2312.4, 3111.5],
      "source": "model",
      "confidence": 0.9266
    }
  ]
}
```

Coordinates are absolute image pixels in `[left, top, right, bottom]` order.

`predictions/page-XXXX.json` preserves the original model output and timing.
The editor updates only `reviewed/` and `manifest.json`.

## 2. Correct Boxes

Open the review UI:

```bash
.venv-omr/bin/python scripts/omr_review_system_boxes.py /tmp/omr-review/score-book
```

The editor shows one rendered page at a time.

Colors:

```text
orange = pending model system box
green  = accepted/reviewed system box
blue   = selected system box
gray   = helper grand_staff box
```

Controls:

```text
Right / n     next page
Left / p      previous page
a             add box mode
Escape        cancel add mode
Delete        delete selected box
s             save
Enter         mark page reviewed and save
h             toggle helper boxes
+ / -         zoom
mouse drag    move selected box
corner drag   resize selected box
```

The editor auto-saves the current page when navigating away or closing.

## 3. Export Training Data

Export reviewed pages to a system-only YOLO dataset. Replace
`/tmp/omr-review/score-book` with the `review_dir` printed by the prepare step:

```bash
.venv-omr/bin/python scripts/omr_export_reviewed_yolo.py \
  /tmp/omr-review/score-book \
  --out /tmp/simple-ai-omr-training/datasets/reviewed-systems \
  --force
```

By default the exporter reads only manifest pages with `status: reviewed`,
exports only boxes with `class: system` and `status: accepted`, and keeps
reviewed pages with zero accepted boxes as negative examples.

Useful options:

```text
--page-statuses CSV   page statuses to export, default reviewed
--box-statuses CSV    box statuses to export, default accepted
--train-split N       training split ratio, default 0.9
--seed N              deterministic train/val shuffle seed, default 7
--no-keep-empty       skip reviewed pages with no accepted system boxes
--no-copy             hard-link images when possible instead of copying
--force               replace an existing output directory
```

The output is:

```text
export-dir/
  images/train/
  images/val/
  labels/train/
  labels/val/
  data.yaml
```

For the first specialist model:

```yaml
names:
  0: system
```

Each YOLO label line:

```text
0 x_center y_center width height
```

All coordinates are normalized by the rendered image width and height.

## 4. Retrain

The retraining plan is a system-only specialist:

```text
base: previous specialist checkpoint for the main run
comparison: upstream/base checkpoint for a fresh baseline run
data: reviewed campaign pages, including explicit empty negative pages
class set: system only
```

For small reviewed datasets, do a fresh fine-tune initialized from a checkpoint.
That means pass `--model path/to/checkpoint.pt` and leave `--resume` false. Do
not resume the old optimizer state unless intentionally continuing an interrupted
training job.

Main run, initialized from the previous specialist checkpoint:

```bash
.venv-omr/bin/python scripts/omr_train_staff_system_yolo.py \
  --data /path/to/omr-campaigns/systems-v002/datasets/systems-v002-reviewed-systems/data.yaml \
  --model /path/to/reviewed-system-yolo-best.pt \
  --project /path/to/omr-campaigns/systems-v002/runs \
  --name systems-v002-from-current-checkpoint \
  --device 0 \
  --batch 8 \
  --epochs 80
```

Comparison run, initialized from the upstream/base checkpoint:

```bash
.venv-omr/bin/python scripts/omr_train_staff_system_yolo.py \
  --data /path/to/omr-campaigns/systems-v002/datasets/systems-v002-reviewed-systems/data.yaml \
  --model /path/to/base-layout-model.pt \
  --project /path/to/omr-campaigns/systems-v002/runs \
  --name systems-v002-from-base-checkpoint \
  --device 0 \
  --batch 8 \
  --epochs 80
```

The trainer name is historical; it accepts any YOLO `data.yaml`, including the
system-only export above. The first run should be the production candidate. The
base-checkpoint run is a sanity check: if it matches or beats the chained
specialist, the previous fine-tune is carrying bias and should not be used as
the next base.

Corrected target pages should be oversampled during early experiments because
they encode the annotation policy we care about for score-book page cuts.

## Validation Commands

Syntax and focused geometry tests:

```bash
python3 -m py_compile \
  scripts/omr_prepare_review_book.py \
  scripts/omr_review_system_boxes.py \
  scripts/omr_create_review_campaign.py \
  scripts/omr_predict_staff_system_yolo.py \
  tests/test_omr_grand_staff_regions.py

python3 -m unittest discover -s tests -p 'test_omr_*.py'
```
