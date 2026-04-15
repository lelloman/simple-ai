#!/usr/bin/env python3
"""Interactive OMR layout evaluator.

This is a standalone GUI for trying the published OLA YOLO model on a single
music score file. It lets you choose the input file and inference knobs, then
view each page with detection categories turned on/off independently.

Dependencies:

    pip install -r scripts/omr-layout-requirements.txt

On some Linux systems Tkinter is packaged separately:

    sudo apt install python3-tk
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError as exc:
    raise SystemExit(
        "Missing Tkinter. On Debian/Ubuntu install it with: sudo apt install python3-tk"
    ) from exc

try:
    import evaluate_omr_layout as omr_eval
except ImportError as exc:
    raise SystemExit(
        "Run this script from the repository root or from the scripts directory."
    ) from exc


CLASS_COLORS = {
    "staff": "#238b45",
    "staves": "#238b45",
    "stave": "#238b45",
    "staffmeasure": "#2171b5",
    "staff_measure": "#2171b5",
    "stave_measure": "#2171b5",
    "grandstaff": "#800080",
    "grand_staff": "#800080",
    "system": "#d95f0e",
    "systemmeasure": "#cb181d",
    "system_measure": "#cb181d",
}


@dataclass
class EvalSettings:
    file_path: Path
    model_path: Path
    output_dir: Path
    conf: float
    iou: float
    imgsz: int
    pdf_dpi: int
    max_pages: int | None
    device: str
    download_model: bool


@dataclass
class PageResult:
    source: Path
    page_index: int
    image_path: Path
    width: int
    height: int
    latency_ms: float
    detections: list[dict[str, Any]]


def normalize_class_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def class_color(name: str) -> str:
    return CLASS_COLORS.get(normalize_class_name(name), "#505050")


class SetupFrame(ttk.Frame):
    def __init__(self, app: "OmrLayoutGui") -> None:
        super().__init__(app.root, padding=16)
        self.app = app

        self.file_var = tk.StringVar()
        self.model_var = tk.StringVar(value=str(omr_eval.DEFAULT_MODEL_PATH))
        self.output_var = tk.StringVar(value="/tmp/omr-layout-gui")
        self.conf_var = tk.StringVar(value="0.15")
        self.iou_var = tk.StringVar(value="0.7")
        self.imgsz_var = tk.StringVar(value="1600")
        self.pdf_dpi_var = tk.StringVar(value="300")
        self.max_pages_var = tk.StringVar(value="5")
        self.device_var = tk.StringVar(value="cpu")
        self.download_var = tk.BooleanVar(value=True)

        self._build()

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)

        row = 0
        ttk.Label(self, text="Music score").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(self, textvariable=self.file_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(self, text="Choose...", command=self.choose_file).grid(
            row=row, column=2, sticky="ew", padx=(8, 0), pady=4
        )

        row += 1
        ttk.Label(self, text="Model").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(self, textvariable=self.model_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(self, text="Choose...", command=self.choose_model).grid(
            row=row, column=2, sticky="ew", padx=(8, 0), pady=4
        )

        row += 1
        ttk.Checkbutton(
            self,
            text="Download latest model if missing",
            variable=self.download_var,
        ).grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        ttk.Label(self, text="Output folder").grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(self, textvariable=self.output_var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(self, text="Choose...", command=self.choose_output_dir).grid(
            row=row, column=2, sticky="ew", padx=(8, 0), pady=4
        )

        row += 1
        knobs = ttk.LabelFrame(self, text="Detection knobs", padding=12)
        knobs.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(12, 8))
        for column in range(8):
            knobs.columnconfigure(column, weight=1)

        self._knob(knobs, "Confidence", self.conf_var, 0, 0)
        self._knob(knobs, "IoU", self.iou_var, 0, 2)
        self._knob(knobs, "Image size", self.imgsz_var, 0, 4)
        self._knob(knobs, "PDF DPI", self.pdf_dpi_var, 0, 6)
        self._knob(knobs, "Max pages", self.max_pages_var, 1, 0)
        self._knob(knobs, "Device", self.device_var, 1, 2)

        row += 1
        ttk.Button(self, text="Process score", command=self.process).grid(
            row=row, column=0, columnspan=3, sticky="ew", pady=(12, 0)
        )

        row += 1
        self.status = ttk.Label(self, text="")
        self.status.grid(row=row, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def _knob(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        column: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 4), pady=4)
        ttk.Entry(parent, textvariable=variable, width=10).grid(
            row=row,
            column=column + 1,
            sticky="ew",
            padx=(0, 12),
            pady=4,
        )

    def choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a music score",
            filetypes=[
                ("Music score files", "*.pdf *.png *.jpg *.jpeg *.tif *.tiff *.webp"),
                ("PDF files", "*.pdf"),
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.webp"),
                ("All files", "*"),
            ],
        )
        if path:
            self.file_var.set(path)

    def choose_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose YOLO model",
            filetypes=[("PyTorch model", "*.pt"), ("All files", "*")],
        )
        if path:
            self.model_var.set(path)

    def choose_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose output folder")
        if path:
            self.output_var.set(path)

    def parse_settings(self) -> EvalSettings:
        file_path = Path(self.file_var.get()).expanduser()
        if not file_path.is_file():
            raise ValueError("Choose an existing PDF or image file.")

        max_pages_text = self.max_pages_var.get().strip()
        max_pages = int(max_pages_text) if max_pages_text else None
        if max_pages is not None and max_pages < 1:
            raise ValueError("Max pages must be blank or at least 1.")

        return EvalSettings(
            file_path=file_path,
            model_path=Path(self.model_var.get()).expanduser(),
            output_dir=Path(self.output_var.get()).expanduser(),
            conf=float(self.conf_var.get()),
            iou=float(self.iou_var.get()),
            imgsz=int(self.imgsz_var.get()),
            pdf_dpi=int(self.pdf_dpi_var.get()),
            max_pages=max_pages,
            device=self.device_var.get().strip() or "cpu",
            download_model=self.download_var.get(),
        )

    def process(self) -> None:
        try:
            settings = self.parse_settings()
        except ValueError as exc:
            messagebox.showerror("Invalid settings", str(exc))
            return
        self.status.configure(text="Processing...")
        self.app.start_processing(settings)


class ViewerFrame(ttk.Frame):
    def __init__(self, app: "OmrLayoutGui") -> None:
        super().__init__(app.root)
        self.app = app
        self.page_index = 0
        self.zoom = 1.0
        self.category_vars: dict[str, tk.BooleanVar] = {}
        self.photo: Any = None
        self.original_image: Any = None

        self._build()

    def _build(self) -> None:
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        toolbar = ttk.Frame(self, padding=8)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(9, weight=1)

        ttk.Button(toolbar, text="Back to setup", command=self.app.show_setup).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Previous", command=self.previous_page).grid(row=0, column=1)
        ttk.Button(toolbar, text="Next", command=self.next_page).grid(row=0, column=2, padx=(4, 12))
        ttk.Button(toolbar, text="Fit", command=self.fit_page).grid(row=0, column=3)
        ttk.Button(toolbar, text="100%", command=self.actual_size).grid(row=0, column=4, padx=(4, 12))
        ttk.Button(toolbar, text="-", width=3, command=lambda: self.change_zoom(0.8)).grid(
            row=0, column=5
        )
        ttk.Button(toolbar, text="+", width=3, command=lambda: self.change_zoom(1.25)).grid(
            row=0, column=6, padx=(4, 12)
        )
        ttk.Button(toolbar, text="Save visible page", command=self.save_visible_page).grid(
            row=0, column=7, padx=(0, 12)
        )
        self.page_label = ttk.Label(toolbar, text="")
        self.page_label.grid(row=0, column=8, sticky="w")

        body = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        body.grid(row=1, column=0, sticky="nsew")

        self.canvas_frame = ttk.Frame(body)
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)
        body.add(self.canvas_frame, weight=4)

        self.canvas = tk.Canvas(self.canvas_frame, background="#202020", highlightthickness=0)
        y_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        x_scroll = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.canvas.bind("<Configure>", lambda _event: self.render())

        side = ttk.Frame(body, padding=12)
        body.add(side, weight=1)
        side.columnconfigure(0, weight=1)

        ttk.Label(side, text="Categories").grid(row=0, column=0, sticky="w")
        self.category_frame = ttk.Frame(side)
        self.category_frame.grid(row=1, column=0, sticky="new", pady=(8, 16))

        self.stats_label = ttk.Label(side, text="", justify=tk.LEFT)
        self.stats_label.grid(row=2, column=0, sticky="new")

        ttk.Label(
            side,
            text="Tip: hide noisy categories first, then inspect systems/staves/measures separately.",
            wraplength=220,
        ).grid(row=3, column=0, sticky="sw", pady=(16, 0))

    def load_results(self) -> None:
        self.page_index = 0
        categories = sorted(
            {
                detection["class_name"]
                for page in self.app.results
                for detection in page.detections
            }
        )
        for child in self.category_frame.winfo_children():
            child.destroy()
        self.category_vars = {}
        for row, category in enumerate(categories):
            var = tk.BooleanVar(value=True)
            self.category_vars[category] = var
            check = ttk.Checkbutton(
                self.category_frame,
                text=category,
                variable=var,
                command=self.render,
            )
            check.grid(row=row, column=0, sticky="w", pady=2)
        self.fit_page()

    def current_page(self) -> PageResult:
        return self.app.results[self.page_index]

    def visible_categories(self) -> set[str]:
        return {
            category
            for category, variable in self.category_vars.items()
            if variable.get()
        }

    def previous_page(self) -> None:
        if self.page_index > 0:
            self.page_index -= 1
            self.render()

    def next_page(self) -> None:
        if self.page_index + 1 < len(self.app.results):
            self.page_index += 1
            self.render()

    def actual_size(self) -> None:
        self.zoom = 1.0
        self.render()

    def fit_page(self) -> None:
        if not self.app.results:
            return
        page = self.current_page()
        canvas_width = max(self.canvas.winfo_width(), 200)
        canvas_height = max(self.canvas.winfo_height(), 200)
        self.zoom = min(canvas_width / page.width, canvas_height / page.height, 1.0)
        self.render()

    def change_zoom(self, factor: float) -> None:
        self.zoom = min(max(self.zoom * factor, 0.1), 4.0)
        self.render()

    def render(self) -> None:
        if not self.app.results:
            return
        Image, ImageDraw, ImageFont = omr_eval.require_pillow()
        page = self.current_page()
        image = Image.open(page.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        visible = self.visible_categories()

        for detection in page.detections:
            if detection["class_name"] not in visible:
                continue
            left, top, right, bottom = detection["bbox"]
            color = class_color(detection["class_name"])
            draw.rectangle((left, top, right, bottom), outline=color, width=4)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            text_bbox = draw.textbbox((left, top), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((left, top), label, fill="#ffffff", font=font)

        scaled_width = max(1, int(image.width * self.zoom))
        scaled_height = max(1, int(image.height * self.zoom))
        image = image.resize((scaled_width, scaled_height))

        from PIL import ImageTk

        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=(0, 0, scaled_width, scaled_height))
        self.page_label.configure(
            text=f"Page {self.page_index + 1} / {len(self.app.results)}  "
            f"Zoom {self.zoom * 100:.0f}%"
        )
        self.update_stats()

    def update_stats(self) -> None:
        page = self.current_page()
        visible = self.visible_categories()
        counts: dict[str, int] = {}
        visible_count = 0
        for detection in page.detections:
            class_name = detection["class_name"]
            counts[class_name] = counts.get(class_name, 0) + 1
            if class_name in visible:
                visible_count += 1

        lines = [
            f"Source: {page.source.name}",
            f"Page: {page.page_index + 1}",
            f"Page size: {page.width} x {page.height}",
            f"Latency: {page.latency_ms:.0f} ms",
            f"Visible boxes: {visible_count}",
            "",
            "Detected:",
        ]
        lines.extend(f"{name}: {count}" for name, count in sorted(counts.items()))
        self.stats_label.configure(text="\n".join(lines))

    def save_visible_page(self) -> None:
        if not self.app.results:
            return
        path = filedialog.asksaveasfilename(
            title="Save visible page",
            defaultextension=".jpg",
            filetypes=[("JPEG image", "*.jpg"), ("PNG image", "*.png"), ("All files", "*")],
        )
        if not path:
            return
        Image, ImageDraw, ImageFont = omr_eval.require_pillow()
        page = self.current_page()
        image = Image.open(page.image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        visible = self.visible_categories()
        for detection in page.detections:
            if detection["class_name"] not in visible:
                continue
            left, top, right, bottom = detection["bbox"]
            color = class_color(detection["class_name"])
            draw.rectangle((left, top, right, bottom), outline=color, width=4)
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
            text_bbox = draw.textbbox((left, top), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((left, top), label, fill="#ffffff", font=font)
        image.save(path)


class OmrLayoutGui:
    def __init__(self) -> None:
        print("Starting OMR Layout Evaluator GUI...", file=sys.stderr, flush=True)
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise SystemExit(
                "No graphical display found. Run this from a desktop terminal, "
                "or set DISPLAY/WAYLAND_DISPLAY."
            )

        self.root = tk.Tk()
        self.root.title("OMR Layout Evaluator")
        self.root.geometry("1200x850")
        self.root.minsize(900, 650)
        self.root.after(250, self.raise_window)
        self.results: list[PageResult] = []
        self.temp_dir: TemporaryDirectory[str] | None = None
        self.worker_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

        self.setup_frame = SetupFrame(self)
        self.viewer_frame = ViewerFrame(self)
        self.setup_frame.pack(fill=tk.BOTH, expand=True)
        print("GUI window created. Close the window to return to the shell.", file=sys.stderr, flush=True)

    def raise_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.root.attributes("-topmost", True)
        self.root.after(500, lambda: self.root.attributes("-topmost", False))

    def show_setup(self) -> None:
        self.viewer_frame.pack_forget()
        self.setup_frame.pack(fill=tk.BOTH, expand=True)

    def show_viewer(self) -> None:
        self.setup_frame.pack_forget()
        self.viewer_frame.pack(fill=tk.BOTH, expand=True)
        self.viewer_frame.load_results()

    def start_processing(self, settings: EvalSettings) -> None:
        thread = threading.Thread(target=self.process_worker, args=(settings,), daemon=True)
        thread.start()
        self.root.after(100, self.poll_worker)

    def poll_worker(self) -> None:
        try:
            event, payload = self.worker_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self.poll_worker)
            return

        if event == "progress":
            self.setup_frame.status.configure(text=payload)
            self.root.after(100, self.poll_worker)
        elif event == "done":
            self.results = payload
            self.setup_frame.status.configure(text="")
            self.show_viewer()
        elif event == "error":
            self.setup_frame.status.configure(text="")
            messagebox.showerror("OMR evaluation failed", str(payload))

    def process_worker(self, settings: EvalSettings) -> None:
        try:
            results = self.run_evaluation(settings)
        except Exception as exc:  # noqa: BLE001 - display GUI errors to the user.
            self.worker_queue.put(("error", exc))
            return
        self.worker_queue.put(("done", results))

    def run_evaluation(self, settings: EvalSettings) -> list[PageResult]:
        YOLO = omr_eval.require_ultralytics()
        omr_eval.require_pillow()
        if settings.file_path.suffix.lower() in omr_eval.PDF_EXTENSIONS:
            omr_eval.require_pdfium()

        if not settings.model_path.exists():
            if settings.download_model:
                self.worker_queue.put(("progress", "Downloading model..."))
                omr_eval.download_model(settings.model_path)
            else:
                raise RuntimeError(f"Model not found: {settings.model_path}")

        settings.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = TemporaryDirectory(prefix="simple-ai-omr-gui-")
        pages = omr_eval.collect_pages(
            [settings.file_path],
            Path(self.temp_dir.name),
            settings.pdf_dpi,
            settings.max_pages,
        )
        if not pages:
            raise RuntimeError("No pages found in the selected file.")

        model = YOLO(str(settings.model_path))
        results: list[PageResult] = []
        for index, page in enumerate(pages, start=1):
            self.worker_queue.put(("progress", f"Processing page {index} / {len(pages)}..."))
            Image, _, _ = omr_eval.require_pillow()
            with Image.open(page.image_path) as image:
                width, height = image.size

            start = time.perf_counter()
            prediction = model.predict(
                source=str(page.image_path),
                conf=settings.conf,
                iou=settings.iou,
                imgsz=settings.imgsz,
                device=settings.device,
                verbose=False,
            )[0]
            latency_ms = (time.perf_counter() - start) * 1000.0
            detections = omr_eval.extract_detections(prediction)
            results.append(
                PageResult(
                    source=page.source,
                    page_index=page.page_index,
                    image_path=page.image_path,
                    width=width,
                    height=height,
                    latency_ms=latency_ms,
                    detections=detections,
                )
            )

        self.write_results(settings, results)
        return results

    def write_results(self, settings: EvalSettings, results: list[PageResult]) -> None:
        serializable = [
            {
                "source": str(page.source),
                "page": page.page_index + 1,
                "width": page.width,
                "height": page.height,
                "latency_ms": page.latency_ms,
                "detections": page.detections,
            }
            for page in results
        ]
        (settings.output_dir / "gui-detections.json").write_text(
            json.dumps(serializable, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def run(self) -> int:
        self.root.mainloop()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
        return 0


def main() -> int:
    return OmrLayoutGui().run()


if __name__ == "__main__":
    raise SystemExit(main())
