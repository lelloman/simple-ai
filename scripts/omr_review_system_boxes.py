#!/usr/bin/env python3
"""Review and correct OMR system boxes in a prepared review directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError as exc:
    raise SystemExit("Tkinter is required for the review UI.") from exc

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Pillow is required. Install with: pip install -r scripts/omr-layout-requirements.txt"
    ) from exc


HANDLE_RADIUS = 12
EDGE_HIT_RADIUS = 10
MIN_BOX_SIZE = 32
NEW_BOX_WIDTH = 320
NEW_BOX_HEIGHT = 160
SELECTION_HIT_PADDING = 16
NUDGE_STEP = 4
FAST_NUDGE_STEP = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review editable OMR system boxes.")
    parser.add_argument("review_dir", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def normalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    left, top, right, bottom = bbox
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    left = clamp(left, 0, width)
    right = clamp(right, 0, width)
    top = clamp(top, 0, height)
    bottom = clamp(bottom, 0, height)
    if right - left < MIN_BOX_SIZE:
        right = clamp(left + MIN_BOX_SIZE, 0, width)
        left = clamp(right - MIN_BOX_SIZE, 0, width)
    if bottom - top < MIN_BOX_SIZE:
        bottom = clamp(top + MIN_BOX_SIZE, 0, height)
        top = clamp(bottom - MIN_BOX_SIZE, 0, height)
    return [left, top, right, bottom]


def point_in_bbox(x: float, y: float, bbox: list[float], padding: float = 0) -> bool:
    left, top, right, bottom = bbox
    return (
        left - padding <= x <= right + padding
        and top - padding <= y <= bottom + padding
    )


def hit_test_box_edge(x: float, y: float, bbox: list[float], radius: float) -> str | None:
    left, top, right, bottom = bbox
    handles = {
        "nw": (left, top),
        "ne": (right, top),
        "sw": (left, bottom),
        "se": (right, bottom),
    }
    for name, (hx, hy) in handles.items():
        if abs(x - hx) <= radius and abs(y - hy) <= radius:
            return name
    inside_y = top - radius <= y <= bottom + radius
    inside_x = left - radius <= x <= right + radius
    if inside_y and abs(x - left) <= radius:
        return "w"
    if inside_y and abs(x - right) <= radius:
        return "e"
    if inside_x and abs(y - top) <= radius:
        return "n"
    if inside_x and abs(y - bottom) <= radius:
        return "s"
    return None


class ReviewApp:
    def __init__(self, root: tk.Tk, review_dir: Path) -> None:
        self.root = root
        self.review_dir = review_dir
        self.manifest_path = review_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise SystemExit(f"Missing manifest: {self.manifest_path}")
        self.manifest = read_json(self.manifest_path)
        self.pages = self.manifest.get("pages") or []
        if not self.pages:
            raise SystemExit("Review manifest contains no pages.")

        self.page_index = 0
        self.page_data: dict[str, Any] = {}
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.rendered_zoom: float | None = None
        self.rendered_image_size: tuple[int, int] | None = None
        self.zoom = 0.35
        self.selected_box: int | None = None
        self.mode = "select"
        self.drag_action: str | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_original_bbox: list[float] | None = None
        self.pan_start: tuple[int, int] | None = None
        self.dirty = False
        self.show_helpers = True

        self.root.title("OMR System Box Review")
        self.build_ui()
        self.bind_shortcuts()
        self.load_page(0)

    def build_ui(self) -> None:
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(toolbar, text="Prev", command=self.previous_page).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Next", command=self.next_page).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Add Box", command=self.enable_add_mode).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Delete", command=self.delete_selected).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Save", command=self.save_page).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Mark Reviewed", command=self.mark_reviewed).pack(side=tk.LEFT)
        tk.Button(toolbar, text="Helpers", command=self.toggle_helpers).pack(side=tk.LEFT)
        tk.Button(toolbar, text="+", command=lambda: self.change_zoom(1.2)).pack(side=tk.LEFT)
        tk.Button(toolbar, text="-", command=lambda: self.change_zoom(1 / 1.2)).pack(side=tk.LEFT)

        self.status = tk.StringVar(value="")
        tk.Label(toolbar, textvariable=self.status, anchor="w").pack(side=tk.LEFT, padx=12)

        body = tk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, background="#222")
        x_scroll = tk.Scrollbar(body, orient=tk.HORIZONTAL, command=self.canvas.xview)
        y_scroll = tk.Scrollbar(body, orient=tk.VERTICAL, command=self.canvas.yview)
        self.canvas.configure(xscrollcommand=x_scroll.set, yscrollcommand=y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

    def bind_shortcuts(self) -> None:
        self.root.bind("<Right>", lambda _event: self.next_page())
        self.root.bind("n", lambda _event: self.next_page())
        self.root.bind("<Left>", lambda _event: self.previous_page())
        self.root.bind("p", lambda _event: self.previous_page())
        self.root.bind("a", lambda _event: self.enable_add_mode())
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("s", lambda _event: self.save_page())
        self.root.bind("r", lambda _event: self.mark_reviewed())
        self.root.bind("<Return>", lambda _event: self.mark_reviewed())
        self.root.bind("f", lambda _event: self.first_unreviewed_page())
        self.root.bind("h", lambda _event: self.toggle_helpers())
        self.root.bind("+", lambda _event: self.change_zoom(1.2))
        self.root.bind("=", lambda _event: self.change_zoom(1.2))
        self.root.bind("-", lambda _event: self.change_zoom(1 / 1.2))
        self.root.bind("<Escape>", lambda _event: self.cancel_add_mode())
        self.root.bind("<Shift-Left>", lambda _event: self.nudge_selected(-FAST_NUDGE_STEP, 0))
        self.root.bind("<Shift-Right>", lambda _event: self.nudge_selected(FAST_NUDGE_STEP, 0))
        self.root.bind("<Shift-Up>", lambda _event: self.nudge_selected(0, -FAST_NUDGE_STEP))
        self.root.bind("<Shift-Down>", lambda _event: self.nudge_selected(0, FAST_NUDGE_STEP))
        self.root.bind("<Control-Left>", lambda _event: self.resize_selected("w", -FAST_NUDGE_STEP))
        self.root.bind("<Control-Right>", lambda _event: self.resize_selected("e", FAST_NUDGE_STEP))
        self.root.bind("<Control-Up>", lambda _event: self.resize_selected("n", -FAST_NUDGE_STEP))
        self.root.bind("<Control-Down>", lambda _event: self.resize_selected("s", FAST_NUDGE_STEP))

    def page_entry(self) -> dict[str, Any]:
        return self.pages[self.page_index]

    def review_path(self) -> Path:
        return self.review_dir / self.page_entry()["review"]

    def image_path(self) -> Path:
        return self.review_dir / self.page_entry()["image"]

    def load_page(self, index: int) -> None:
        if self.dirty:
            self.save_page()
        self.page_index = index
        self.page_data = read_json(self.review_path())
        self.image = Image.open(self.image_path()).convert("RGB")
        self.photo = None
        self.rendered_zoom = None
        self.rendered_image_size = None
        self.selected_box = None
        self.mode = "select"
        self.dirty = False
        self.render()

    def render(self) -> None:
        self.render_base_image()
        self.render_overlays()
        self.update_status()

    def render_base_image(self) -> None:
        assert self.image is not None
        image_size = (self.image.width, self.image.height)
        if (
            self.photo is not None
            and self.rendered_zoom == self.zoom
            and self.rendered_image_size == image_size
        ):
            return

        width = max(1, int(image_size[0] * self.zoom))
        height = max(1, int(image_size[1] * self.zoom))
        resample = Image.Resampling.BOX if self.zoom < 1.0 else Image.Resampling.BICUBIC
        resized = self.image.resize((width, height), resample)
        self.photo = ImageTk.PhotoImage(resized)
        self.rendered_zoom = self.zoom
        self.rendered_image_size = image_size

        self.canvas.delete("base")
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW, tags=("base",))
        self.canvas.tag_lower("base")
        self.canvas.configure(scrollregion=(0, 0, width, height))

    def render_overlays(self) -> None:
        self.canvas.delete("overlay")
        if self.show_helpers:
            for helper in self.page_data.get("helpers", []):
                self.draw_box(helper["bbox"], "#9a9a9a", helper["class"], dash=(4, 4), width=2)

        for index, box in enumerate(self.page_data.get("boxes", [])):
            selected = index == self.selected_box
            color = (
                "#2f80ed"
                if selected
                else "#2ca25f" if box.get("status") == "accepted" else "#f28e2b"
            )
            self.draw_box(
                box["bbox"],
                color,
                f"{index + 1}: {box['class']}",
                width=4 if selected else 3,
            )
            if selected:
                self.draw_handles(box["bbox"], color)

    def draw_box(
        self,
        bbox: list[float],
        color: str,
        label: str,
        dash: tuple[int, int] | None = None,
        width: int = 3,
    ) -> None:
        left, top, right, bottom = [value * self.zoom for value in bbox]
        self.canvas.create_rectangle(
            left,
            top,
            right,
            bottom,
            outline=color,
            width=width,
            dash=dash,
            tags=("overlay",),
        )
        text_id = self.canvas.create_text(
            left + 4,
            top + 4,
            text=label,
            fill="white",
            anchor=tk.NW,
            tags=("overlay",),
        )
        text_bbox = self.canvas.bbox(text_id)
        if text_bbox:
            rect_id = self.canvas.create_rectangle(
                text_bbox,
                fill=color,
                outline=color,
                tags=("overlay",),
            )
            self.canvas.tag_lower(rect_id, text_id)

    def draw_handles(self, bbox: list[float], color: str) -> None:
        left, top, right, bottom = [value * self.zoom for value in bbox]
        radius = HANDLE_RADIUS
        for x, y in [
            (left, top),
            ((left + right) / 2, top),
            (right, top),
            (left, (top + bottom) / 2),
            (right, (top + bottom) / 2),
            (left, bottom),
            ((left + right) / 2, bottom),
            (right, bottom),
        ]:
            self.canvas.create_rectangle(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline="white",
                tags=("overlay",),
            )

    def update_status(self) -> None:
        page = self.page_data
        dirty = "modified" if self.dirty else "saved"
        mode = "add" if self.mode == "add" else "select"
        self.status.set(
            f"Page {self.page_index + 1}/{len(self.pages)} "
            f"(source page {page.get('page')}) | "
            f"{len(page.get('boxes', []))} boxes | "
            f"{page.get('status', 'pending')} | {dirty} | mode: {mode} | "
            "a add, s save, r reviewed, f first pending | edges/corners resize, wheel zoom, right/middle drag pan"
        )

    def canvas_to_image(self, event: tk.Event) -> tuple[float, float]:
        x = self.canvas.canvasx(event.x) / self.zoom
        y = self.canvas.canvasy(event.y) / self.zoom
        return x, y

    def select_box_at(self, x: float, y: float) -> int | None:
        boxes = self.page_data.get("boxes", [])
        for index in range(len(boxes) - 1, -1, -1):
            padding = SELECTION_HIT_PADDING / self.zoom
            if point_in_bbox(x, y, boxes[index]["bbox"], padding):
                return index
        return None

    def selected_edge_at(self, x: float, y: float) -> str | None:
        if self.selected_box is None:
            return None
        bbox = self.page_data["boxes"][self.selected_box]["bbox"]
        return hit_test_box_edge(x, y, bbox, EDGE_HIT_RADIUS / self.zoom)

    def cursor_for_action(self, action: str | None) -> str:
        return {
            "n": "sb_v_double_arrow",
            "s": "sb_v_double_arrow",
            "e": "sb_h_double_arrow",
            "w": "sb_h_double_arrow",
            "ne": "top_right_corner",
            "sw": "bottom_left_corner",
            "nw": "top_left_corner",
            "se": "bottom_right_corner",
            "move": "fleur",
            "add": "crosshair",
        }.get(action, "arrow")

    def on_mouse_move(self, event: tk.Event) -> None:
        if self.drag_action:
            return
        if self.mode == "add":
            self.canvas.configure(cursor="crosshair")
            return
        x, y = self.canvas_to_image(event)
        edge = self.selected_edge_at(x, y)
        if edge:
            self.canvas.configure(cursor=self.cursor_for_action(edge))
            return
        selected = self.select_box_at(x, y)
        self.canvas.configure(cursor="fleur" if selected is not None else "arrow")

    def on_mouse_down(self, event: tk.Event) -> None:
        x, y = self.canvas_to_image(event)
        if self.mode == "add":
            self.drag_action = "add"
            self.drag_start = (x, y)
            self.drag_original_bbox = [x, y, x + NEW_BOX_WIDTH, y + NEW_BOX_HEIGHT]
            bbox = normalize_bbox(
                [x, y, x + NEW_BOX_WIDTH, y + NEW_BOX_HEIGHT],
                int(self.page_data["width"]),
                int(self.page_data["height"]),
            )
            self.page_data.setdefault("boxes", []).append(
                {
                    "id": self.next_box_id(),
                    "class": "system",
                    "bbox": bbox,
                    "source": "human",
                    "confidence": None,
                    "status": "accepted",
                }
            )
            self.selected_box = len(self.page_data["boxes"]) - 1
            self.dirty = True
            self.render()
            return

        selected = self.select_box_at(x, y)
        self.selected_box = selected
        self.drag_action = None
        self.drag_start = None
        self.drag_original_bbox = None
        if selected is not None:
            bbox = self.page_data["boxes"][selected]["bbox"]
            handle = hit_test_box_edge(x, y, bbox, EDGE_HIT_RADIUS / self.zoom)
            self.drag_action = handle or "move"
            self.drag_start = (x, y)
            self.drag_original_bbox = list(bbox)
            self.canvas.configure(cursor=self.cursor_for_action(self.drag_action))
        self.render()

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.selected_box is None or self.drag_action is None or self.drag_start is None:
            return
        x, y = self.canvas_to_image(event)
        boxes = self.page_data.get("boxes", [])
        box = boxes[self.selected_box]
        original = self.drag_original_bbox or list(box["bbox"])
        left, top, right, bottom = original
        start_x, start_y = self.drag_start

        if self.drag_action == "move":
            dx = x - start_x
            dy = y - start_y
            bbox = [left + dx, top + dy, right + dx, bottom + dy]
        elif self.drag_action == "add":
            bbox = [start_x, start_y, x, y]
        else:
            bbox = [left, top, right, bottom]
            if "n" in self.drag_action:
                bbox[1] = y
            if "s" in self.drag_action:
                bbox[3] = y
            if "w" in self.drag_action:
                bbox[0] = x
            if "e" in self.drag_action:
                bbox[2] = x

        box["bbox"] = normalize_bbox(
            bbox,
            int(self.page_data["width"]),
            int(self.page_data["height"]),
        )
        box["status"] = "accepted"
        if box.get("source") == "model":
            box["source"] = "human-edited"
        self.page_data["status"] = "edited"
        self.pages[self.page_index]["status"] = "edited"
        self.dirty = True
        self.render()

    def on_mouse_up(self, _event: tk.Event) -> None:
        self.drag_action = None
        self.drag_start = None
        self.drag_original_bbox = None
        if self.mode == "add":
            self.mode = "select"
            self.render()
        self.canvas.configure(cursor="arrow")

    def on_pan_start(self, event: tk.Event) -> None:
        self.pan_start = (event.x, event.y)
        self.canvas.scan_mark(event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def on_pan_drag(self, event: tk.Event) -> None:
        if self.pan_start is None:
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mouse_wheel(self, event: tk.Event) -> None:
        if hasattr(event, "num") and event.num == 4:
            factor = 1.15
        elif hasattr(event, "num") and event.num == 5:
            factor = 1 / 1.15
        elif getattr(event, "delta", 0) > 0:
            factor = 1.15
        else:
            factor = 1 / 1.15
        self.change_zoom(factor)

    def mark_selected_changed(self) -> None:
        if self.selected_box is None:
            return
        box = self.page_data["boxes"][self.selected_box]
        box["status"] = "accepted"
        if box.get("source") in {"model", "derived-grand-staff"}:
            box["source"] = "human-edited"
        self.page_data["status"] = "edited"
        self.pages[self.page_index]["status"] = "edited"
        self.dirty = True

    def nudge_selected(self, dx: float, dy: float) -> None:
        if self.selected_box is None:
            return
        box = self.page_data["boxes"][self.selected_box]
        left, top, right, bottom = box["bbox"]
        box["bbox"] = normalize_bbox(
            [left + dx, top + dy, right + dx, bottom + dy],
            int(self.page_data["width"]),
            int(self.page_data["height"]),
        )
        self.mark_selected_changed()
        self.render()

    def resize_selected(self, edge: str, delta: float) -> None:
        if self.selected_box is None:
            return
        box = self.page_data["boxes"][self.selected_box]
        bbox = list(box["bbox"])
        if edge == "w":
            bbox[0] += delta
        elif edge == "e":
            bbox[2] += delta
        elif edge == "n":
            bbox[1] += delta
        elif edge == "s":
            bbox[3] += delta
        box["bbox"] = normalize_bbox(
            bbox,
            int(self.page_data["width"]),
            int(self.page_data["height"]),
        )
        self.mark_selected_changed()
        self.render()

    def next_box_id(self) -> str:
        page_number = int(self.page_data.get("page", self.page_index + 1))
        existing = len(self.page_data.get("boxes", [])) + 1
        return f"p{page_number:04d}-box-{existing:04d}"

    def enable_add_mode(self) -> None:
        self.mode = "add"
        self.update_status()

    def cancel_add_mode(self) -> None:
        self.mode = "select"
        self.update_status()

    def delete_selected(self) -> None:
        if self.selected_box is None:
            return
        del self.page_data["boxes"][self.selected_box]
        self.selected_box = None
        self.page_data["status"] = "edited"
        self.pages[self.page_index]["status"] = "edited"
        self.dirty = True
        self.render()

    def save_page(self) -> None:
        write_json(self.review_path(), self.page_data)
        write_json(self.manifest_path, self.manifest)
        self.dirty = False
        self.update_status()

    def mark_reviewed(self) -> None:
        self.page_data["status"] = "reviewed"
        for box in self.page_data.get("boxes", []):
            box["status"] = "accepted"
        self.pages[self.page_index]["status"] = "reviewed"
        self.dirty = True
        self.save_page()
        if self.page_index + 1 < len(self.pages):
            self.load_page(self.page_index + 1)

    def next_page(self) -> None:
        if self.page_index + 1 < len(self.pages):
            self.load_page(self.page_index + 1)

    def previous_page(self) -> None:
        if self.page_index > 0:
            self.load_page(self.page_index - 1)

    def first_unreviewed_page(self) -> None:
        for index, page in enumerate(self.pages):
            if page.get("status") != "reviewed":
                self.load_page(index)
                return

    def toggle_helpers(self) -> None:
        self.show_helpers = not self.show_helpers
        self.render()

    def change_zoom(self, factor: float) -> None:
        self.zoom = clamp(self.zoom * factor, 0.05, 4.0)
        self.render()

    def on_close(self) -> None:
        if self.dirty:
            self.save_page()
        self.root.destroy()


def main() -> int:
    args = parse_args()
    review_dir = args.review_dir
    if not review_dir.exists():
        raise SystemExit(f"Missing review directory: {review_dir}")
    root = tk.Tk()
    try:
        app = ReviewApp(root, review_dir)
    except Exception as exc:
        messagebox.showerror("OMR review", str(exc))
        raise
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.geometry("1400x900")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
