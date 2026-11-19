
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import json

BASE_DIR = Path("dataset")
IMAGE_DIR = BASE_DIR / "images"
MASK_DIR = BASE_DIR / "masks"

SPLITS = ["train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

class AnnotationTool:
    def __init__(self, root):
        self.root = root
        self.root.title("MEASURE IT - Polygon Annotation Tool")
        self.root.geometry("1200x850")

        self.split_index = 0
        self.image_index = 0
        self.images = []
        self.points = []
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.current_image = None
        self.tk_image = None
        self.canvas_image_id = None

        self.canvas = tk.Canvas(root, bg="#333333", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(root)
        controls.pack(fill=tk.X, padx=8, pady=8)

        tk.Button(
            controls, text="Undo point", command=self.undo_point
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            controls, text="Clear outline", command=self.clear_points
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            controls, text="Save mask", command=self.save_mask
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            controls, text="Next image", command=self.next_image
        ).pack(side=tk.LEFT, padx=4)

        self.status = tk.Label(
            root, text="Starting...", anchor="w"
        )
        self.status.pack(fill=tk.X, padx=8, pady=5)

        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Configure>", self.on_resize)

        self.load_images()

    def load_images(self):
        split = SPLITS[self.split_index]
        folder = IMAGE_DIR / split

        self.images = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not self.images:
            self.status.config(text=f"No images found in {folder}")
            messagebox.showerror("No images", f"No images found in {folder}")
            return

        self.image_index = 0
        self.load_current_image()

    def load_current_image(self):
        self.points = []
        image_path = self.images[self.image_index]
        self.current_image = Image.open(image_path).convert("RGB")

        self.display_image()

        self.update_status()

    def display_image(self):
        if self.current_image is None:
            return

        canvas_width = max(self.canvas.winfo_width(), 400)
        canvas_height = max(self.canvas.winfo_height(), 300)

        image_width, image_height = self.current_image.size

        self.display_scale = min(
            canvas_width / image_width,
            canvas_height / image_height,
            1.0
        )

        new_width = int(image_width * self.display_scale)
        new_height = int(image_height * self.display_scale)

        preview = self.current_image.resize(
            (new_width, new_height), Image.Resampling.LANCZOS
        )

        self.tk_image = ImageTk.PhotoImage(preview)

        offset_x = (canvas_width - new_width) // 2
        offset_y = (canvas_height - new_height) // 2
        self.display_offset = (offset_x, offset_y)

        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(
            offset_x, offset_y, anchor=tk.NW, image=self.tk_image
        )

        self.redraw_points()

    def on_resize(self, event):
        if self.current_image is not None:
            self.display_image()

    def add_point(self, event):
        if self.current_image is None:
            return

        offset_x, offset_y = self.display_offset
        image_width, image_height = self.current_image.size

        x = (event.x - offset_x) / self.display_scale
        y = (event.y - offset_y) / self.display_scale

        if not (0 <= x < image_width and 0 <= y < image_height):
            return

        self.points.append((x, y))
        self.redraw_points()
        self.update_status()

    def redraw_points(self):
        self.canvas.delete("annotation")

        offset_x, offset_y = self.display_offset
        scale = self.display_scale

        display_points = [
            (offset_x + x * scale, offset_y + y * scale)
            for x, y in self.points
        ]

        for x, y in display_points:
            r = 4
            self.canvas.create_oval(
                x-r, y-r, x+r, y+r,
                fill="red", outline="white", tags="annotation"
            )

        if len(display_points) >= 2:
            for i in range(len(display_points) - 1):
                self.canvas.create_line(
                    *display_points[i],
                    *display_points[i+1],
                    fill="yellow",
                    width=2,
                    tags="annotation"
                )

        if len(display_points) >= 3:
            self.canvas.create_line(
                *display_points[-1],
                *display_points[0],
                fill="cyan",
                width=1,
                dash=(4, 3),
                tags="annotation"
            )

    def undo_point(self):
        if self.points:
            self.points.pop()
            self.redraw_points()
            self.update_status()

    def clear_points(self):
        self.points = []
        self.redraw_points()
        self.update_status()

    def save_mask(self):
        if len(self.points) < 3:
            messagebox.showwarning(
                "Incomplete outline",
                "Add at least 3 points around the object."
            )
            return

        image_path = self.images[self.image_index]
        split = SPLITS[self.split_index]

        mask = Image.new("L", self.current_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.polygon(self.points, fill=255)

        output_dir = MASK_DIR / split
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / (image_path.stem + ".png")
        mask.save(output_path)

        messagebox.showinfo(
            "Mask saved",
            f"Saved mask:\n{output_path}"
        )

        self.next_image()

    def next_image(self):
        if self.image_index + 1 < len(self.images):
            self.image_index += 1
            self.load_current_image()
            return

        if self.split_index + 1 < len(SPLITS):
            self.split_index += 1
            self.load_images()
            return

        messagebox.showinfo(
            "Completed",
            "All dataset images have been processed."
        )

    def update_status(self):
        if not self.images:
            return

        image_path = self.images[self.image_index]
        self.status.config(
            text=(
                f"Split: {SPLITS[self.split_index]} | "
                f"Image: {self.image_index + 1}/{len(self.images)} | "
                f"File: {image_path.name} | "
                f"Points: {len(self.points)}"
            )
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotationTool(root)
    root.mainloop()
