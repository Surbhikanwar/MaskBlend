import os
from dataclasses import dataclass
from typing import Optional, Tuple

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image
from tkinter import filedialog, messagebox


PREVIEW_MAX_SIZE = 460
DEFAULT_OUTPUT_NAME = "pattern_transfer_result.png"


@dataclass
class TransferData:
    clean_bgr_float: np.ndarray
    clean_l: np.ndarray
    clean_a: np.ndarray
    clean_b: np.ndarray
    textured_l: np.ndarray
    textured_l_enhanced: np.ndarray
    textured_l_detail: np.ndarray
    textured_a_detail: np.ndarray
    textured_b_detail: np.ndarray
    mask_alpha: np.ndarray


def load_color_image(path: str, label: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load {label}: {path}")
    return image


def load_mask_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load mask image: {path}")
    return image


def fit_texture(texture_bgr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = target_size
    if texture_bgr.shape[:2] == (target_h, target_w):
        return texture_bgr
    return cv2.resize(texture_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def build_mask(mask_gray: np.ndarray, target_size: Tuple[int, int], threshold: int = 127, feather: float = 2.0) -> np.ndarray:
    target_h, target_w = target_size
    if mask_gray.shape[:2] != (target_h, target_w):
        mask_gray = cv2.resize(mask_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    mask = (mask_gray >= threshold).astype(np.float32)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather, sigmaY=feather)
    return np.clip(mask, 0.0, 1.0)[..., None]


def extract_detail(channel: np.ndarray, sigma: float = 11.0) -> np.ndarray:
    smooth = cv2.GaussianBlur(channel, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return channel - smooth


def enhance_luminance(channel: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(channel.astype(np.uint8)).astype(np.float32)


def prepare_transfer(clean_bgr: np.ndarray, textured_bgr: np.ndarray, mask_gray: np.ndarray) -> TransferData:
    textured_bgr = fit_texture(textured_bgr, clean_bgr.shape[:2])
    mask_alpha = build_mask(mask_gray, clean_bgr.shape[:2])

    clean_lab = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    textured_lab = cv2.cvtColor(textured_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    clean_l, clean_a, clean_b = cv2.split(clean_lab)
    textured_l, textured_a, textured_b = cv2.split(textured_lab)

    return TransferData(
        clean_bgr_float=clean_bgr.astype(np.float32),
        clean_l=clean_l,
        clean_a=clean_a,
        clean_b=clean_b,
        textured_l=textured_l,
        textured_l_enhanced=enhance_luminance(textured_l),
        textured_l_detail=extract_detail(textured_l),
        textured_a_detail=extract_detail(textured_a),
        textured_b_detail=extract_detail(textured_b),
        mask_alpha=mask_alpha,
    )


def render_transfer(
    data: TransferData,
    strength: float,
    clean_color_scale: float = 1.0,
    dark_region_scale: float = 1.0,
    color_strength: float = 0.20,
) -> np.ndarray:
    strength_weight = np.clip(strength, 0.0, 1.0)
    effective_clean_color_scale = 1.0 - ((1.0 - clean_color_scale) * strength_weight)
    effective_dark_region_scale = 1.0 - ((1.0 - dark_region_scale) * strength_weight)

    neutral_ab = 128.0
    clean_chroma_a = data.clean_a - neutral_ab
    clean_chroma_b = data.clean_b - neutral_ab
    clean_chroma = np.sqrt((clean_chroma_a ** 2) + (clean_chroma_b ** 2))
    chroma_weight = np.clip(clean_chroma / 55.0, 0.0, 1.0)
    fade_amount = 1.0 - effective_clean_color_scale

    # Base-color-aware pigment aging: vivid colors keep a softer version of their
    # original hue, while weak colors wash out more like old plaster.
    residual_color = 0.10 + (0.62 * chroma_weight) + (0.16 * (1.0 - (data.clean_l / 255.0)))
    aged_color_scale = effective_clean_color_scale + (fade_amount * residual_color)
    color_lift = fade_amount * (0.30 + (0.55 * (1.0 - (data.clean_l / 255.0)))) * (1.0 - (0.25 * chroma_weight))
    aged_clean_l = data.clean_l + ((220.0 - data.clean_l) * color_lift)
    faded_clean_a = neutral_ab + (clean_chroma_a * aged_color_scale)
    faded_clean_b = neutral_ab + (clean_chroma_b * aged_color_scale)

    shadow_reveal = 1.0 - effective_dark_region_scale
    darkness_weight = np.power(1.0 - (aged_clean_l / 255.0), 0.95)
    shadow_blend = np.clip(darkness_weight * shadow_reveal * (0.65 + (0.35 * strength_weight)), 0.0, 1.0)
    lifted_clean_l = aged_clean_l + ((235.0 - aged_clean_l) * shadow_blend)
    texture_l_mix = np.clip(shadow_blend * 0.9, 0.0, 0.9)
    detail_gain = 1.0 + (shadow_blend * 4.5)
    base_l = (lifted_clean_l * (1.0 - texture_l_mix)) + (data.textured_l_enhanced * texture_l_mix)

    out_l = np.clip(base_l + (data.textured_l_detail * strength * detail_gain), 0, 255)
    out_a = np.clip(faded_clean_a + (data.textured_a_detail * strength * color_strength), 0, 255)
    out_b = np.clip(faded_clean_b + (data.textured_b_detail * strength * color_strength), 0, 255)

    merged_lab = cv2.merge([out_l, out_a, out_b]).astype(np.uint8)
    textured_result = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    result = data.clean_bgr_float * (1.0 - data.mask_alpha) + textured_result * data.mask_alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def resize_for_preview(image_bgr: np.ndarray, max_size: int = PREVIEW_MAX_SIZE) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    scale = min(max_size / max(height, width), 1.0)
    if scale == 1.0:
        return image_bgr.copy()

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)


def bgr_to_ctk_image(image_bgr: np.ndarray) -> ctk.CTkImage:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    return ctk.CTkImage(
        light_image=pil_image,
        dark_image=pil_image,
        size=(image_bgr.shape[1], image_bgr.shape[0]),
    )


class PatternTransferApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        self.title("Pattern Transfer")
        self.geometry("1000x640")
        self.minsize(860, 540)

        self.clean_path_var = ctk.StringVar()
        self.textured_path_var = ctk.StringVar()
        self.mask_path_var = ctk.StringVar()
        self.strength_var = ctk.DoubleVar(value=0.85)
        self.clean_color_scale_var = ctk.DoubleVar(value=1.0)
        self.dark_region_scale_var = ctk.DoubleVar(value=0.30)
        self.status_var = ctk.StringVar(value="Select clean, textured, and mask images.")

        self.full_clean: Optional[np.ndarray] = None
        self.full_textured: Optional[np.ndarray] = None
        self.full_mask: Optional[np.ndarray] = None
        self.preview_data: Optional[TransferData] = None
        self.preview_result: Optional[np.ndarray] = None
        self.clean_preview_photo = None
        self.result_preview_photo = None
        self.after_id: Optional[str] = None

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_columnconfigure(0, weight=1)
        sidebar.grid_rowconfigure(10, weight=1)

        main = ctk.CTkFrame(self, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure((0, 1), weight=1)
        main.grid_rowconfigure(1, weight=1)

        title = ctk.CTkLabel(sidebar, text="Pattern Transfer", font=ctk.CTkFont(size=20, weight="bold"))
        title.grid(row=0, column=0, padx=16, pady=(16, 4), sticky="w")

        subtitle = ctk.CTkLabel(
            sidebar,
            text="Give the app three images:\n1. Clean image\n2. Textured image\n3. Mask image",
            justify="left",
            text_color=("gray25", "gray75"),
        )
        subtitle.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="w")

        self._add_file_row(sidebar, 2, "Clean Image", self.clean_path_var)
        self._add_file_row(sidebar, 4, "Textured Image", self.textured_path_var)
        self._add_file_row(sidebar, 6, "Mask Image", self.mask_path_var)

        actions = ctk.CTkFrame(sidebar, fg_color="transparent")
        actions.grid(row=8, column=0, padx=16, pady=(10, 10), sticky="ew")
        actions.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(actions, text="Load Preview", command=self.load_preview).grid(
            row=0, column=0, padx=(0, 6), sticky="ew"
        )
        ctk.CTkButton(actions, text="Save Result", command=self.save_result).grid(
            row=0, column=1, padx=(6, 0), sticky="ew"
        )

        slider_card = ctk.CTkFrame(sidebar)
        slider_card.grid(row=9, column=0, padx=16, pady=(0, 12), sticky="ew")
        slider_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(slider_card, text="Pattern Strength", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=0, column=0, padx=12, pady=(12, 4), sticky="w"
        )

        self.strength_value_label = ctk.CTkLabel(
            slider_card,
            text=f"{self.strength_var.get():.2f}",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.strength_value_label.grid(row=1, column=0, padx=12, pady=(0, 3), sticky="w")

        strength_slider = ctk.CTkSlider(
            slider_card,
            from_=0.0,
            to=1.0,
            number_of_steps=100,
            variable=self.strength_var,
            command=self.on_strength_change,
        )
        strength_slider.grid(row=2, column=0, padx=12, pady=(4, 8), sticky="ew")

        ctk.CTkLabel(
            slider_card,
            text="Move the slider to adjust the transfer in real time.",
            justify="left",
            wraplength=250,
            text_color=("gray25", "gray75"),
        ).grid(row=3, column=0, padx=12, pady=(0, 8), sticky="w")

        ctk.CTkLabel(slider_card, text="Clean Color Fade", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=4, column=0, padx=12, pady=(2, 4), sticky="w"
        )

        self.clean_color_scale_value_label = ctk.CTkLabel(
            slider_card,
            text=f"{self.clean_color_scale_var.get():.2f}",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.clean_color_scale_value_label.grid(row=5, column=0, padx=12, pady=(0, 3), sticky="w")

        clean_color_scale_slider = ctk.CTkSlider(
            slider_card,
            from_=1.0,
            to=0.0,
            number_of_steps=100,
            variable=self.clean_color_scale_var,
            command=self.on_clean_color_scale_change,
        )
        clean_color_scale_slider.grid(row=6, column=0, padx=12, pady=(4, 8), sticky="ew")

        ctk.CTkLabel(
            slider_card,
            text="Lower values fade masked color more, and the fade follows pattern strength.",
            justify="left",
            wraplength=250,
            text_color=("gray25", "gray75"),
        ).grid(row=7, column=0, padx=12, pady=(0, 12), sticky="w")

        ctk.CTkLabel(slider_card, text="Dark Region Fade", font=ctk.CTkFont(size=15, weight="bold")).grid(
            row=8, column=0, padx=12, pady=(2, 4), sticky="w"
        )

        self.dark_region_scale_value_label = ctk.CTkLabel(
            slider_card,
            text=f"{self.dark_region_scale_var.get():.2f}",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.dark_region_scale_value_label.grid(row=9, column=0, padx=12, pady=(0, 3), sticky="w")

        dark_region_scale_slider = ctk.CTkSlider(
            slider_card,
            from_=1.0,
            to=0.0,
            number_of_steps=100,
            variable=self.dark_region_scale_var,
            command=self.on_dark_region_scale_change,
        )
        dark_region_scale_slider.grid(row=10, column=0, padx=12, pady=(4, 8), sticky="ew")

        ctk.CTkLabel(
            slider_card,
            text="Lower values reveal more texture in darker masked regions, and it follows pattern strength.",
            justify="left",
            wraplength=250,
            text_color=("gray25", "gray75"),
        ).grid(row=11, column=0, padx=12, pady=(0, 12), sticky="w")

        ctk.CTkLabel(
            sidebar,
            textvariable=self.status_var,
            justify="left",
            wraplength=270,
            text_color=("gray25", "gray75"),
        ).grid(row=10, column=0, padx=16, pady=(4, 16), sticky="sw")

        ctk.CTkLabel(main, text="Preview", font=ctk.CTkFont(size=22, weight="bold")).grid(
            row=0, column=0, columnspan=2, padx=18, pady=(18, 10), sticky="w"
        )

        self.clean_preview_label = self._build_preview_panel(main, 1, 0, "Clean Preview")
        self.result_preview_label = self._build_preview_panel(main, 1, 1, "Result Preview")

    def _add_file_row(self, parent, row: int, title: str, variable: ctk.StringVar) -> None:
        ctk.CTkLabel(parent, text=title, font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=row, column=0, padx=16, pady=(6, 4), sticky="w"
        )

        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid(row=row + 1, column=0, padx=16, pady=(0, 2), sticky="ew")
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkEntry(frame, textvariable=variable).grid(row=0, column=0, padx=(0, 8), sticky="ew")
        ctk.CTkButton(
            frame,
            text="Browse",
            width=84,
            command=lambda v=variable, t=title: self.pick_image(v, t),
        ).grid(row=0, column=1, sticky="e")

    def _build_preview_panel(self, parent, row: int, column: int, title: str):
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=column, padx=22, pady=(0, 22), sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, padx=18, pady=(16, 10), sticky="w"
        )

        preview_label = ctk.CTkLabel(
            frame,
            text="No image loaded",
            fg_color=("gray90", "gray16"),
            corner_radius=16,
        )
        preview_label.grid(row=1, column=0, padx=16, pady=(0, 16), sticky="nsew")
        return preview_label

    def pick_image(self, variable: ctk.StringVar, title: str) -> None:
        selected = filedialog.askopenfilename(
            title=f"Select {title}",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp")],
        )
        if selected:
            variable.set(selected)
            self.status_var.set(f"Selected {title}: {os.path.basename(selected)}")

    def on_strength_change(self, value) -> None:
        self.strength_value_label.configure(text=f"{float(value):.2f}")
        self.schedule_preview_update()

    def on_clean_color_scale_change(self, value) -> None:
        self.clean_color_scale_value_label.configure(text=f"{float(value):.2f}")
        self.schedule_preview_update()

    def on_dark_region_scale_change(self, value) -> None:
        self.dark_region_scale_value_label.configure(text=f"{float(value):.2f}")
        self.schedule_preview_update()

    def schedule_preview_update(self) -> None:
        if self.preview_data is None:
            return

        if self.after_id is not None:
            self.after_cancel(self.after_id)
        self.after_id = self.after(40, self.update_result_preview)

    def load_preview(self) -> None:
        try:
            clean_path = self.clean_path_var.get().strip()
            textured_path = self.textured_path_var.get().strip()
            mask_path = self.mask_path_var.get().strip()

            if not clean_path or not textured_path or not mask_path:
                raise ValueError("Please select all three images: clean, textured, and mask.")

            self.full_clean = load_color_image(clean_path, "clean image")
            self.full_textured = load_color_image(textured_path, "textured image")
            self.full_mask = load_mask_image(mask_path)

            clean_preview = resize_for_preview(self.full_clean)
            self.preview_data = prepare_transfer(clean_preview, self.full_textured, self.full_mask)

            self.clean_preview_photo = bgr_to_ctk_image(clean_preview)
            self.clean_preview_label.configure(image=self.clean_preview_photo, text="")

            self.update_result_preview()
            self.status_var.set("Preview loaded successfully.")
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Error", str(exc))

    def update_result_preview(self) -> None:
        self.after_id = None
        if self.preview_data is None:
            return

        self.preview_result = render_transfer(
            self.preview_data,
            self.strength_var.get(),
            self.clean_color_scale_var.get(),
            self.dark_region_scale_var.get(),
        )
        self.result_preview_photo = bgr_to_ctk_image(self.preview_result)
        self.result_preview_label.configure(image=self.result_preview_photo, text="")

    def save_result(self) -> None:
        try:
            if self.full_clean is None or self.full_textured is None or self.full_mask is None:
                self.load_preview()
                if self.full_clean is None or self.full_textured is None or self.full_mask is None:
                    return

            output_path = filedialog.asksaveasfilename(
                title="Save Result",
                defaultextension=".png",
                initialfile=DEFAULT_OUTPUT_NAME,
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg"), ("All Files", "*.*")],
            )
            if not output_path:
                return

            full_data = prepare_transfer(self.full_clean, self.full_textured, self.full_mask)
            final_result = render_transfer(
                full_data,
                self.strength_var.get(),
                self.clean_color_scale_var.get(),
                self.dark_region_scale_var.get(),
            )

            if not cv2.imwrite(output_path, final_result):
                raise IOError(f"Could not save image: {output_path}")

            self.status_var.set(f"Saved result to {output_path}")
            messagebox.showinfo("Saved", f"Result saved to:\n{output_path}")
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Error", str(exc))


def main() -> None:
    app = PatternTransferApp()
    app.mainloop()


if __name__ == "__main__":
    main()
