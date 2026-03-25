# MaskBlend

A desktop GUI application that blends texture patterns onto images using LAB color space analysis. Load a clean image, a textured image, and a mask — then tune three sliders to control how the texture is applied, previewing changes in real time before saving the full-resolution result.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green) ![CustomTkinter](https://img.shields.io/badge/CustomTkinter-5.x-purple)

---

## Features

- **Real-time preview** — slider adjustments are reflected instantly on a live preview panel
- **LAB color space processing** — texture blending happens in LAB space for perceptually accurate results
- **CLAHE luminance enhancement** — adaptive histogram equalization sharpens texture detail before blending
- **Feathered masking** — Gaussian-blurred mask edges for smooth, natural-looking transitions
- **Three-way control:**
  - **Pattern Strength** — overall intensity of the texture transfer
  - **Clean Color Fade** — how much the original image's color desaturates under the texture
  - **Dark Region Fade** — how much texture is revealed in darker masked areas
- **Full-resolution save** — preview runs on a downscaled copy for speed; saving always uses the original resolution
- **System theme support** — automatically follows light/dark OS appearance via CustomTkinter

---

## How It Works

1. The clean and textured images are converted to LAB color space.
2. The mask image determines where the texture is applied (white = masked region).
3. Luminance detail is extracted from the texture using high-frequency separation.
4. `render_transfer` blends the texture's luminance detail and color detail onto the clean image, respecting chroma weight and shadow depth.
5. The mask (with feathered edges) composites the textured result back onto the original.

---

## Requirements

- Python 3.8+
- [OpenCV](https://pypi.org/project/opencv-python/) (`opencv-python`)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [Pillow](https://pypi.org/project/Pillow/)
- NumPy

Install all dependencies:

```bash
pip install opencv-python customtkinter pillow numpy
```

---

## Usage

```bash
python pattern_transfer.py
```

### Steps

1. **Clean Image** — Browse to a plain/flat image (the base).
2. **Textured Image** — Browse to an image containing the pattern or texture you want to transfer.
3. **Mask Image** — Browse to a grayscale image where **white** marks the area to receive the texture.
4. Click **Load Preview** to render the preview panels.
5. Adjust the three sliders to taste.
6. Click **Save Result** to export the full-resolution output.

### Inputs

| Input | Format | Notes |
|---|---|---|
| Clean Image | Color (BGR) | PNG, JPG, BMP, TIFF, WebP |
| Textured Image | Color (BGR) | Resized automatically to match clean image |
| Mask Image | Grayscale | White = apply texture, Black = preserve original |

### Output

The result is saved as a PNG by default (`pattern_transfer_result.png`). JPEG is also supported via the save dialog.

---

## Project Structure

```
pattern_transfer.py   # Single-file application
README.md
```

### Key Functions

| Function | Description |
|---|---|
| `prepare_transfer()` | Loads and preprocesses all three images into a `TransferData` struct |
| `render_transfer()` | Applies the texture blend given current slider values |
| `build_mask()` | Thresholds and feathers the grayscale mask |
| `enhance_luminance()` | Runs CLAHE on the texture's L channel |
| `extract_detail()` | High-frequency detail extraction via Gaussian subtraction |

---


