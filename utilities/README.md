# Image comparison utilities

This package compares images that may have different dimensions, zoom levels, or crop boundaries. Run it from the project root:

```powershell
uv run python -m utilities.compare_images
uv run python -m utilities.compare_images original.png zummed_in.png --feature orb
uv run python -m utilities.compare_images --blur-kernel 3
uv run python -m utilities.compare_images --visualize
```

The default command finds the project PNGs, compares every pair, and prints a JSON report. Add `--output comparisons.json` to save the report.

Add `--visualize` to save per-pair keypoint images, RANSAC inlier match lines, a green-tinted overlap overlay, and an overlap-only difference image in `comparison_visuals/`. Pass a directory after the flag to choose another destination, for example `--visualize artifacts/map-check`.

Use `--blur-kernel 3` to apply a 3×3 Gaussian blur to the grayscale images before SIFT/ORB feature detection. The default `0` keeps blur disabled. The kernel must be a positive odd number. This preprocessing changes only the feature-matching path; pixel, patch, color/layout, and embedding comparisons use the original images.

## Methods

- **Resized SSIM** resizes both images to a common canvas and computes structural similarity. This is a quick baseline and can under-score a shared scene when framing differs.
- **SIFT or ORB features** match local points and estimate a homography with RANSAC. Inlier count and ratio indicate alignment evidence; `aligned_ssim` compares the approximate overlapping bounding region. SIFT is the default; ORB is available with `--feature orb`.
- **Multi-scale patch matching** searches for the smaller image within the larger image at a range of scales. It is useful for crop/zoom relationships; its correlation can be fooled by repetitive textures.
- **Coarse region/layout comparison** compares color histograms and edge density in corresponding grid cells. It is a broad structural baseline, not semantic object detection or OCR.
- **Multi-scale global correlation** reports a correlation at several downsampled sizes to show whether broad composition agrees.
- **Optional embeddings** can be requested with `--onnx path\to\embedding-model.onnx`. This expects an ONNX image encoder accepting a 224x224 RGB image tensor scaled to 0–1 and returning an embedding vector. No model is bundled.

The visualization overlay shows the transformed first image blended with the second inside the estimated overlap. The difference view is contrast-stretched within that overlap, so it highlights small changes but does not show their original brightness scale.

A 3×3 blur improved the SIFT inlier ratio on the graffiti and Messi examples but slightly reduced it on the box pair. Keep blur optional and compare inlier quality for the images you care about.

Scores from these methods have different meanings and should not be averaged into one uncalibrated score. For map imagery, feature inliers plus patch correlation are the most useful overlap indicators in this starter implementation.
