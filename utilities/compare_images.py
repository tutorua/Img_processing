from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def resized_pair(a: np.ndarray, b: np.ndarray, size: tuple[int, int] = (512, 512)):
    return (cv2.resize(a, size, interpolation=cv2.INTER_AREA),
            cv2.resize(b, size, interpolation=cv2.INTER_AREA))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM after same-size resizing; a coarse baseline, not an alignment metric."""
    a, b = resized_pair(a, b)
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mu_x, mu_y = cv2.GaussianBlur(x, (11, 11), 1.5), cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
    return float(np.mean(score))


def _detector(name: str):
    if name == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=5000), cv2.NORM_L2
    if name not in ("sift", "orb"):
        raise ValueError("feature must be 'sift' or 'orb'")
    return cv2.ORB_create(nfeatures=6000, scaleFactor=1.2, nlevels=8), cv2.NORM_HAMMING


def feature_match(a: np.ndarray, b: np.ndarray, feature: str = "sift") -> dict[str, Any]:
    """Estimate a projective alignment and report inlier count and overlap geometry."""
    detector, norm = _detector(feature)
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ka, da = detector.detectAndCompute(ga, None)
    kb, db = detector.detectAndCompute(gb, None)
    result: dict[str, Any] = {"feature": feature, "keypoints_a": len(ka), "keypoints_b": len(kb),
                              "good_matches": 0, "inliers": 0, "inlier_ratio": 0.0,
                              "overlap_fraction_of_a": 0.0, "aligned_ssim": None}
    if da is None or db is None or len(da) < 2 or len(db) < 2:
        return result
    matcher = cv2.BFMatcher(norm)
    pairs = matcher.knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < 0.72 * n.distance]
    result["good_matches"] = len(good)
    if len(good) < 4:
        return result
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if homography is None or mask is None:
        return result
    inliers = int(mask.sum())
    result["inliers"] = inliers
    result["inlier_ratio"] = inliers / len(good)
    corners = np.float32([[0, 0], [0, a.shape[0]], [a.shape[1], a.shape[0]], [a.shape[1], 0]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, homography)
    projected_area = abs(cv2.contourArea(projected))
    target_bounds = np.float32([[0, 0], [0, b.shape[0] - 1], [b.shape[1] - 1, b.shape[0] - 1], [b.shape[1] - 1, 0]])
    try:
        visible_area, _ = cv2.intersectConvexConvex(projected.astype(np.float32), target_bounds)
        result["overlap_fraction_of_a"] = float(np.clip(visible_area / max(projected_area, 1.0), 0.0, 1.0))
    except cv2.error:
        result["overlap_fraction_of_a"] = 0.0
    # Compare only pixels where the warped source has coverage, avoiding black borders.
    warped = cv2.warpPerspective(a, homography, (b.shape[1], b.shape[0]))
    mask_a = cv2.warpPerspective(np.full(a.shape[:2], 255, np.uint8), homography, (b.shape[1], b.shape[0]))
    points = cv2.findNonZero(mask_a)
    if points is None:
        return result
    # Use the exact nonzero bounding rectangle for a stable overlap-only SSIM estimate.
    points = points.reshape(-1, 2)
    xs, ys = points[:, 0], points[:, 1]
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    if x1 - x0 >= 32 and y1 - y0 >= 32:
        crop_a, crop_b = warped[y0:y1, x0:x1], b[y0:y1, x0:x1]
        result["aligned_ssim"] = ssim(crop_a, crop_b)
    return result


def patch_match(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Find the smaller image as a crop in the larger image over multiple scales."""
    if a.shape[0] * a.shape[1] <= b.shape[0] * b.shape[1]:
        template, search, template_name = a, b, "a"
    else:
        template, search, template_name = b, a, "b"
    th, tw = template.shape[:2]
    best = {"template_image": template_name, "correlation": -1.0, "scale": None, "location": None}
    # Test the template at its source scale plus nearby zoom adjustments.
    for scale in np.geomspace(0.45, 2.2, 28):
        w, h = int(tw * scale), int(th * scale)
        if w < 32 or h < 32 or w > search.shape[1] or h > search.shape[0]:
            continue
        query = cv2.resize(template, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        scores = cv2.matchTemplate(cv2.cvtColor(search, cv2.COLOR_BGR2GRAY),
                                   cv2.cvtColor(query, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(scores)
        if score > best["correlation"]:
            best.update(correlation=float(score), scale=float(scale), location=[int(loc[0]), int(loc[1])])
    return best


def layout_similarity(a: np.ndarray, b: np.ndarray, grid: int = 4) -> dict[str, Any]:
    """Compare coarse per-cell color and edge structure, useful as a layout/region baseline."""
    a, b = resized_pair(a, b, (512, 512))
    features = []
    for image in (a, b):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        cells = []
        for row in range(grid):
            for col in range(grid):
                ys = slice(row * 512 // grid, (row + 1) * 512 // grid)
                xs = slice(col * 512 // grid, (col + 1) * 512 // grid)
                hist = cv2.calcHist([hsv[ys, xs]], [0, 1], None, [12, 8], [0, 180, 0, 256])
                cv2.normalize(hist, hist)
                cells.append((hist.ravel(), float(edges[ys, xs].mean() / 255)))
        features.append(cells)
    color_scores = [cv2.compareHist(x[0].astype(np.float32), y[0].astype(np.float32), cv2.HISTCMP_CORREL)
                    for x, y in zip(*features)]
    edge_scores = [1 - abs(x[1] - y[1]) for x, y in zip(*features)]
    return {"grid": f"{grid}x{grid}", "mean_cell_color_correlation": float(np.mean(color_scores)),
            "mean_cell_edge_similarity": float(np.mean(edge_scores))}


def multiscale_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Compare globally at progressively smaller resolutions; robust to fine texture changes."""
    scores = {}
    for side in (512, 256, 128, 64):
        x, y = resized_pair(a, b, (side, side))
        gx = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gy = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Correlation is invariant to overall brightness/contrast changes.
        scores[str(side)] = float(cv2.matchTemplate(gx, gy, cv2.TM_CCOEFF_NORMED)[0, 0])
    return {"scale_correlations": scores, "mean": float(np.mean(list(scores.values())))}


def optional_onnx_embedding(path_a: str | Path, path_b: str | Path, model_path: str | Path) -> float:
    """Cosine similarity from a user-supplied image-embedding ONNX model."""
    net = cv2.dnn.readNetFromONNX(str(model_path))
    vectors = []
    for path in (path_a, path_b):
        image = load_image(path)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1 / 255.0,
                                     swapRB=True, crop=False)
        net.setInput(blob)
        vector = net.forward().astype(np.float32).ravel()
        vectors.append(vector / max(float(np.linalg.norm(vector)), 1e-12))
    return float(np.dot(vectors[0], vectors[1]))


def compare(path_a: Path, path_b: Path, feature: str = "sift", model: Path | None = None) -> dict[str, Any]:
    a, b = load_image(path_a), load_image(path_b)
    result: dict[str, Any] = {
        "image_a": path_a.name, "image_b": path_b.name,
        "size_a": [int(a.shape[1]), int(a.shape[0])], "size_b": [int(b.shape[1]), int(b.shape[0])],
        "resized_ssim": ssim(a, b),
        "features": feature_match(a, b, feature),
        "patch_match": patch_match(a, b),
        "layout": layout_similarity(a, b),
        "multiscale": multiscale_similarity(a, b),
    }
    if model:
        result["embedding_cosine_similarity"] = optional_onnx_embedding(path_a, path_b, model)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare image pairs despite size or zoom differences.")
    parser.add_argument("images", nargs="*", type=Path, help="Images to compare; defaults to project PNGs")
    parser.add_argument("--feature", choices=("sift", "orb"), default="sift")
    parser.add_argument("--onnx", type=Path, help="Optional image-embedding ONNX model")
    parser.add_argument("--output", type=Path, help="Write JSON results to this path")
    args = parser.parse_args()
    images = args.images or sorted(Path(__file__).resolve().parent.parent.glob("*.png"))
    if len(images) < 2:
        parser.error("Provide at least two readable images")
    results = [compare(a, b, args.feature, args.onnx) for i, a in enumerate(images) for b in images[i + 1:]]
    rendered = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
