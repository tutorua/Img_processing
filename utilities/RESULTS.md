# Sample comparison results

The three project images were compared with both SIFT and ORB feature matching. The most reliable signal on these samples is the feature alignment (RANSAC inliers and overlap-only SSIM).

| Pair | SIFT inliers / good matches | Estimated overlap of image A | Aligned SSIM | Reading |
| --- | ---: | ---: | ---: | --- |
| `original.png` ↔ `zummed_in.png` | 32 / 35 | 23% | 0.826 | Same map area is present, but only a smaller part of the original is covered by the zoomed-in view. |
| `original.png` ↔ `zummed_out.png` | 201 / 203 | 60% | 0.858 | Strong match with substantial shared coverage. |
| `zummed_in.png` ↔ `zummed_out.png` | 301 / 306 | 100% | 0.982 | Strongest pair; the zoomed-in image maps fully inside the zoomed-out image. |

ORB also found the same ranking: aligned SSIM values were 0.775, 0.797, and 0.966 respectively. SIFT produced more inliers and slightly higher alignment quality on these examples.

Resized SSIM alone ranged from 0.47 to 0.54 and obscured the strong shared content because the framing and aspect ratios differ. The multi-scale template correlation was also weak (about 0.25 or lower), so it should be treated as a diagnostic only for these map samples. The coarse region color comparison favored the `zummed_in`/`zummed_out` pair, in line with feature matching.

These are sample-specific results rather than universal thresholds. For this project, use SIFT/RANSAC as the primary similarity and overlap check, with aligned SSIM as a visual-content check. A future map workflow could improve overlap estimates by using georeferencing when available.
