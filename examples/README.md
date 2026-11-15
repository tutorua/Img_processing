# Comparison examples

The official OpenCV sample images provide more legible demonstrations than the satellite screenshots for a few specific methods. The images and generated views are downloaded locally and ignored by Git; the Python setup script and commands are tracked.

Download them and create the template crop:

```powershell
uv run python examples/download_opencv_samples.py
```

## Suggested demos

### Viewpoint change and homography

```powershell
uv run python -m utilities.compare_images examples/images/opencv_samples/graf1.png examples/images/opencv_samples/graf3.png --output examples/local_outputs/graffiti.json --visualize examples/local_outputs/graffiti
```

Add `--blur-kernel 3` to try a light Gaussian blur before feature detection; use a separate output directory to compare its keypoints and inlier matches with the unblurred run.

The graffiti images show the same textured wall at different viewpoints. OpenCV's AKAZE tutorial uses this pair and supplies a reference homography, making it useful for examining SIFT/ORB inliers and the estimated overlap.

### Small object inside a larger scene

```powershell
uv run python -m utilities.compare_images examples/images/opencv_samples/box.png examples/images/opencv_samples/box_in_scene.png --output examples/local_outputs/box.json --visualize examples/local_outputs/box
```

This pair is used in OpenCV's feature matching and homography tutorial. It demonstrates that a small region can match inside a larger image.

### Patch search at multiple scales

```powershell
uv run python -m utilities.compare_images examples/images/opencv_samples/messi5.jpg examples/images/opencv_samples/messi_face_template.png --output examples/local_outputs/messi_patch.json --visualize examples/local_outputs/messi_patch
```

The template crop is generated from `messi5.jpg` by the setup script. It gives the patch matcher an obvious small crop to locate in the full image.

## Sources and use

- OpenCV sample files: [box.png](https://github.com/opencv/opencv/blob/4.x/samples/data/box.png), [box_in_scene.png](https://github.com/opencv/opencv/blob/4.x/samples/data/box_in_scene.png), [graf1.png](https://github.com/opencv/opencv/blob/4.x/samples/data/graf1.png), [graf3.png](https://github.com/opencv/opencv/blob/4.x/samples/data/graf3.png), and [messi5.jpg](https://github.com/opencv/opencv/blob/4.x/samples/data/messi5.jpg).
- OpenCV tutorials: [feature matching and homography](https://docs.opencv.org/4.13.0/d1/de0/tutorial_py_feature_homography.html), [AKAZE graffiti matching](https://docs.opencv.org/5.0/tutorials/features/akaze/akaze_matching/akaze_matching.html), and [template matching](https://docs.opencv.org/5.0/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html).
- The graffiti images originate from Oxford's affine feature evaluation data. OpenCV's [AKAZE tutorial](https://docs.opencv.org/5.0/tutorials/features/akaze/akaze_matching/akaze_matching.html) uses this pair and references `H1to3p.xml`; the Oxford page describes its test packages as including images and homographies. The OpenCV sample repository does not provide a separate image license in this folder, so these downloaded assets and generated visualizations are excluded from Git. Confirm the source image rights before redistributing them.
- The same local-only treatment is used for the box and Messi sample images: the project tracks the download/setup script and source references, not the image files or visualizations.

These examples are chosen to explain individual techniques. They are not a replacement for the project's satellite examples, which better represent map screenshots with different zoom and coverage.

## Results from this project

The commands above were run with the current SIFT implementation:

| Example | SIFT inliers / good matches | Overlap of image A | Aligned SSIM | Most useful takeaway |
| --- | ---: | ---: | ---: | --- |
| Graffiti viewpoint pair | 371 / 432 | 97% | 0.488 | Raw resized SSIM was only 0.143 despite hundreds of geometrically consistent matches. Viewpoint and appearance changes lower pixel similarity. |
| Box inside scene | 72 / 76 | ~100% | 0.529 | A smaller object can be located and aligned inside a larger image; raw resized SSIM was 0.201. |
| Messi face crop in full photo | 73 / 79 | 5% | 0.994 | The multi-scale patch matcher found the crop at correlation 0.933 and scale 1.02. This is the clearest patch-search demo. |

The reports and visualization images are in `examples/local_outputs/` after running the commands. They are also ignored by Git because they are derived from the sample images.
