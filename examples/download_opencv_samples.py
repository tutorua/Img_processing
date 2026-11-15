"""Download sample images used in the OpenCV comparison demonstrations.

The downloaded image files are kept local and ignored by Git. Check the
upstream image/dataset terms before redistributing them.
"""

from pathlib import Path
from urllib.request import urlretrieve

import cv2


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "images" / "opencv_samples"
BASE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/"
ASSETS = ("graf1.png", "graf3.png", "H1to3p.xml", "box.png", "box_in_scene.png", "messi5.jpg")


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename in ASSETS:
        target = DESTINATION / filename
        if not target.exists():
            print(f"Downloading {filename}")
            urlretrieve(BASE_URL + filename, target)

    image = cv2.imread(str(DESTINATION / "messi5.jpg"))
    if image is None:
        raise FileNotFoundError(DESTINATION / "messi5.jpg")
    # A 100x100 crop around Messi's face for the patch-search example.
    template_path = DESTINATION / "messi_face_template.png"
    if not template_path.exists():
        template = image[45:145, 190:290]
        if template.shape[:2] != (100, 100) or not cv2.imwrite(str(template_path), template):
            raise RuntimeError("Could not create the Messi template crop")
    print(f"Sample images are ready in {DESTINATION}")


if __name__ == "__main__":
    main()
