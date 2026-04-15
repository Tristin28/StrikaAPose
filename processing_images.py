import mediapipe as mp
import cv2
from pathlib import Path

def converting_to_jpg(folder):
    images_dir = Path(folder).resolve()
    for file_path in images_dir.rglob("*"):

        if not file_path.is_file():
            continue

        img = cv2.imread(str(file_path))
        if img is None:
            print(f"[UNREADABLE] {file_path}")
            continue

        new_path = file_path.with_suffix(".jpg")
        success = cv2.imwrite(str(new_path), img)

        if success and new_path.exists():
            print(f"[CONVERTED] {file_path} to {new_path}")
            if file_path != new_path:
                file_path.unlink()
        else:
            print(f"[SAVE FAILED] {file_path}")
            continue

def resize_if_small(image,min_width=200):
    '''
        Resizes image only if its width is smaller than min_width.
    '''

    height, width = image.shape[:2]

    if width >= min_width:
        return image  

    scale = min_width / width

    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image


if __name__ == "__main__":
    converting_to_jpg("Images")