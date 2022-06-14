import os
import torch

from chi_text_detection import EAST, compute_boxes
from chi_text_erasing import InpaintGenerator, resize_and_padding, torch_img_transform, reverse_preprocessing
from PIL import Image
from glob import glob
from src.utils import clean_image
from src.device import device


def main():
    east = EAST().to(device)
    east.load_state_dict(
        torch.load("chi_text_detection/final_pth/model_epoch_9.pth", map_location=device)
    )
    east.eval()

    inpaint_gen = InpaintGenerator().to(device)

    inpaint_gen.load_state_dict(
        torch.load("chi_text_erasing/final_pth/model_epoch_15.pth", map_location=device)
    )
    inpaint_gen.eval()

    test_img_paths = glob("test/in/*.jpg") + glob("test/in/*.png")
    detection_detail_dir = "test/out/detection_detail"

    if not os.path.exists(detection_detail_dir):
        os.makedirs(detection_detail_dir)

    preview_dir = "test/out"

    for id, img_path in enumerate(test_img_paths):
        img_id = str(id).zfill(4)
        detail_dir_for_curr_img = f"{detection_detail_dir}/{img_id}".replace("jpg", "").replace("png", "")
        img, detections, erased_detections = clean_image(img_path, east, inpaint_gen, return_detections=True)

        if not os.path.exists(detail_dir_for_curr_img):
            os.makedirs(detail_dir_for_curr_img)

        for count, (detection, erased) in enumerate(zip(detections, erased_detections)):
            detection.save(f"{detail_dir_for_curr_img}/result_{count}_detected.png")
            erased.save(f"{detail_dir_for_curr_img}/result_{count}_erased.png")

        Image.open(img_path).save(f"{preview_dir}/{img_id}_original.jpg")
        img.save(f"{preview_dir}/{img_id}_erased.jpg")

        print(f"Saved result in {detail_dir_for_curr_img}.")


if __name__ == "__main__":
    main()
