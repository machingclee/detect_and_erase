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
        torch.load("chi_text_erasing/final_pth/model.pth", map_location=device)
    )
    inpaint_gen.eval()

    test_img_paths = glob("test/in/*.jpg") + glob("test_img/in/*.png")
    out_dir = "test/out"

    for img_path in test_img_paths:
        basename = os.path.basename(img_path)
        out_path = f"{out_dir}/{basename}"
        img = clean_image(img_path, east, inpaint_gen)
        img.save(out_path)
        print(f"Saved result in {out_path}.")


if __name__ == "__main__":
    main()
