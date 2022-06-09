import os
import sys
import torch

parent_dir = os.path.normpath( os.path.join(os.path.realpath(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from chi_text_detection import EAST, compute_boxes
from chi_text_erasing import InpaintGenerator, resize_and_padding, prefeed_img_transform, reverse_preprocessing
from PIL import Image
from glob import glob
from src.utils import clean_image
from src.device import device


import torch

def main():
    east = EAST().to("cpu")
    east.load_state_dict(
        torch.load("chi_text_detection/final_pth/model_epoch_9.pth", map_location=device)
    )
    east.eval()

    inpaint_gen = InpaintGenerator().to("cpu")
    inpaint_gen.load_state_dict(
        torch.load("chi_text_erasing/final_pth/model_epoch_2.pth", map_location=device)
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