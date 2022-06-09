from chi_text_detection import EAST, compute_boxes
from chi_text_erasing import InpaintGenerator, preprocessing, prefeed_img_transform, reverse_preprocessing
from PIL import Image
from glob import glob
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch


def clean_image(img_path, text_detection_module, inpaint_module):
    img = Image.open(img_path)
    bboxes = compute_boxes(img, text_detection_module, device)

    for bbox in bboxes:
        """
        bbox: eight coordinates of 4 points
        """
        upper_left = (int(bbox[0]), int(bbox[1])) # x, y
        box = (bbox[0], bbox[1], bbox[4], bbox[5])
        txt_img = img.crop(box)
        txt_img, padding_window, (ori_h, ori_w) = preprocessing(txt_img, return_window=True)
        txt_img = prefeed_img_transform(txt_img)
        erasied, _ = inpaint_module(txt_img[None,...])
        erasied = erasied[0].permute(1, 2, 0).detach().numpy()
        erasied = reverse_preprocessing(erasied, padding_window, ori_w, ori_h)
        img.paste(erasied, upper_left)
    img.show()




def main():
    east = EAST().to("cpu")
    east.load_state_dict(
        torch.load("chi_text_detection/final_pth/model_epoch_9.pth", map_location=device)
    )
    east.eval()

    inpaint_gen = InpaintGenerator().to("cpu")
    inpaint_gen.load_state_dict(
        torch.load("chi_text_erasing/final_pth/model_epoch_4.pth", map_location=device)
    )
    inpaint_gen.eval()

    test_img_paths = glob("test_img/*.jpg") + glob("test_img/*.png")

    for img_path in test_img_paths:
        clean_image(img_path, east, inpaint_gen)
    
    

    


if __name__ == "__main__":
    main()