from chi_text_detection import compute_boxes
from chi_text_erasing import resize_and_padding, torch_img_transform, reverse_preprocessing
from PIL import Image
from .device import device


def clean_image(img_path, text_detection_module, inpaint_module, return_detections=False):
    img = Image.open(img_path)
    bboxes = compute_boxes(img, text_detection_module, device)
    detections = []
    erased_detections = []
    if bboxes is not None:
        for bbox in bboxes:
            """
            bbox: eight coordinates of 4 points
            """
            try:
                upper_left = (int(bbox[0]), int(bbox[1]))  # x, y
                box = (bbox[0], bbox[1], bbox[4], bbox[5])
                txt_img = img.crop(box)
                txt_img_ = img.crop(box)
                txt_img, padding_window, original_wh = resize_and_padding(txt_img, return_window=True)
                txt_img = torch_img_transform(txt_img).to(device)
                erased, _ = inpaint_module(txt_img[None, ...])
                erased = erased[0].permute(1, 2, 0).cpu().detach().numpy()
                erased = reverse_preprocessing(erased, padding_window, original_wh)
                img.paste(erased, upper_left)

                if return_detections:
                    detections.append(txt_img_)
                    erased_detections.append(erased)
            except Exception as err:
                print(f"{err}")
    if return_detections:
        return img, detections, erased_detections
    else:
        return img
