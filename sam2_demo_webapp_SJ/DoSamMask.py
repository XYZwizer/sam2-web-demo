import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
#import matplotlib.pyplot as plt
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device} for sam2")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2


sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

def apply_mask(image ,mask, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([255, 255, 0, 255])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    print("mask shape",mask.shape)
    #mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    mask_image = np.zeros((h,w,4),dtype=np.uint8)
    mask_image[mask.reshape(h, w) == 1] =color
   # if borders:
   #     import cv2
   #     contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
   #     # Try to smooth contours
   #     contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
   #     mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
   #     cv2.imshow('Image Window', mask_image)
    test = Image.fromarray(mask_image)
    test.save("geeks.png")
    return mask_image

def gen_mask(image,points):
    predictor.set_image(image)

    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    return masks[0]

def make_masked_img(image,points):
    image = Image.open(image)
    image = np.array(image.convert("RGB"))

    mask_points = gen_mask(image,points)
    new_image   = apply_mask(image,mask_points)

    return new_image