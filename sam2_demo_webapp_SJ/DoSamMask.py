import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import json
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


sam2_checkpoint = "../sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

def apply_mask(image ,mask, random_color=False, borders = True):
    Background_color = np.array([255, 255, 255])

    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask = mask.reshape(h, w)

    masked_image_array = image.copy()
    masked_image_array[mask == 0] = Background_color

    # masked_image.save("geeks.png")
    return masked_image_array

def points_from_json(points_json : str ,img_height,img_width):
    points = []
    labels = []

    point_label_pairs = json.loads(points_json)

    for point_label_pair in point_label_pairs:
        points.append([int(img_width * point_label_pair['x']),int(img_height * point_label_pair['y'])]) # transforms the points from presentges to in image coordinates
        labels.append(1 if point_label_pair['type'] == "include" else 0)
    return np.array(points), np.array(labels)

def make_masked_img(image, json_points):
    image = Image.open(image)
    image = np.array(image.convert("RGB"))

    predictor.set_image(image)

    img_height = image.shape[0]
    img_width = image.shape[1]

    print(f"image width:{img_width} , height:{img_height}")
    print("making mask")

    input_points, input_labels = points_from_json(json_points, img_height, img_width)
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )

    masked_image_array = apply_mask(image,masks[0])
    print(masked_image_array.shape)

    #for testing put green pixel on slected point
    #for point in input_points:
    #    masked_image_array[point[1]][point[0]] = np.array([0, 255, 0])

    return Image.fromarray(masked_image_array)