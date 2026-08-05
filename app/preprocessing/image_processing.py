import io
import base64
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128

predict_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def resize_with_padding(image, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    target_w, target_h = target_size
    w, h = image.size

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized_img = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
    left = (target_w - new_w) // 2
    top = (target_h - new_h) // 2

    canvas.paste(resized_img, (left, top))
    return canvas


def preprocess_image(image_input, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input format. Expected file path, bytes, or PIL Image.")

    padded_image = resize_with_padding(image, target_size)
    tensor_image = predict_transform(padded_image).unsqueeze(0)
    return tensor_image, padded_image


def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def create_otsu_visualization(pil_img):
    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    res_img = Image.fromarray(thresh).convert("RGB")
    return image_to_base64(res_img)
