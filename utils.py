# ==========================================================\n# Handwritten Mathematical Formula Recognition System
# Module: utils.py
# Description: Preprocessing routines, image transforms, and search decoders (Greedy/Beam).
# ==========================================================\n
import io
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Preprocessing Constants matching training pipeline
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 128
MAX_FORMULA_LENGTH = 128

# ImageNet normalization transforms
predict_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def resize_with_padding(image, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    """
    Resizes image preserving original aspect ratio and pads the remaining background 
    canvas with white (255, 255, 255).
    """
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
    """
    Accepts an image input (file path, bytes, or PIL Image), preprocesses it, 
    and returns both the normalized tensor batch and the processed PIL canvas.
    """
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input format. Expected file path, bytes, or PIL Image.")

    padded_image = resize_with_padding(image, target_size)
    tensor_image = predict_transform(padded_image).unsqueeze(0)  # Shape: (1, 3, H, W)
    return tensor_image, padded_image


def segment_formula_lines(image_input, padding=12, min_height=20, min_width=40):
    """
    OpenCV Line Segmenter: Detects individual formula lines in a multi-line image.
    Crops each line into a standalone PIL Image.
    
    Args:
        image_input (str, bytes, PIL.Image): Input image.
        padding (int): Pixel margin added around each cropped bounding box.
        min_height (int): Minimum height filter for noise.
        min_width (int): Minimum width filter for noise.
        
    Returns:
        list of PIL.Image: Cropped line sub-images ordered from top to bottom.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input format.")

    img_np = np.array(pil_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Binarize with Otsu thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilation with horizontal kernel to group characters on the same line
    kernel_width = max(15, int(pil_img.width * 0.15))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 5))
    dilated = cv2.dilate(thresh, h_kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    img_h, img_w = gray.shape
    
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_width and h >= min_height and (w * h) >= (img_w * img_h * 0.015):
            boxes.append((x, y, w, h))

    # If 0 or 1 line box found, or box covers >85% of total area, return whole image
    if len(boxes) <= 1:
        return [pil_img]

    # Sort boxes top to bottom
    boxes = sorted(boxes, key=lambda b: b[1])

    # Crop each line sub-image with padding
    crops = []
    for x, y, w, h in boxes:
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
        
        crop = pil_img.crop((x1, y1, x2, y2))
        crops.append(crop)
        
    return crops


def create_causal_mask(seq_len, device="cpu"):
    """Creates a triangular causal mask for transformer target sequence."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


@torch.no_grad()
def greedy_decode(model, image_tensor, vocab, device="cpu", max_length=MAX_FORMULA_LENGTH):
    """
    Performs greedy search autoregressive decoding to generate a formula string.
    """
    model.eval()
    sos_idx = vocab.token2idx["<SOS>"]
    eos_idx = vocab.token2idx["<EOS>"]
    pad_idx = vocab.token2idx["<PAD>"]

    memory = model.encoder(image_tensor)
    seq = torch.tensor([[sos_idx]], device=device)

    for _ in range(max_length):
        tgt_mask = create_causal_mask(seq.size(1), device=device)
        tgt_padding_mask = (seq == pad_idx)
        
        output = model.decoder(
            memory=memory,
            tgt=seq,
            tgt_mask=tgt_mask,
            tgt_padding_mask=tgt_padding_mask
        )
        
        next_token = torch.argmax(output[:, -1, :], dim=-1).unsqueeze(1)
        seq = torch.cat([seq, next_token], dim=1)
        
        if next_token.item() == eos_idx:
            break

    indices = seq.squeeze().tolist()
    return vocab.decode(indices)


@torch.no_grad()
def beam_search_decode(model, image_tensor, vocab, device="cpu", beam_size=5, max_length=MAX_FORMULA_LENGTH, length_penalty=0.7):
    """
    Performs beam search autoregressive decoding with length normalization.
    """
    model.eval()
    sos_idx = vocab.token2idx["<SOS>"]
    eos_idx = vocab.token2idx["<EOS>"]
    pad_idx = vocab.token2idx["<PAD>"]

    memory = model.encoder(image_tensor)
    beams = [(torch.tensor([[sos_idx]], device=device), 0.0)]
    completed = []

    for _ in range(max_length):
        candidates = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:
                completed.append((seq, score))
                continue

            tgt_mask = create_causal_mask(seq.size(1), device=device)
            tgt_padding_mask = (seq == pad_idx)

            output = model.decoder(
                memory=memory,
                tgt=seq,
                tgt_mask=tgt_mask,
                tgt_padding_mask=tgt_padding_mask
            )
            logits = output[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            values, indices = torch.topk(log_probs, beam_size, dim=-1)

            for k in range(beam_size):
                token = indices[0, k].view(1, 1)
                new_seq = torch.cat([seq, token], dim=1)
                new_score = score + values[0, k].item()
                candidates.append((new_seq, new_score))

        if not candidates:
            break

        # Normalize score by sequence length
        candidates = sorted(
            candidates,
            key=lambda x: x[1] / ((x[0].size(1)) ** length_penalty),
            reverse=True
        )
        beams = candidates[:beam_size]

    if completed:
        best_seq = max(completed, key=lambda x: x[1] / ((x[0].size(1)) ** length_penalty))[0]
    else:
        best_seq = beams[0][0]

    indices = best_seq.squeeze().tolist()
    return vocab.decode(indices)
