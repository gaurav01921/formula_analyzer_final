# Preprocessing routines, image transforms, and search decoders (Greedy/Beam).
import io
import base64
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
    tensor_image = predict_transform(padded_image).unsqueeze(0)  # Shape: (1, 3, H, W)
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


def segment_formula_lines(image_input, padding=12, min_line_height=18):
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
    img_h, img_w = gray.shape

    # 1. Smooth slightly to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Adaptive Gaussian thresholding 
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # 3. Horizontal Projection Profile 
    row_sums = np.sum(thresh, axis=1)
    max_row = np.max(row_sums)
    
    if max_row == 0:
        return [pil_img]

    norm_profile = row_sums / max_row
    
    # Threshold to identify rows with active handwriting ink
    is_text = norm_profile > 0.03
    
    # 4. Find contiguous vertical ranges (start_y, end_y)
    line_ranges = []
    in_line = False
    start_y = 0
    
    for y, active in enumerate(is_text):
        if active and not in_line:
            in_line = True
            start_y = y
        elif not active and in_line:
            in_line = False
            if (y - start_y) >= min_line_height:
                line_ranges.append((start_y, y))
                
    if in_line and (img_h - start_y) >= min_line_height:
        line_ranges.append((start_y, img_h))

    merged_ranges = []
    min_gap = int(img_h * 0.05)  # 5% of total image height gap threshold
    for r in line_ranges:
        if not merged_ranges:
            merged_ranges.append(r)
        else:
            prev_start, prev_end = merged_ranges[-1]
            curr_start, curr_end = r
            if (curr_start - prev_end) < min_gap:
                merged_ranges[-1] = (prev_start, curr_end)
            else:
                merged_ranges.append(r)

    if len(merged_ranges) <= 1:
        return [pil_img]

    # 6. Crop each detected formula line sub-image
    crops = []
    for y1, y2 in merged_ranges:
        crop_y1 = max(0, y1 - padding)
        crop_y2 = min(img_h, y2 + padding)
        crop = pil_img.crop((0, crop_y1, img_w, crop_y2))
        crops.append(crop)
        
    return crops


def create_causal_mask(seq_len, device="cpu"):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float("-inf"))
    return mask


@torch.no_grad()
def greedy_decode(model, image_tensor, vocab, device="cpu", max_length=MAX_FORMULA_LENGTH):
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
