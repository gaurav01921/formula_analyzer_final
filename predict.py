# ==========================================================\n# Handwritten Mathematical Formula Recognition System
# Module: predict.py
# Description: Model Predictor Singleton for high-performance inference.
#              Loads PyTorch model weights into GPU/CPU memory ONCE on application startup.
# ==========================================================\n
import os
import torch
from model import FormulaRecognizer, load_vocab_file
from utils import (
    preprocess_image, 
    greedy_decode, 
    beam_search_decode, 
    segment_formula_lines,
    image_to_base64,
    create_otsu_visualization
)

# Global Singleton Predictor Instance
_PREDICTOR_INSTANCE = None


class ModelPredictor:
    """
    Singleton predictor class managing Vocabulary and PyTorch FormulaRecognizer model state.
    Ensures model weights are loaded exactly ONCE during backend startup.
    """

    def __init__(self, model_path="weights/best_model.pth", vocab_path="weights/vocab.pkl"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("=" * 60)
        print(f"[ModelPredictor] Initializing Inference Engine on Device: {self.device}")

        # 1. Load Vocabulary
        self.vocab = load_vocab_file(vocab_path)
        print(f"[ModelPredictor] Vocabulary Loaded (Size: {len(self.vocab)})")

        # 2. Instantiate Model Architecture
        self.model = FormulaRecognizer(
            vocab_size=len(self.vocab),
            embed_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1,
            pad_idx=self.vocab.token2idx["<PAD>"]
        ).to(self.device)

        # 3. Load Saved Weights
        if not os.path.exists(model_path):
            if os.path.exists("checkpoints/best_model.pth"):
                model_path = "checkpoints/best_model.pth"
            else:
                raise FileNotFoundError(f"Model checkpoint not found at '{model_path}'.")

        # Support checkpoint unpickling where Vocabulary was saved under __main__
        import sys
        from model import Vocabulary
        sys.modules['__main__'].Vocabulary = Vocabulary

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Support state_dict loading directly or from checkpoint dictionary
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[ModelPredictor] Checkpoint Loaded (Epoch: {checkpoint.get('epoch', 'N/A')})")
        elif isinstance(checkpoint, dict):
            self.model.load_state_dict(checkpoint)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print("[SUCCESS] [ModelPredictor] Model Weights Loaded & Set to Evaluation Mode!")
        print("=" * 60)

    def predict(self, image_input, decode_method="beam", beam_size=5):
        """
        Executes model forward pass and decoding.
        
        Args:
            image_input (str, bytes, PIL.Image): Image file path, raw bytes, or PIL Image object.
            decode_method (str): Decoding strategy - 'beam' or 'greedy'.
            beam_size (int): Beam width for beam search decoding.

        Returns:
            str: Predicted LaTeX formula string.
        """
        tensor_image, _ = preprocess_image(image_input)
        tensor_image = tensor_image.to(self.device)

        if decode_method == "greedy":
            prediction = greedy_decode(
                self.model, tensor_image, self.vocab, device=self.device
            )
        else:
            prediction = beam_search_decode(
                self.model, tensor_image, self.vocab, device=self.device, beam_size=beam_size
            )
        return prediction

    def predict_multiline(self, image_input, decode_method="beam", beam_size=5):
        """
        Detects formula lines using OpenCV segmenter and predicts each line independently.
        Returns a dict containing 'predictions' list, combined formula string, and pipeline images.
        """
        line_crops = segment_formula_lines(image_input)
        line_preds = []
        line_crops_b64 = []

        for crop in line_crops:
            pred = self.predict(crop, decode_method=decode_method, beam_size=beam_size)
            if pred and pred.strip():
                line_preds.append(pred.strip())
            line_crops_b64.append(image_to_base64(crop))
        
        if len(line_preds) > 1:
            combined = " \\\\ \n".join(line_preds)
        elif len(line_preds) == 1:
            combined = line_preds[0]
        else:
            combined = ""

        # Generate preprocessed tensor visualization for Step 2 UI
        _, padded_canvas = preprocess_image(line_crops[0])
        preprocessed_b64 = image_to_base64(padded_canvas)
        otsu_b64 = create_otsu_visualization(padded_canvas)

        return {
            "is_multiline": len(line_crops) > 1,
            "line_count": len(line_crops),
            "lines": line_preds,
            "prediction": combined,
            "preprocessed_image_base64": preprocessed_b64,
            "otsu_image_base64": otsu_b64,
            "line_crops_base64": line_crops_b64,
            "tensor_shape": "[1, 3, 128, 512]"
        }


def init_predictor(model_path="weights/best_model.pth", vocab_path="weights/vocab.pkl"):
    """Initializes global singleton predictor during application startup."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = ModelPredictor(model_path, vocab_path)
    return _PREDICTOR_INSTANCE


def get_predictor():
    """Gets existing singleton predictor instance."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = init_predictor()
    return _PREDICTOR_INSTANCE


def predict(image_path, decode_method="beam", beam_size=5):
    """
    Main prediction wrapper matching specification:
    predict(image_path) -> returns prediction formula string.
    """
    predictor = get_predictor()
    return predictor.predict(image_path, decode_method=decode_method, beam_size=beam_size)


def predict_multiline(image_path, decode_method="beam", beam_size=5):
    """
    Multi-line prediction wrapper returning structured predictions dictionary.
    """
    predictor = get_predictor()
    return predictor.predict_multiline(image_path, decode_method=decode_method, beam_size=beam_size)
