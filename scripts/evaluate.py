import os
import random
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from jiwer import cer, wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from app.models.predict import predict

# Configuration

ROOT = Path(__file__).resolve().parent.parent

TEST_LABEL_FILE = ROOT / "dataset" / "HME100K" / "test" / "test_labels.txt"
TEST_IMAGE_DIR = ROOT / "dataset" / "HME100K" / "test" / "test_images"

OUTPUT_DIR = ROOT / "evaluation"
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_IMAGES = 100
BEAM_SIZE = 5
RANDOM_SEED = 42


# Load Test Labels

label_map = {}

with open(TEST_LABEL_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if not line.strip():
            continue

        parts = line.strip().split("\t")

        if len(parts) < 2:
            continue

        label_map[parts[0]] = parts[1]

filenames = [
    f for f in sorted(os.listdir(TEST_IMAGE_DIR))
    if f in label_map
]

random.seed(RANDOM_SEED)

filenames = random.sample(
    filenames,
    min(NUM_IMAGES, len(filenames))
)

print("=" * 60)
print(f"Evaluating {len(filenames)} images")
print("=" * 60)

# Evaluation

smooth = SmoothingFunction().method1

results = []

cer_scores = []
wer_scores = []
bleu_scores = []

correct = 0

prediction_times = []

for filename in tqdm(filenames):

    image_path = TEST_IMAGE_DIR / filename

    ground_truth = label_map[filename].strip()

    start = time.perf_counter()

    try:

        prediction = predict(
            str(image_path),
            decode_method="beam",
            beam_size=BEAM_SIZE
        )

    except Exception as e:

        prediction = ""

    elapsed = time.perf_counter() - start

    prediction_times.append(elapsed)

    exact = prediction.strip() == ground_truth

    if exact:
        correct += 1

    cer_value = cer(ground_truth, prediction)
    wer_value = wer(ground_truth, prediction)

    bleu = sentence_bleu(
        [ground_truth.split()],
        prediction.split(),
        smoothing_function=smooth
    )

    cer_scores.append(cer_value)
    wer_scores.append(wer_value)
    bleu_scores.append(bleu)

    results.append({

        "Image": filename,

        "Ground Truth": ground_truth,

        "Prediction": prediction,

        "Exact Match": exact,

        "CER": cer_value,

        "WER": wer_value,

        "BLEU": bleu,

        "Inference Time (s)": elapsed

    })

# Summary Metrics

exact_accuracy = correct / len(results)

avg_cer = np.mean(cer_scores)
avg_wer = np.mean(wer_scores)
avg_bleu = np.mean(bleu_scores)

avg_time = np.mean(prediction_times)

summary = pd.DataFrame({

    "Metric": [

        "Images Evaluated",

        "Exact Match Accuracy",

        "Average CER",

        "Average WER",

        "Average BLEU",

        "Average Inference Time (s)"

    ],

    "Value": [

        len(results),

        exact_accuracy,

        avg_cer,

        avg_wer,

        avg_bleu,

        avg_time

    ]

})

summary.to_csv(
    OUTPUT_DIR / "metrics.csv",
    index=False
)

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "sample_predictions.csv",
    index=False
)

# CER Histogram

plt.figure(figsize=(8,5))

plt.hist(
    cer_scores,
    bins=20
)

plt.title("Character Error Rate Distribution")

plt.xlabel("CER")

plt.ylabel("Images")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cer_histogram.png",
    dpi=300
)

plt.close()

# BLEU Histogram

plt.figure(figsize=(8,5))

plt.hist(
    bleu_scores,
    bins=20
)

plt.title("BLEU Score Distribution")

plt.xlabel("BLEU")

plt.ylabel("Images")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "bleu_histogram.png",
    dpi=300
)

plt.close()


# Best / Worst Predictions

best = results_df.sort_values(
    "CER"
).head(20)

worst = results_df.sort_values(
    "CER",
    ascending=False
).head(20)

best.to_csv(
    OUTPUT_DIR / "best_predictions.csv",
    index=False
)

worst.to_csv(
    OUTPUT_DIR / "worst_predictions.csv",
    index=False
)

# Text Summary


with open(
    OUTPUT_DIR / "evaluation_summary.txt",
    "w"
) as f:

    f.write("=" * 60 + "\n")

    f.write("FORMULA RECOGNITION EVALUATION\n")

    f.write("=" * 60 + "\n\n")

    f.write(f"Images Evaluated      : {len(results)}\n")

    f.write(f"Exact Match Accuracy : {exact_accuracy:.4f}\n")

    f.write(f"Average CER          : {avg_cer:.4f}\n")

    f.write(f"Average WER          : {avg_wer:.4f}\n")

    f.write(f"Average BLEU         : {avg_bleu:.4f}\n")

    f.write(f"Average Time         : {avg_time:.4f} sec\n")

print("\n" + "=" * 60)
print(summary)
print("=" * 60)

print("\nSaved to:")
print(OUTPUT_DIR)