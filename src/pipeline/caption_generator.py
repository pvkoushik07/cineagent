"""
Caption Generator — Phase 1 data pipeline.

Uses Gemini Flash vision to auto-generate rich textual descriptions
of poster images and scene stills. Captions become the third retrieval
modality (caption-based text search) in Ablation 1.

Saves all captions to data/processed/captions.json.

Usage:
    python src/pipeline/caption_generator.py
"""

import base64
import json
import logging
import time
from pathlib import Path

import google.generativeai as genai

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GEMINI_API_KEY, RAW_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

CAPTIONS_FILE = PROCESSED_DIR / "captions.json"

# Prompts designed to extract MOOD and AESTHETIC — not just objects.
# This is critical because our cross-modal queries are mood-based.
POSTER_CAPTION_PROMPT = """Describe this movie poster in detail, focusing on:
1. Visual mood and atmosphere (e.g. cold, warm, dark, hopeful, oppressive)
2. Colour palette and dominant tones
3. Composition and visual style
4. Any symbolic or thematic elements visible
5. The overall aesthetic feeling it evokes

Be specific about visual qualities. Do not describe the plot.
Write 3-4 sentences."""

STILL_CAPTION_PROMPT = """Describe this movie scene still in detail, focusing on:
1. Visual atmosphere and lighting (e.g. harsh fluorescent, golden hour, cold grey overcast)
2. Setting and environment (interior/exterior, urban/rural, time period feel)
3. Colour palette and mood
4. Compositional elements (framing, space, isolation vs crowding)

Be specific and vivid. Focus on what a viewer would feel visually.
Write 2-3 sentences."""


def encode_image_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string for Gemini API.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_caption(image_path: str, prompt: str, model: genai.GenerativeModel) -> str:
    """
    Generate a caption for a single image using Gemini Flash vision.

    Args:
        image_path: Path to the image file
        prompt: The captioning prompt to use
        model: Gemini GenerativeModel instance

    Returns:
        Caption string, or empty string if generation failed
    """
    try:
        image_data = encode_image_base64(image_path)
        response = model.generate_content([
            {"mime_type": "image/jpeg", "data": image_data},
            prompt,
        ])
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Caption generation failed for {image_path}: {e}")
        return ""


def load_existing_captions() -> dict:
    """Load previously generated captions to allow resuming."""
    if CAPTIONS_FILE.exists():
        with open(CAPTIONS_FILE) as f:
            return json.load(f)
    return {}


def save_captions(captions: dict) -> None:
    """Save captions dict to disk."""
    with open(CAPTIONS_FILE, "w") as f:
        json.dump(captions, f, indent=2)


def run_captioning() -> None:
    """
    Main entry point. Captions all images in data/raw/images/.

    Resumes from where it left off if captions.json already exists.
    Saves progress every 50 images to avoid losing work.
    """
    logger.info("Starting caption generation pipeline...")

    model = genai.GenerativeModel("gemini-1.5-flash")
    captions = load_existing_captions()

    images_dir = RAW_DIR / "images"
    all_images = list(images_dir.glob("*.jpg"))
    logger.info(f"Found {len(all_images)} images to caption")

    processed = 0
    for i, image_path in enumerate(all_images):
        image_key = image_path.stem  # e.g. "27205_poster" or "27205_still_0"

        if image_key in captions:
            continue  # Already captioned, skip

        # Choose prompt based on image type
        prompt = POSTER_CAPTION_PROMPT if "poster" in image_key else STILL_CAPTION_PROMPT
        caption = generate_caption(str(image_path), prompt, model)

        if caption:
            captions[image_key] = caption
            processed += 1

        # Save progress every 50 images
        if processed % 50 == 0 and processed > 0:
            save_captions(captions)
            logger.info(f"Progress: {i}/{len(all_images)} images captioned")

        time.sleep(0.5)  # Gemini free tier: 15 requests/min

    save_captions(captions)
    logger.info(f"Captioning complete. {processed} new captions generated.")
    logger.info(f"Total captions in file: {len(captions)}")


if __name__ == "__main__":
    run_captioning()
