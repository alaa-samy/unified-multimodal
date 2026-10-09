"""
clip_color.py

Zero-shot color classification using CLIP itself, to replace pixel-averaged
color extraction in preprocess_pix3d.py / convert_pix3d_models.py.

WHY THIS EXISTS
----------------
The original `extract_color_from_image()` / `rgb_to_basic_name()` pipeline
averages every pixel in the product photo (background, floor, shadows,
studio lighting included) and maps that single average RGB value to the
nearest "basic color" by Euclidean distance. This is why searches like
"red sofa" can return black or brown sofas: the stored color often reflects
the photo's background tone, not the product itself.

This module instead asks CLIP directly, in a zero-shot classification
setup, "which color word best matches this product photo, given it's a
{category}?" This is grounded in CLIP's learned visual-semantic space
(the same space used for retrieval) rather than raw pixel statistics, so
it is much less sensitive to background/lighting contamination.

No new model or dependency is introduced — this reuses the CLIP model and
processor you already load in preprocessing (`clip_model`, `clip_processor`)
and in main.py (`_clip_model`, `_clip_proc`).

USAGE (drop-in replacement)
----------------------------
Before (in preprocess_pix3d.py):

    rgb = extract_color_from_image(img)
    color = rgb_to_basic_name(rgb)

After:

    from clip_color import classify_color_clip
    color = classify_color_clip(clip_model, clip_processor, img, category, device)

Nothing else in the preprocessing script needs to change — `color` is still
a plain string like "red", "black", "brown", etc., matching the existing
`baseColour` / `color` fields in clean_products.json, so main.py's
`_extract_color` / `_matches_color` logic keeps working unmodified.
"""

import torch

# Same 14 colors already used in main.py's _COLORS dict, so results stay
# compatible with the existing color-filter logic (_extract_color / _matches_color).
COLOR_CATEGORIES = [
    "black", "white", "red", "green", "blue", "yellow", "brown",
    "gray", "orange", "pink", "purple", "beige", "navy", "burgundy",
]


def classify_color_clip(clip_model, clip_processor, pil_image, category_name, device):
    """
    Zero-shot classify the dominant *product* color of a single product photo.

    Args:
        clip_model: a loaded CLIPModel (already in eval() mode)
        clip_processor: the matching CLIPProcessor
        pil_image: PIL.Image, RGB
        category_name: str, e.g. "sofa", "chair" — used to make the prompt
                        specific ("a photo of a red sofa" scores better than
                        just "a photo of red")
        device: torch.device

    Returns:
        str: one of COLOR_CATEGORIES, e.g. "red"
    """
    category_name = (category_name or "product").lower()
    prompts = [f"a photo of a {c} {category_name}" for c in COLOR_CATEGORIES]

    inputs = clip_processor(
        text=prompts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        # logits_per_image: (1, num_prompts) — similarity of the image to each prompt
        probs = outputs.logits_per_image.softmax(dim=-1)[0]

    best_idx = int(probs.argmax().item())
    return COLOR_CATEGORIES[best_idx]


def classify_colors_batch(clip_model, clip_processor, pil_images, category_names, device, batch_size=16):
    """
    Optional batched version for faster offline preprocessing over many images.
    pil_images and category_names must be the same length and order.
    Returns a list of color strings, same order as input.
    """
    results = []
    for i in range(0, len(pil_images), batch_size):
        batch_imgs = pil_images[i:i + batch_size]
        batch_cats = category_names[i:i + batch_size]

        # Build one prompt set per image (categories can differ within a batch)
        for img, cat in zip(batch_imgs, batch_cats):
            results.append(classify_color_clip(clip_model, clip_processor, img, cat, device))

    return results