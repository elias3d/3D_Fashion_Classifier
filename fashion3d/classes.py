"""The 8 fashion types the model knows about (order = model output order)."""

CLASS_NAMES = ["tshirt", "pants", "pullover", "dress", "coat", "shirt", "skirt", "shoes"]

DISPLAY_NAMES = {
    "tshirt": "T-shirt",
    "pants": "Pants",
    "pullover": "Pullover",
    "dress": "Dress",
    "coat": "Coat",
    "shirt": "Shirt",
    "skirt": "Skirt",
    "shoes": "Shoes",
}


def display(label: str) -> str:
    return DISPLAY_NAMES.get(label, label)
