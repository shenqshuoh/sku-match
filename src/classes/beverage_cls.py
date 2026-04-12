"""Trimmed beverage container categories for YOLOE detection.

Focused on off-the-shelf beverage containers with different physical shapes.
Excludes accessories, cups, and content-specific distinctions.
Content type (cola vs orange juice) should be determined by SKU matcher.
"""

# Core beverage containers by physical form
# Used with YOLOE set_classes() for focused detection
BEVERAGE_CONTAINER_CLASSES = [
    # Bottles - all shapes and sizes
    "bottle",
    # Cans - aluminum/steel
    "canned",
    # Box - paper/cardboard
    "juice box",
    # Large containers - kegs, barrels, jugs
    "keg",
    "barrel",
    "canteen",
    "water jug",
    # Serving containers
    "pitcher",
    "teapot",
    "coffeepot",
    # Catch-all for other beverage containers
    "beverage",
]
