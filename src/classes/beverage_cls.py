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
    # Paper carton & box - paper/cardboard
    "carton",
    # "paper box", # seperate from carton
    "empty paper box",
    "full paper box",
    "strawed drink",
    # Large containers - kegs, barrels, jugs
    "keg",
    # "jug",
    # Serving containers
    # "dish",
    # "pitcher",
    # "teapot",
    # "coffeepot",
    # # Catch-all for other beverage containers
    # "beverage",
]
