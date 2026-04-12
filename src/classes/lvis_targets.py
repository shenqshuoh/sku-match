"""LVIS drink, beverage, and fluid container categories.

Extracted from LVIS dataset v1.0 (1203 categories).
Includes beverages, cartons, cans, bottles, kettles, and fluid containers.
"""

LVIS_DRINK_BEVERAGE_CATEGORIES = [
    # Beverages/Drinks
    (4, "alcohol/alcoholic beverage"),
    (205, "cappuccino/coffee cappuccino"),
    (249, "chocolate milk"),
    (256, "cider/cyder"),
    (281, "cocoa/cocoa beverage/hot chocolate/hot chocolate beverage/drinking chocolate"),
    (475, "fruit juice"),
    (639, "lemonade"),
    (650, "liquor/spirits/hard liquor/liqueur/cordial"),
    (688, "milk"),
    (690, "milkshake"),
    (735, "orange juice"),
    (831, "pop/pop soda/soda/soda pop/tonic/soft drink"),
    (893, "root beer"),
    (973, "smoothie"),
    (1079, "tequila"),
    (1145, "vodka"),
    (1199, "yogurt/yoghurt/yoghourt"),
    # Bottles
    (82, "beer bottle"),
    (132, "bottle"),
    (1081, "thermos bottle"),
    (1161, "water bottle"),
    (1187, "wine bottle"),
    # Cans
    (83, "beer can"),
    (191, "can/tin can"),
    (689, "milk can"),
    # Cartons
    (219, "carton"),
    # Kettles
    (603, "kettle/boiler"),
    (1068, "teakettle"),
    # Fluid containers (pitchers, jugs, pots, etc.)
    (52, "barrel/cask"),
    (159, "bucket/pail"),
    (201, "canteen"),
    (285, "coffeepot"),
    (325, "cream pitcher"),
    # (343, "cup"),
    # (376, "Dixie cup/paper cup"),
    # (461, "flute glass/champagne flute"),
    # (497, "glass/glass drink container/drinking glass"),
    (601, "keg"),
    (707, "mug"),
    (813, "pitcher/pitcher vessel for liquid/ewer"),
    # (835, "pot"),
    (909, "saltshaker"),
    (933, "shaker"),
    # (951, "shot glass"),
    # (986, "soup bowl"),
    (1058, "tank/tank storage vessel/storage tank"),
    # (1066, "tea bag"),
    # (1067, "teacup"),
    (1069, "teapot"),
    (1136, "urn"),
    # (1158, "washbasin/basin/basin for washing/washbowl/washstand/handbasin"),
    (1162, "water cooler"),
    (1165, "water jug"),
    (1188, "wine bucket/wine cooler"),
    # (1189, "wineglass"),
    # Lids/Caps
    # (203, "bottle cap/cap/cap container lid"),
    # (305, "cork/cork bottle plug/bottle cork"),
    # (307, "corkscrew/bottle screw"),
    # (192, "can opener/tin opener"),
    # # Related accessories
    # (275, "coaster"),
    # (914, "saucer"),
    # (1023, "straw/straw for drinking/drinking straw"),
    # (1030, "sugar bowl"),
]

# Extract just the class IDs for detection filtering
LVIS_DRINK_BEVERAGE_CLASS_IDS = [cat[0] for cat in LVIS_DRINK_BEVERAGE_CATEGORIES]

# Extract just the class names
LVIS_DRINK_BEVERAGE_CLASS_NAMES = [cat[1] for cat in LVIS_DRINK_BEVERAGE_CATEGORIES]
