__author__ = "Igor Kim"
__credits__ = ["Igor Kim"]
__maintainer__ = "Igor Kim"
__email__ = "igor.skh@gmail.com"
__status__ = "Development"
__date__ = "05/2019"
__license__ = "MIT"

from dateutil import parser
import itertools

CHECK_NONE = 0
CHECK_REPLACE = 1
CHECK_DISTANCE = 2

TESSERACT_CONF = '--psm 6'

TRIGGER_WORD = "back"

DEBUG_FOLDER = "build/test"

FOLDERS = [
    "build", "build/images", "build/results", "build/test"
]

# lower case is enough
REPLACE_RULES = {
    "1,-1": ["i", "[", "]", "l", "7", "?", "t"],
    "0,-1": ["o"],
    "q,-2": ["g"],
    "0,": ["0o", "o0", "00", "oo"],
    "q,": ["qg","qq","gg","gq"]
}

# Order matters
EXPECTED_KEYS = {
    "phy_cell_id": {
        "corr": CHECK_DISTANCE,
        "map": int,
        "range": (0, 503)
    },
    "timestamp": {
        "corr": CHECK_DISTANCE,
        "map": parser.parse
    },
    "rsrp0": {
        "corr": CHECK_REPLACE,
        "map": int,
        "range": (-150, -40)
    },
    "rsrp1": {
        "corr": CHECK_REPLACE,
        "map": int,
        "range": (-150, -40)
    },
    "rsrq0": {
        "corr": CHECK_REPLACE,
        "map": int,
        "range": (-51, -1)
    },
    "rsrq1": {
        "corr": CHECK_REPLACE,
        "map": int,
        "range": (-51, -1)
    },
    "sinr0": {
        "corr": CHECK_REPLACE,
        "map": float,
        "range": (-40, 40)
    },
    "sinr1": {
        "corr": CHECK_REPLACE,
        "map": float,
        "range": (-40, 40)
    }
}