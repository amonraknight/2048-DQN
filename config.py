# All configurations are here:
# Game settings:
SIZE = 400
GRID_LEN = 4
GRID_PADDING = 10

BACKGROUND_COLOR_GAME = "#92877d"
BACKGROUND_COLOR_CELL_EMPTY = "#9e948a"

BACKGROUND_COLOR_DICT = {
    2: "#eee4da",
    4: "#ede0c8",
    8: "#f2b179",
    16: "#f59563",
    32: "#f67c5f",
    64: "#f65e3b",
    128: "#edcf72",
    256: "#edcc61",
    512: "#edc850",
    1024: "#edc53f",
    2048: "#edc22e",
    4096: "#eee4da",
    8192: "#edc22e",
    16384: "#f2b179",
    32768: "#f59563",
    65536: "#f67c5f",
}

CELL_COLOR_DICT = {
    2: "#776e65",
    4: "#776e65",
    8: "#f9f6f2",
    16: "#f9f6f2",
    32: "#f9f6f2",
    64: "#f9f6f2",
    128: "#f9f6f2",
    256: "#f9f6f2",
    512: "#f9f6f2",
    1024: "#f9f6f2",
    2048: "#f9f6f2",
    4096: "#776e65",
    8192: "#f9f6f2",
    16384: "#776e65",
    32768: "#776e65",
    65536: "#f9f6f2",
}

FONT = ("Verdana", 40, "bold")

KEY_QUIT = "Escape"
KEY_BACK = "b"

KEY_UP = "Up"
KEY_DOWN = "Down"
KEY_LEFT = "Left"
KEY_RIGHT = "Right"

KEY_UP_ALT1 = "w"
KEY_DOWN_ALT1 = "s"
KEY_LEFT_ALT1 = "a"
KEY_RIGHT_ALT1 = "d"

KEY_UP_ALT2 = "i"
KEY_DOWN_ALT2 = "k"
KEY_LEFT_ALT2 = "j"
KEY_RIGHT_ALT2 = "l"

RECUR_DEPTH = 2

ACTION_NUMBERS = {
    0: KEY_UP,
    1: KEY_DOWN,
    2: KEY_LEFT,
    3: KEY_RIGHT
}

# DQN parameters
LEARNING_RATE = 0.0001
GAMMA = 0.99
CAPACITY = 5000
BATCH_SIZE = 128
NUM_EPISODES = 150000
MIDDLE_LAYER_1_SIZE = 1024
MIDDLE_LAYER_2_SIZE = 128
TARGET_NET_UPDATE_INTERVAL = 4
ACCEPT_THRESHOLD = 3
INITIAL_EPSILON = 0.15
TRAIN_INTERVAL = 2

MASK_FINAL_STEPS = False
INVALID_STEP_TOLERATE = 2
INVALID_STEP_SCORE = 0.0

OUTPUT_PATH = '..\\outputs\\'
MODULE_PATH = OUTPUT_PATH + 'module_{}.pt'
MODULE_LIST_RECORD = OUTPUT_PATH + 'MODULE_LIST'
BACKUP_INTERVAL = 200
BACKUP_AMOUNT = 5

# 'merged cells': The sum of the values on all cells merged.
# 'mono-sequential': How many times a mono-sequence is broken.
# 'merged cells & mono-sequential'
REWARD_STRATEGY = 'merged cells & mono-sequential'

# Will skip the actions which is not possible?
SKIP_IMPOSSIBLE_ACTION = False

TEST_ROUND = 100

# The action conversion in all 8 directions.
ACTION_NUMBERS_IN_RESHAPE = {
    0: {
        0: 0,
        1: 1,
        2: 2,
        3: 3
    },
    1: {
        0: 0,
        1: 1,
        2: 3,
        3: 2
    },
    2: {
        0: 1,
        1: 0,
        2: 2,
        3: 3
    },
    3: {
        0: 2,
        1: 3,
        2: 0,
        3: 1
    },
    4: {
        0: 1,
        1: 0,
        2: 3,
        3: 2
    },
    5: {
        0: 2,
        1: 3,
        2: 1,
        3: 0
    },
    6: {
        0: 3,
        1: 2,
        2: 1,
        3: 0
    },
    7: {
        0: 3,
        1: 2,
        2: 0,
        3: 1
    }
}
