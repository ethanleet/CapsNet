import os
# Directory to save models
SAVE_DIR = "saved_models"
# Directory to save plots
PLOT_DIR = "plots"
# Directory to save logs
LOG_DIR = "logs"
# Directory to save options
OPTIONS_DIR = "options"
# Directory to save images
IMAGES_SAVE_DIR = "reconstructions"
# Directory to save smallNorb Dataset
SMALL_NORB_PATH = os.path.join("datasets", "smallNORB")

# Default values for command arguments
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_ANNEAL_TEMPERATURE = 8 # Anneal Alpha
DEFAULT_ALPHA = 0.0005 # Scaling factor for reconstruction loss
DEFAULT_DATASET = "small_norb" # 'mnist', 'small_norb'
DEFAULT_DECODER = "FC" # 'FC' or 'Conv'
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 300 # DEFAULT_EPOCHS = 300
DEFAULT_USE_GPU = True
DEFAULT_ROUTING_ITERATIONS = 3
DEFAULT_VALIDATION_SIZE = 1000

# Random seed for validation split
VALIDATION_SEED = 889256487