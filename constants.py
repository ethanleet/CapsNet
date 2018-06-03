import os
# Directory to save models
SAVE_DIR = "saved_models"
# Directory to save plots
PLOT_DIR = "plots"
# Directory to save logs
LOG_DIR = "logs"
# Directory to save images
IMAGES_SAVE_DIR = "reconstructions"
# Directory to save smallNorb Dataset
SMALL_NORB_PATH = os.path.join("datasets", "smallNORB")

# Default values for command arguments
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_ALPHA = 0.0005 # Scaling factor for reconstruction loss
DEFAULT_DATASET = "small_norb" # 'mnist', 'small_norb'
DEFAULT_DECODER = "FC" # 'FC' or 'Conv'
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 300 # DEFAULT_EPOCHS = 300
DEFAULT_USE_GPU = True
DEFAULT_DISPLAY_STEP = 469 # Interval between two stats saved
