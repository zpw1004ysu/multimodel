# config.py

# Random seed
SEED = 220

# Hyperparameters
LEARNING_RATE = 0.002
EPOCHS = 400
HIDDEN_SIZE = 128
THRESHOLD = 0.2
DROPOUT = 0.7
OUTPUT_SIZE = 9

# Paths
TRAIN_PATH = "dataset/train_data.csv"
TEST_PATH = "dataset/blind_data.csv"
OUTPUT_PREDICTIONS = "outputs/predicted_tree.txt"
OUTPUT_CONF_MATRIX = "outputs/confusion_matrix_tree.pkl"
