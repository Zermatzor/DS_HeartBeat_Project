import os
from dataclasses import dataclass

# Path to input folder (relative to this file)
INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "../..", "input")

@dataclass(frozen=True)
class DatasetFiles:
    train: str = "mitbih_train.csv"
    test: str = "mitbih_test.csv"
    abnormal: str = "ptbdb_abnormal.csv"
    normal: str = "ptbdb_normal.csv"

# Create instance
FILES = DatasetFiles()

# Optional: full paths dictionary
FULL_PATHS = {
    "train": os.path.join(INPUT_FOLDER, FILES.train),
    "test": os.path.join(INPUT_FOLDER, FILES.test),
    "abnormal": os.path.join(INPUT_FOLDER, FILES.abnormal),
    "normal": os.path.join(INPUT_FOLDER, FILES.normal)
}
