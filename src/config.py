from pathlib import Path


class Config():
    PROJECT_PATH = Path(__file__).parent.parent
    DATASET_PATH = PROJECT_PATH / 'datasets'
    SEED = 12345


config = Config()
