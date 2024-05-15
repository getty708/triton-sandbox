from enum import Enum


class ModelServePatern(Enum):
    NO_PIPELINE = 0
    PIPELINE_SIZE_1 = 1
    PIPELINE_SIZE_2 = 2
    PIPELINE_SIZE_3 = 3
    PIPELINE_SIZE_4 = 4
    PIPELINE_SIZE_5 = 5
    SINGLE_MODEL_1 = 10
    SINGLE_MODEL_2 = 20
    SINGLE_MODEL_3 = 30
    SINGLE_MODEL_4 = 40
