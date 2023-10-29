from nerBERT.config.configuration import ConfigurationManager
from nerBERT.components.data_transformation import DataTransformation
from nerBERT.logging import logger


class DataTransformationtionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()

        