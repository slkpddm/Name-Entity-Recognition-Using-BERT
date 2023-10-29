import os
import datasets
from nerBERT.utils.common import get_size
from src.nerBERT.logging import logger
import datasets
from nerBERT.entity import *

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    
    def data_loading(self):
           data = datasets.load_dataset(self.config.source_dataset)
           data.save_to_disk(self.config.dataset)
           if (get_size(Path(self.config.dataset))!=0):
              logger.info("Data loaded into the dataset folder:{}".format(self.config.dataset))
           else:
              logger.info("Data not loaded into the dataset folder:{}".format(self.config.dataset))
                   

