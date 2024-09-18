import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformationn import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging

class TrainingPipeline:
    def start_data_ingestion(self):
        try:
            data_ingetaion=DataIngestion()
            feature_store_file_path=data_ingetaion.initiate_data_ingestion() 
            return feature_store_file_path
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def start_data_transformation(self,feature_Store_file_path):
        try:
            data_transformation=DataTransformation(feature_store_file_path=feature_Store_file_path)
            train_arr,test_arr,preprocessor_path=data_transformation.initate_data_transformation
            return train_arr,test_arr,preprocessor_path
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_model_training(self,train_arr,test_arr):
        try:
            model_trainer=ModelTrainer()
            model_score=model_trainer.initiate_model_trainer(train_arr,test_arr)    

            return model_score
        except Exception as e:
            raise CustomException(e,sys)
        
    def run_pipline(self):
        try:
            feature_store_file_path=self.start_data_ingestion()
            
            train_arr,test_arr,preprocessor_path=self.start_data_transformation(feature_store_file_path)
            
            r2square=self.start_model_training(train_arr,test_arr)
            
            print(f"trainging Completed. Trained model score: {r2square}")
        
        except Exception as e:
            raise CustomException (e,sys)                