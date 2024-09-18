import shutil
import os
import sys
import pandas as pd
import pickle 

from flask import request


from src.logger import logging
from src.exception import CustomException
from src.Constant import *
from src.Utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class PredictionPipelineConfig:
    predection_output_dirname:str ="predictions"
    predection_file_name:str="predection_file.csv"
    model_file_path:str=os.path.join(artifact_folder,"model.pkl")
    preprocessor_path:str=os.path.join(artifact_folder,"preprocessor.pkl")
    predection_file_path:str=os.path.join(predection_output_dirname,predection_file_name)
    
class PredectionPipeline:
    def __init__(self,request:request):
        
        self.request=request
        self.utils=MainUtils()
        self.predection_pipeline_config=PredictionPipelineConfig()
    
    def save_input_file(self) -> str:
        try:
            pred_file_input_dir="predection_artifacets"
            os.makedirs(pred_file_input_dir,exist_ok=True)
            
            input_csv_file=self.request.file['file']
            pred_file_path=os.path.join(pred_file_input_dir,input_csv_file.filename)
            
            input_csv_file.save(pred_file_path)
            
            return pred_file_path
        except Exception as e:
            raise CustomException(e,sys)
    
    def predict(self,features):
        try:
            model=self.utils.load_object(self.predection_pipeline_config.model_file_path)
            preprocessor=self.utils.load_object(file_path=self.predection_pipeline_config.preprocessor_path)
            
            transformen_x=preprocessor.transform(features)
            preds=model.predict(transformen_x)
            
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    
    def get_pred_dataframe(self,input_dataframe_path:pd.DataFrame):
        
        try:
            prediction_column_name:str=TARGET_COLUMN
            input_dataframe:pd.DataFrame=pd.read_csv(input_dataframe_path)
            input_dataframe=input_dataframe.drop(columns='Unnamed: 0') if "Unnamed: 0" in input_dataframe.columns else input_dataframe
            
            predictions=self.predict(input_dataframe)
            
            input_dataframe[prediction_column_name]=[pred for pred in predictions]
            
            target_column_maping={0:'bad',1:'good'}
            
            input_dataframe[prediction_column_name]=input_dataframe[prediction_column_name].map(target_column_maping)
            os.makedirs(self.predection_pipeline_config.predection_output_dirname,exist_ok=True)
            input_dataframe.to_csv(self.predection_pipeline_config.predection_file_path,index=False)
            
            logging.info("predictions completed")
        
        except Exception as e:
            raise CustomException (e,sys) from e
        
    def run_pipeline(self):
        try:
            input_csv_path=self.save_input_file()
            self.get_pred_dataframe(input_csv_path)
            
            return self.predection_pipeline_config
        except Exception as e:
            raise CustomException(e,sys)
        
        
        
        
        
            
            
        
        
                    
            