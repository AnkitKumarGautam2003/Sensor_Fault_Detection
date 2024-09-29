import os
import sys
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import accuracy_score

from src.Constant import *
from src.logger import logging
from src.Utils.main_utils import MainUtils
from src.exception import CustomException

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder=os.path.join(artifact_folder)
    tained_model_path=os.path.join(artifact_folder,"model.pkl")
    expected_accuracy=0.45
    model_config_file_path=os.path.join('config','model.yaml')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
        self.utils=MainUtils()
        
        self.models={
            'XGBClassifier':XGBClassifier(),
            "GradientBoostingClassifier":GradientBoostingClassifier(),
            "RandomForestClassifier":RandomForestClassifier(),
            "SVC":SVC()
        }
        
    def evaluate_model(self,x,y,models):
        try:
            x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
            
            report={}
            
            for i in range(len(list(models))):
                model = list(models.values())[i]
                
                model.fit(x_train,y_train)
                y_train_prid=model.predict(x_train)
                
                y_test_prid=model.predict(x_test)
                
                train_model_score=accuracy_score(y_train,y_train_prid)
                test_model_score=accuracy_score(y_test,y_test_prid)
                
                report[list(models.keys())[i]]=test_model_score
                
            return report
        
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def get_best_model(self,
                       x_train:np.array,
                       y_train:np.array,
                       x_test:np.arange,
                       y_test:np.array):
        
        try:
            
            model_report:dict=self.evaluate_model(
                x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test,models=self.models)
            
            
            print(model_report)
        
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name =list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                
            ]
        
            best_model_object=self.models[best_model_name]
            
            return best_model_score,best_model_name,best_model_object
        
        except Exception as e:
            raise CustomException (e,sys)
        
        
    def finetune_best_model(self,
                            best_model_obj:object,
                            best_model_name,
                            x_train,
                            y_train) -> object:
        
        try:
            model_param_grid=self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
            
            grid_searc=GridSearchCV(
                best_model_obj,param_grid=model_param_grid,cv=5,n_jobs=-1,verbose=1
            )
            
            grid_searc.fit(x_train,y_train)
            
            best_perams=grid_searc.best_params_
            
            print("best params are:", best_perams)   
            
            finetuned_model=best_model_obj.set_params(**best_perams)
            
            return finetuned_model    
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_model_trainer(self,train_arry,test_array):
        try:
            logging.info(f"splitting training and testing input and target feature")
            
            x_train,y_train,x_test,y_test=(
                train_arry[:,:-1],
                train_arry[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            ) 
            logging.info(f"Extracting model config file path")
            
            model_report:dict=self.evaluate_model(x=x_train,y=y_train,models=self.models)
            
            # To get best model name from dict
            best_model_score=max(sorted(model_report.values()))
            
            #To get best model name from dict
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model=self.models[best_model_name]
            
            best_model=self.finetune_best_model(
                best_model_name=best_model_name,
                best_model_obj=best_model,
                x_train=x_train,
                y_train=y_train
                
            )
            
            best_model.fit(x_train,y_train)
            y_prid=best_model.predict(x_test)
            
            best_model_score=accuracy_score(y_test,y_prid)
            
            print(f"Best model name{best_model_name} and score:{best_model_score}")   
            
            if best_model_score < 0.5: 
                raise Exception("No test model found with an accuracy greater then threshold 0.6")
            
            logging.info(f"Best found model on both training and testing dataset")
            
            logging.info(
                
                f"Saving model at path: {self.model_trainer_config.tained_model_path}"
                
            )
            
            os.makedirs(os.path.dirname(self.model_trainer_config.tained_model_path),exist_ok=True)
            
            self.utils.save_object(
                file_path=self.model_trainer_config.tained_model_path,
                obj=best_model
            )
        
            return self.model_trainer_config.tained_model_path
    
        except Exception as e:
            raise CustomException(e,sys)
        
        
    