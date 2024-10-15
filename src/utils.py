import os 
import sys
import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}
        best_model_name = None
        best_score = float('-inf')

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            #para=param[list(models.keys())[i]]

            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)

            # model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

            # Log the R2 scores for both train and test sets
            logging.info(f"Model: {model_name}")
            logging.info(f"Training R2 Score: {train_model_score}")
            logging.info(f"Test R2 Score: {test_model_score}")
            logging.info("------------------------------")

            report[list(models.keys())[i]] = test_model_score

            # Update the best model if the current model's test score is higher
            if test_model_score > best_score:
                best_score = test_model_score
                best_model_name = model_name
        
            logging.info(f"Best Model: {best_model_name} with Test R2 Score: {best_score}")



            #report[list(models.keys())[i]] = test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)
    
    