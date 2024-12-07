from typing import List, Dict, Any, Sequence, Optional, Tuple
from experiments.evaluator.base  import BaseEvaluator
from experiments.utils import registry
# from src.models.test_llava1 import LLaVA
from experiments.datasets.mimic import Mimic
from experiments import ImageTxtSample, _OutputType
import random
from typing import Tuple, Callable
import numpy as np
import re
import torch
import json
import os


@registry.register_evaluator()
class OpenEndedVQAEvaluator(BaseEvaluator):

    def __init__(self):
        pass




@registry.register_evaluator()
class FairnessEvaluator(BaseEvaluator):
    """
    Evaluator for testing fairness of LLAVA models.
    """

    evaluator_ids: List[str] = ['fairness_eval']

    def __init__(self, evaluator_id: str, metrics_cfg: Dict[str, Any],template:Callable) -> None:
        """
        Initializes the FairnessEvaluator.
        
        Arguments:
            evaluator_id: Identifier for the evaluator
            model_id: The model identifier (e.g., llava-v1.5-7b)
            metrics_cfg: A dictionary of fairness metrics configurations
            device: Device to run the model on (default is "cuda")
        """
        super().__init__(evaluator_id, metrics_cfg)
        self.prompt_template = template
        self.raceList  =["African","American","Asian","Caucasian","Hispanic","Native American","Unknown"]
        self.genderList=["Male","Female","Unknown"]
    
    def contains_yes_or_no(self,sentence):
        if re.search(r'\byes\b', sentence, re.IGNORECASE):
            return 1  
        elif re.search(r'\bno\b', sentence, re.IGNORECASE):
            return 0  
        else:
            return 1
            #raise Exception(f"No label detected : {sentence}")
            
    def setup(self, model,dataset: Mimic) -> None:
        """
        Set the model for evaluation.

        Arguments:
            model: The model to be evaluated.
        """
        self.model = model
        self.dataset = dataset
        print("model and dataset set up")
        
    # def gen_prompt(self, info: dict) -> list[str]:
    #     """
    #     Generate a prompt based on demographic info, medical question, and answer type specification.

    #     Arguments:
    #         info: A dictionary with demographic information and medical question data.

    #     Returns:
    #         A formatted prompt string based on the provided info.
    #     """
    #     # Extract demographic and medical question information
    #     gender = "Male" if info["extra"]["gender"] == "M" else "Female"
    #     race = info["extra"]["race"]
    #     medical_question = info["text"]

    #     # Call the template function to generate the prompt
    #     prompt = self.prompt_template(gender,race,medical_question)
        
    #     return prompt
    
    def comb_eval(self):
        dataList=[]
        dataTemplate={
            "question":None,
            "gender":None,
            "race":None,
            "expected_answer":None,
            "predicted_answer":None
        }
        for idx in random.sample(range(len(self.dataset)), 10):
            question = self.dataset[idx]["text"]
            for r in self.raceList:
                for g in self.genderList:
                    response = self.process(g,r,question,self.dataset[idx]["image_path"])
                    singledata = dataTemplate.copy()
                    singledata["question"] = question
                    singledata["gender"] = g
                    singledata["race"]= r
                    singledata["expected_answer"] = self.dataset[idx]["target"]
                    singledata["predicted_answer"] = "Yes" if self.contains_yes_or_no(response) else "No"  
                    dataList.append(singledata)
        return dataList

    def fair_eval(self) -> Dict[str, Any]:
        # processed_results = []  #This will store the results from model
        preds  = []
        labels = []
        extras = []
        for idx in random.sample(range(len(self.dataset)), 10):
            entry = self.dataset[idx]
            assert type(entry) == ImageTxtSample

            response = self.process(entry["extra"]["gender"],entry["extra"]["race"],entry["text"],entry["image_path"])
            
            preds.append(response)
            labels.append(entry["target"])
            extras.append(entry["extra"]["gender"])
    
        processed_preds = np.array([self.contains_yes_or_no(s) for s in preds ])
        processed_labels= np.array([self.contains_yes_or_no(s) for s in labels])
        processed_extras= np.array([0 if s=="F" else 1         for s in extras])
        
        P_f, P_m = self.metrics["statistical_parity"](processed_preds, processed_extras)
        TPR_f, TPR_m = self.metrics["equal_opportunity"](processed_labels, processed_preds, processed_extras)
        (TPR_f, FPR_f), (TPR_m, FPR_m) = self.metrics["equalized_odds"](processed_labels,processed_preds, processed_extras)
        accuracy_f, accuracy_m = self.metrics["overall_accuracy_equality"](processed_labels, processed_preds, processed_extras)
        error_ratio_f, error_ratio_m = self.metrics["treatment_equality"](processed_labels, processed_preds, processed_extras)
        eval_results = {
            "statistical_parity" : {
                "P_f": P_f,  
                "P_m": P_m  
            },
            "equal_opportunity": {
                "TPR_f": TPR_f, 
                "TPR_m": TPR_m   
            },
            "equalized_odds": {
                "TPR_f": TPR_f, 
                "FPR_f": FPR_f,
                "TPR_m": TPR_m,
                "FPR_m": FPR_m   
            },
            "overall_accuracy_equality": {
                "accuracy_f": accuracy_f, 
                "accuracy_m": accuracy_m 
            },
            "treatment_equality": {
                "error_ratio_f": error_ratio_f, 
                "error_ratio_m": error_ratio_m 
            }
        }
        return eval_results
    
    def process(self, gender, race, question, imagepath ) :
        gnerated_prompt = self.prompt_template(gender, race, question)
        response = self.model.ask(image_path = imagepath, question = gnerated_prompt)
        return response


