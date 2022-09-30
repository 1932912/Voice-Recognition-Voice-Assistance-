import torch
import config
class IntentScenarioDataset:
    '''
    Dataset class for Intent and Scenario Classification
    '''
    def __init__(self,text,intent,scenario,
                 require_text=False):
        self.texts = text 
        self.intent = intent
        self.scenario = scenario

        self.require_text = require_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    