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
       
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,item):
        text = self.texts[item]
        intent = self.intent[item]
        scenario = self.scenario[item]

        ids, mask, token_type_ids = None, None, None
        if self.require_text:
            out = self.tokenizer.encode_plus(
                text,
                None,
                add_special_token=True,
                max_length = self.max_len,
                padding = 'max_length'
            )
                        ids = out['input_ids']
            mask = out['attention_mask']
            token_type_ids = out['token_type_ids']
            return {
                'ids': torch.tensor(ids,dtype=torch.long),
                'target_intent': torch.tensor(intent,dtype=torch.long),
                'target_scenario': torch.tensor(scenario,dtype=torch.long),
                'mask': torch.tensor(mask,dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids,dtype=torch.long)
            }
                   else:
            return {
                'ids': None,
                'target_intent': torch.tensor(intent,dtype=torch.long),
                'target_scenario': torch.tensor(scenario,dtype=torch.long),
                'mask': None,
                'token_type_ids':None
            }
            
  