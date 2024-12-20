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
            
             
class EntityDataset:
    '''
    Dataset class for entity recognition
    '''
    def __init__(self, text, entity):
        self.texts = text
        self.entity = entity
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN 
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,item):
        text = self.texts[item]
        entity = self.entity[item]

        ids = []
        target_entity = []
        for i,word in enumerate(text):
            token_ids = self.tokenizer.encode(word,
                                          add_special_tokens=False)
            word_piece_entity = [entity[i]]*len(token_ids)

            ids.extend(token_ids)
            target_entity.extend(word_piece_entity)

        #adujst to ids and target_entity to max_len, as max_len special_tokens inclusive
        ids = ids[:self.max_len-2]
        target_entity = target_entity[:self.max_len-2]
        #add cls token
        ids = [101] + ids + [102]
        target_entity = [0] + target_entity + [0]
        
        #create mask and token_type_id
        mask,token_type_id = [1]*len(ids),[0]*len(ids)

        #padding
        padding_len = self.max_len - len(ids)
        
        ids = ids + ([0] * padding_len)
        target_entity = target_entity + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_id = token_type_id + ([0] * padding_len)

        return {
            'ids': torch.tensor(ids,dtype=torch.long),
            'target_entity': torch.tensor(target_entity,dtype=torch.long),
            'mask': torch.tensor(mask,dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_id,dtype=torch.long)
        } 