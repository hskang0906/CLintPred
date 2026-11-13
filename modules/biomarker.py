from typing import Any

import os
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoConfig, RobertaModel, BertModel

from clearance_ft.modules.dataloader import BiomakerDataModule
from clearance_ft.modules.LayerModule import deleteEncodingLayers


class BiomarkerModel(pl.LightningModule):
    def __init__(self, lr, dropout, layer_features, drug_model_name, prot_model_name, d_pretrained=True, p_pretrained=True):
        super().__init__()
        self.lr = lr

        #-- Pretrained Model Setting
        self.drug_config = AutoConfig.from_pretrained(drug_model_name)
        self.d_model = RobertaModel(self.drug_config) if not d_pretrained else RobertaModel.from_pretrained(drug_model_name, num_labels=2)

        self.prot_config = AutoConfig.from_pretrained(prot_model_name)
        self.p_model = BertModel(self.prot_config) if not p_pretrained else BertModel.from_pretrained(prot_model_name)
        self.p_model = deleteEncodingLayers(self.p_model, 18)

        #-- Decoder Layer Setting
        layers = []
        firstfeature = self.d_model.config.hidden_size + self.p_model.config.hidden_size
        for feature_idx in range(0, len(layer_features) - 1):
            layers.append(nn.Linear(firstfeature, layer_features[feature_idx]))
            firstfeature = layer_features[feature_idx]

            layers.append(nn.Tanh()) if feature_idx is len(layer_features)-2 else layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
    
        layers.append(nn.Linear(firstfeature, layer_features[-1]))
        self.decoder = nn.Sequential(*layers)

        # self.save_hyperparameters()

    def load_state_dict(self, state_dict, strict: bool = True):
        # Remove unexpected keys from the state_dict
        keys_to_remove = ['d_model.embeddings.position_ids', 'p_model.embeddings.position_ids']
        for key in keys_to_remove:
            if key in state_dict:
                del state_dict[key]
        super().load_state_dict(state_dict, strict)

    def forward(self, drug_inputs, prot_inputs):
        d_outputs = self.d_model(drug_inputs['input_ids'], drug_inputs['attention_mask'])
        p_outputs = self.p_model(prot_inputs['input_ids'], prot_inputs['attention_mask'])

        outs = torch.cat((d_outputs.last_hidden_state[:, 0], p_outputs.last_hidden_state[:, 0]), dim=1)
        outs = self.decoder(outs)
        outs = outs.squeeze(dim=1)  

        return outs

    def attention_output(self, drug_inputs, prot_inputs):
        d_outputs = self.d_model(drug_inputs['input_ids'], drug_inputs['attention_mask'])
        p_outputs = self.p_model(prot_inputs['input_ids'], prot_inputs['attention_mask'])

        outs = torch.cat((d_outputs.last_hidden_state[:, 0], p_outputs.last_hidden_state[:, 0]), dim=1)
        outs = self.decoder(outs)        

        return d_outputs['attentions'], p_outputs['attentions'], outs
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return {"ids": batch[0],  "outputs": self(batch[1], batch[2])}
        


class BindingAffinityModel(pl.LightningModule):
    def __init__(self, lr, dropout, layer_features, drug_model_name, prot_model_name, d_pretrained=True, p_pretrained=True):
        super().__init__()
        self.lr = lr

        #-- Pretrained Model Setting
        self.drug_config = AutoConfig.from_pretrained(drug_model_name)
        self.d_model = RobertaModel(self.drug_config) if not d_pretrained else RobertaModel.from_pretrained(drug_model_name, num_labels=2)

        self.prot_config = AutoConfig.from_pretrained(prot_model_name)
        self.p_model = BertModel(self.prot_config) if not p_pretrained else BertModel.from_pretrained(prot_model_name)
        self.p_model = deleteEncodingLayers(self.p_model, 18)

        #-- Decoder Layer Setting
        layers = []
        firstfeature = self.d_model.config.hidden_size + self.p_model.config.hidden_size

        for feature_idx in range(0, len(layer_features) - 1):
            layers.append(nn.Linear(firstfeature, layer_features[feature_idx]))
            firstfeature = layer_features[feature_idx]
            layers.append(nn.ReLU())  
            layers.append(nn.Dropout(dropout))
    
        layers.append(nn.Linear(firstfeature, layer_features[-1]))
        self.decoder = nn.Sequential(*layers)

        self.save_hyperparameters()

    def forward(self, drug_inputs, prot_inputs):
        d_outputs = self.d_model(drug_inputs['input_ids'], drug_inputs['attention_mask'])
        p_outputs = self.p_model(prot_inputs['input_ids'], prot_inputs['attention_mask'])

        outs = torch.cat((d_outputs.last_hidden_state[:, 0], p_outputs.last_hidden_state[:, 0]), dim=1)
        outs = self.decoder(outs)  
        outs = outs.squeeze(dim=1)    

        return outs

    def attention_output(self, drug_inputs, prot_inputs):
        d_outputs = self.d_model(drug_inputs['input_ids'], drug_inputs['attention_mask'])
        p_outputs = self.p_model(prot_inputs['input_ids'], prot_inputs['attention_mask'])

        outs = torch.cat((d_outputs.last_hidden_state[:, 0], p_outputs.last_hidden_state[:, 0]), dim=1)
        outs = self.decoder(outs)        

        return d_outputs['attentions'], p_outputs['attentions'], outs
    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        return {"ids": batch[0],  "outputs": self(batch[1], batch[2])}
    


def predict_Biomarker(model_name, chem_tokenizer, prot_tokenizer, df_chem, df_prot, pred_file):
    checkpoint_name = str(model_name).lower()

    ## 1. Model 불러오기
    ckpt_file = os.path.join("./checkpoint", f"{checkpoint_name}.ckpt")
    if checkpoint_name == "regression_bindingDB":
        model = BindingAffinityModel.load_from_checkpoint(ckpt_file)
    else:
        model = BiomarkerModel.load_from_checkpoint(ckpt_file)
    
    train_obj = chem_tokenizer(np.array(df_chem["SMILES"]).tolist(), padding='max_length', 
                            max_length=config.chem_max, truncation=True, return_tensors="pt")
    prot_obj = prot_tokenizer(np.array(df_chem["aas"]).tolist(), padding='max_length', 
                            max_length=config.prot_max, truncation=True, return_tensors="pt")
    
    affinity_dm = BiomakerDataModule(train_obj, prot_obj, config)

    ## 2. biomarker Trainer 설정
    trainer = pl.Trainer(accelerator='gpu',
                        strategy='dp', 
                        devices=config.gpus)

    pred_train = trainer.predict(model, datamodule=affinity_dm)

    ## 3. 예측 결과 Output-Amino Acid Binding 결합 결과 저장
    affinity_columns = pd.DataFrame(list(df_prot["name"])).reset_index(drop=True)

    ids = np.array(torch.cat([output['ids'] for output in pred_train], dim=0)).tolist()
    outputs = np.array(torch.cat([output['outputs'] for output in pred_train], dim=0)).tolist()
    pred_list = {"ids": ids, "outputs":outputs}

    start_idx = 0
    for idx in range(1, len(pred_list['ids'])):
        if pred_list["ids"][idx-1] != pred_list["ids"][idx]:
            affinity_dataset = [affinities for affinities in pred_list["outputs"][start_idx : idx]] 
            affinity_columns.loc[len(affinity_columns)] = affinity_dataset

            start_idx = idx
    
    affinity_dataset = [affinities for affinities in pred_list["outputs"][start_idx :]] 
    affinity_columns.loc[len(affinity_columns)] = affinity_dataset

    ## 4. 예측 결과 저장
    affinities = pd.concat([df_chem["SMILES"], affinity_columns], axis=1)
    affinities.to_csv(pred_file, index=False)

    return affinities

