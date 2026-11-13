import os
import json

import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from argparse import ArgumentParser

from clearance_ft.utils.utils import DictX
from clearance_ft.utils.preprocess import load_protdata, get_affinitydata, norm_dataset
from clearance_ft.modules.dataloader import BiomakerDataModule, clearanceDatamodule
from clearance_ft.modules.biomarker import BiomarkerModel
from clearance_ft.modules.clearance import clearanceMLPModel

# 외부에서 사용 가능한 함수들
__all__ = ['load_clearance_models', 'predict_clearance', 'run_clearance']


def _load_json_config(path, required_fields, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} configuration not found: {path}")
    try:
        with open(path, 'r') as file:
            config = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid {label} configuration: {path}") from exc
    missing = sorted(set(required_fields) - set(config))
    if missing:
        raise ValueError(f"{label} configuration is missing: {', '.join(missing)}")
    return config


def _load_pt_state_dict(path, label):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} checkpoint not found: {path}")
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError(f"{label} checkpoint must contain a PyTorch state dict: {path}")
    return state_dict


def concat_predData(pred_result, affinity_columns):
    df_result = pd.DataFrame(columns=affinity_columns)

    ids = np.array(torch.cat([output['ids'] for output in pred_result], dim=0)).tolist()
    outputs = np.array(torch.cat([output['outputs'] for output in pred_result], dim=0)).tolist()
    pred_list = {"ids": ids, "outputs":outputs}

    start_idx = 0
    for idx in range(1, len(pred_list['ids'])):
        if pred_list["ids"][idx-1] != pred_list["ids"][idx]:
            affinity_dataset = [affinities for affinities in pred_list["outputs"][start_idx : idx]] 
            df_result.loc[len(df_result)] = affinity_dataset
            start_idx = idx
    
    affinity_dataset = [affinities for affinities in pred_list["outputs"][start_idx :]] 
    df_result.loc[len(df_result)] = affinity_dataset

    return df_result


def predict_BindingAffinity(model, chem_tokenizer, prot_tokenizer, df_chem, df_prot, result_file, config):

    trainer = pl.Trainer(accelerator='gpu', devices=config.gpus)

    train_obj = chem_tokenizer(np.array(df_chem["SMILES"]).tolist(), padding='max_length', 
                            max_length=config.chem_max, truncation=True, return_tensors="pt")
    prot_obj = prot_tokenizer(np.array(df_prot["aas"]).tolist(), padding='max_length', 
                            max_length=config.prot_max, truncation=True, return_tensors="pt")
    
    affinity_dm = BiomakerDataModule(train_obj, prot_obj, config)
    pred_train = trainer.predict(model, datamodule=affinity_dm)

    affinity_columns = list(df_prot["name"])
    affinities = concat_predData(pred_train, affinity_columns)
    affinities = pd.concat([df_chem["SMILES"], affinities], axis=1)

    affinities.to_csv(result_file, index=False)

    return affinities


def load_clearance_models():
    """Clearance 모델 및 관련 데이터 로드 (한 번만 수행)"""

    # config 파일 경로를 절대 경로로 지정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, 'config', '0_predict.json')
    
    config_dict = _load_json_config(
        config_path,
        {
            'chem_model', 'prot_model', 'chem_max', 'prot_max', 'prot_file',
            'extend_protType', 'feature_type', 'scale', 'augmentation', 'batch_size', 'gpus'
        },
        'Clearance inference',
    )
    config = DictX(config_dict)

    # Tokenizer 로드
    chem_tokenizer = AutoTokenizer.from_pretrained(config.chem_model)
    prot_tokenizer = AutoTokenizer.from_pretrained(config.prot_model)
    
    # Protein 데이터 로드 (상대 경로를 절대 경로로 변환)
    prot_file_path = os.path.join(base_dir, config.prot_file.lstrip('./'))
    if not os.path.isfile(prot_file_path):
        raise FileNotFoundError(f"Clearance protein reference data not found: {prot_file_path}")
    df_protData = load_protdata(prot_file_path, config.extend_protType)
    
    # Biomarker 모델 로드
    biomarker_checkpoint = os.path.join(base_dir, 'checkpoint', 'biomarker.pt')
    biomarker_hparams_path = os.path.join(base_dir, 'checkpoint', 'biomarker_hparams.json')
    
    biomarker_state_dict = _load_pt_state_dict(biomarker_checkpoint, 'Biomarker')
    biomarker_hparams = _load_json_config(
        biomarker_hparams_path,
        {
            'lr', 'dropout', 'layer_features', 'drug_model_name',
            'prot_model_name', 'd_pretrained', 'p_pretrained', 'initialize_pretrained'
        },
        'Biomarker',
    )
    if biomarker_hparams['drug_model_name'] != config.chem_model:
        raise ValueError('Biomarker drug model does not match Clearance inference config')
    if biomarker_hparams['prot_model_name'] != config.prot_model:
        raise ValueError('Biomarker protein model does not match Clearance inference config')
    initialize_biomarker = bool(biomarker_hparams['initialize_pretrained'])
    
    biomarker_model = BiomarkerModel(
        lr=biomarker_hparams['lr'], dropout=biomarker_hparams['dropout'],
        layer_features=biomarker_hparams['layer_features'],
        drug_model_name=biomarker_hparams['drug_model_name'],
        prot_model_name=biomarker_hparams['prot_model_name'],
        d_pretrained=initialize_biomarker and biomarker_hparams['d_pretrained'],
        p_pretrained=initialize_biomarker and biomarker_hparams['p_pretrained']
    )
    biomarker_model.load_state_dict(biomarker_state_dict, strict=True)
    biomarker_model.eval()
    
    # Clearance 모델 로드
    clearance_checkpoint = os.path.join(base_dir, 'checkpoint', 'main.pt')
    clearance_hparams_path = os.path.join(base_dir, 'checkpoint', 'main_hparams.json')
    
    clearance_state_dict = _load_pt_state_dict(clearance_checkpoint, 'Clearance')
    clearance_hparams = _load_json_config(
        clearance_hparams_path,
        {
            'lr', 'dropout', 'model_name', 'feature_length', 'act_func',
            'initialize_pretrained'
        },
        'Clearance',
    )
    if clearance_hparams['model_name'] != config.chem_model:
        raise ValueError('Clearance model does not match inference config')
    
    clearance_model = clearanceMLPModel(
        lr=clearance_hparams['lr'], dropout=clearance_hparams['dropout'],
        model_name=clearance_hparams['model_name'], 
        feature_length=clearance_hparams['feature_length'],
        act_func=clearance_hparams['act_func'],
        initialize_pretrained=bool(clearance_hparams['initialize_pretrained'])
    )
    clearance_model.load_state_dict(clearance_state_dict, strict=True)
    clearance_model.eval()
    
    models = {
        'chem_tokenizer': chem_tokenizer,
        'prot_tokenizer': prot_tokenizer,
        'df_protData': df_protData,
        'biomarker_model': biomarker_model,
        'clearance_model': clearance_model,
        'config': config
    }
    
    return models


def predict_clearance(SMILES, preloaded_models=None, config=None, output_dir=None):
    # 모델 로드 (없으면 새로 로드)
    if preloaded_models is None:
        if config is None:
            raise ValueError("config 또는 preloaded_models가 필요합니다")
        models = load_clearance_models(config)
    else:
        models = preloaded_models
    
    chem_tokenizer = models['chem_tokenizer']
    prot_tokenizer = models['prot_tokenizer']
    df_protData = models['df_protData']
    biomarker_model = models['biomarker_model']
    clearance_model = models['clearance_model']
    config = models['config']
    
    # 입력 데이터 준비 - 항상 입력된 SMILES만 사용 (파일 무시)
    df_predict = pd.DataFrame({
        "SMILES": [SMILES],
        "Clint": [0.0]  # 더미 값 (나중에 예측됨)
    })
    
    # Affinity 예측 - SMILES마다 다른 캐시 파일 사용
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir:
        affinity_folder = os.path.join(output_dir, "clearance_affinity")
    else:
        affinity_folder = os.path.join(base_dir, config.affinity_folder.lstrip('./'))
    
    # SMILES 해시를 사용하여 고유한 캐시 파일 생성
    import hashlib
    smiles_hash = hashlib.md5(SMILES.encode()).hexdigest()[:10]
    result_file = os.path.join(affinity_folder, f"pred_result_{smiles_hash}")
    
    if not os.path.exists(result_file):
        os.makedirs(affinity_folder, exist_ok=True)
        predict_affinity = predict_BindingAffinity(
            biomarker_model, chem_tokenizer, prot_tokenizer, df_predict, df_protData, result_file, config
        )
    else:
        predict_affinity = pd.read_csv(result_file)
    
    # 데이터 전처리
    df_predictData, _, df_sampledpredict = norm_dataset(
        df_predict, config.feature_type, config.scale, config.augmentation
    )
    df_predictAff = get_affinitydata(df_sampledpredict, predict_affinity, config.augmentation)
    df_predictfeatures = (
        pd.concat([df_predictAff.iloc[:, 1:], df_predict.iloc[:, 2:]], axis=1) 
        if config.feature_type.lower() != "default" else df_predictAff.iloc[:, 1:]
    )
    
    # 예측 수행 (PyTorch 직접 사용 - PyTorch Lightning 미사용)
    clearance_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    clearance_model.to(device)
    
    predictions = []
    
    with torch.no_grad():
        # SMILES 토크나이징
        enc = chem_tokenizer(
            df_predict["SMILES"].tolist(), 
            padding='max_length', 
            max_length=config.chem_max, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Features를 tensor로 변환
        if len(df_predictfeatures) > 0:
            features_tensor = torch.tensor(
                df_predictfeatures.values, 
                dtype=torch.float32
            ).to(device)
        else:
            features_tensor = torch.tensor([], dtype=torch.float32).to(device)
        
        # 배치 처리 (메모리 효율성)
        batch_size = config.batch_size
        for i in range(0, len(df_predict), batch_size):
            end_idx = min(i + batch_size, len(df_predict))
            
            # SMILES input 준비
            smiles_input = {
                'input_ids': enc['input_ids'][i:end_idx].to(device),
                'attention_mask': enc['attention_mask'][i:end_idx].to(device)
            }
            
            # Features 준비
            if len(df_predictfeatures) > 0:
                batch_features = [features_tensor[i:end_idx]]
            else:
                batch_features = []
            
            # 모델 추론
            outputs = clearance_model(smiles_input, batch_features)
            predictions.extend(outputs.cpu().numpy().flatten().tolist())
    
    # 결과 정리
    df_result = pd.DataFrame({
        "SMILES": df_predict["SMILES"], 
        "predict": predictions[:len(df_predict)]
    })
    
    # df_predictData의 Clint를 예측 결과로 업데이트
    df_predictData['Clint'] = predictions[:len(df_predict)]
    
    return df_result, df_predictData['Clint'].values


# 외부에서 사용 가능한 함수
def run_clearance(input_smiles:str, preloaded_models, output_dir=None):
    models = preloaded_models
    predicted_clearance, normalized_clint = predict_clearance(
        SMILES=input_smiles,
        preloaded_models=models,
        output_dir=output_dir,
    )
    df_result = predicted_clearance.copy()
    df_result['predict'] = np.expm1(df_result['predict'])
    df_result['Clint'] = np.expm1(normalized_clint)
    return df_result    


def main(config):
    """Clearance 예측 실행 및 결과 저장"""
    try:
        config = DictX(config)
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        
        # 모델 로드
        models = load_clearance_models(config)
        
        # 예측 수행
        predicted_clearance, normalized_clint = predict_clearance(SMILES=None, preloaded_models=models)
        
        # 결과 역변환 및 저장
        os.makedirs('./results', exist_ok=True)
        
        df_result = predicted_clearance.copy()
        df_result['predict'] = np.expm1(df_result['predict'])  # log1p 역변환
        df_result['Clint'] = np.expm1(normalized_clint)
        df_result.to_csv('./results/predict.csv', index=False)
        
        print(f"Prediction completed: {len(df_result)} compounds")
        print(f"Results saved to ./results/predict.csv")
        
        return df_result
    
    except Exception as e:
        print(f"Error: {e}")
        raise



if __name__ == "__main__":
    parser = ArgumentParser(description="Clearance Prediction")
    
    # 모델 설정
    parser.add_argument("--chem_model", type=str, 
                        default="seyonec/PubChem10M_SMILES_BPE_450k",
                        help="Chemistry transformer model")
    parser.add_argument("--prot_model", type=str, 
                        default="Rostlab/prot_bert_bfd",
                        help="Protein transformer model")
    parser.add_argument("--chem_max", type=int, default=510,
                        help="Max length for chemistry tokenizer")
    parser.add_argument("--prot_max", type=int, default=545,
                        help="Max length for protein tokenizer")
    
    # 데이터 경로
    parser.add_argument("--input_file", type=str, 
                        default="./dataset/k_melloddy/test.csv",
                        help="Input SMILES CSV file")
    parser.add_argument("--prot_file", type=str, 
                        default="./dataset/liver_aas.csv",
                        help="Protein data file")
    parser.add_argument("--affinity_folder", type=str, 
                        default="./dataset/affinity_fin/predict/",
                        help="Affinity cache folder")
    parser.add_argument("--pred_result", type=str, 
                        default="./pred_result",
                        help="Affinity result file name")
    
    # 전처리 설정
    parser.add_argument("--extend_protType", type=bool, default=True,
                        help="Extend protein type")
    parser.add_argument("--scale", type=bool, default=False,
                        help="Data scaling")
    parser.add_argument("--augmentation", type=bool, default=False,
                        help="Data augmentation")
    parser.add_argument("--feature_type", type=str, default="default",
                        help="Feature type")
    
    # 실행 설정
    parser.add_argument("--num_seed", type=int, default=120,
                        help="Random seed")
    parser.add_argument("--gpus", type=str, default="0,",
                        help="GPU IDs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers")
    
    args = parser.parse_args()
    config = vars(args)
    main(config)
