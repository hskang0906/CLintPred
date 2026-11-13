import os, copy
import pandas as pd
import numpy as np
from argparse import ArgumentParser

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from transformers import AutoTokenizer

import wandb

from clearance_ft.utils.utils import DictX, load_hparams, save_result, generate_combinations
from clearance_ft.utils.preprocess import load_protdata, norm_dataset, get_rdkitlabel, get_affinitydata

from clearance_ft.modules.dataloader import BiomakerDataModule, clearanceDatamodule
from clearance_ft.modules.biomarker import predict_Biomarker
from clearance_ft.modules.clearance import clearanceMLPModel, clearanceDecoderModel, clearanceEncoderModel
    
    
def train_function(config, logger):
    pl.seed_everything(seed=config.num_seed)

    ## 1. Biomarker 모델 결과 저장 경로 설정
    biomarker_name = str(config.checkpoint_name).lower()
    affinity_path = os.path.join(config.affinity_folder, biomarker_name, config.train_dataType)
    train_affinityFile, test_affinityFile = f"{affinity_path}/{config.train_file}", f"{affinity_path}/{config.test_file}"

    if not os.path.exists(train_affinityFile) or not os.path.exists(test_affinityFile):
        if not os.path.exists(affinity_path):
            os.makedirs(affinity_path)
    
    ## 2. Tokenizer 설정
    chem_tokenizer = AutoTokenizer.from_pretrained(config.chem_model)
    prot_tokenizer = AutoTokenizer.from_pretrained(config.prot_model)

    ## 3. 단백질 데이터셋 로드
    df_protData = load_protdata(config.prot_file, config.extend_protType)
    
    ## 4. 화합물 데이터셋 경로 설정
    train_path =  os.path.join(config.data_folder, config.train_dataType, config.train_file)

    ## 5. 학습 및 평가 데이터셋 생성
    df_train = pd.read_csv(train_path)
    df_test = df_train.sample(frac=config.test_rate, random_state=config.num_seed)
    df_train = df_train.drop(df_test.index)
    df_train, df_test = df_train.reset_index(drop=True), df_test.reset_index(drop=True)
    
    ## 6. Biomarker 모델 예측 결과 생성
    train_affinity = pd.read_csv(train_affinityFile) if os.path.exists(train_affinityFile) else predict_Biomarker(biomarker_name, 
                                                                                                                chem_tokenizer, prot_tokenizer,
                                                                                                                df_train, df_protData,
                                                                                                                train_affinityFile, config)
    test_affinity = pd.read_csv(test_affinityFile) if os.path.exists(test_affinityFile) else predict_Biomarker(biomarker_name, 
                                                                                                                chem_tokenizer, prot_tokenizer,
                                                                                                                df_test, df_protData,
                                                                                                                test_affinityFile, config)
        
    testData_normtype = "all" if config.sampling_type.lower() == "all" else config.feature_type

    ## 7. 훈련 데이터셋 정규화, rdkit 범위 설정
    df_trainData, train_scaler, df_sampledtrain = norm_dataset(df_train, config.feature_type, config.scale, config.augmentation)
    df_trainAff = get_affinitydata(df_sampledtrain, train_affinity, config.augmentation)
    df_trainfeatures = pd.concat([df_trainAff.iloc[:, 1:], df_trainData.iloc[:, 2:]], axis=1) if config.feature_type.lower() != "default" else df_trainAff.iloc[:, 1:]

    df_testData, test_scaler, df_sampledtest = norm_dataset(df_test, testData_normtype, config.scale)
    df_testAff = get_affinitydata(df_sampledtest, test_affinity)
    df_testRdkitLabel = get_rdkitlabel(df_sampledtest)
    df_testfeatures = pd.concat([df_testAff.iloc[:, 1:], df_testData.iloc[:, 2:]], axis=1) if config.sampling_type.lower() != "default" else df_testAff.iloc[:, 1:]
    
    train_labels = df_trainData["Clint"]
    test_labels = df_testData["Clint"]
    
    option_name = ''
    if config.extend_protType == False:
        option_name += "Extend"
    if config.augmentation == False:
        option_name += "Aug"
    if option_name != '':
        option_name = "_Non"+option_name

    checkpoint_callback = ModelCheckpoint(filename=f"{config.model_type}_{config.feature_type}_{config.num_seed}{option_name}", 
                                          save_top_k=1, 
                                          monitor="valid_MSE", 
                                          mode="min")
    trainer = pl.Trainer(
        devices=args.gpus,
        max_epochs=config.max_epoch,
        logger = logger,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        accelerator='gpu',
        strategy='dp'           
    )

    ## -- train, test column name setting -- ##
    train_datacols = list(df_trainData.columns[1:])
    test_datacols = list(df_testData.columns[1:])
    
    ## -- feature column name length -- ##
    feature_length = len(df_trainfeatures.columns)    
    
    ## -- datamodule -- ##
    datamodule = clearanceDatamodule(chem_tokenizer, config, 
                                        df_trainData=df_trainData, 
                                        df_trainLabel=train_labels, 
                                        train_scaler=train_scaler,
                                        df_trainfeatures=df_trainfeatures,)
    
    clearance_model = clearanceMLPModel(config.lr, config.dropout, config.chem_model, feature_length,
                                            config.act_func[0])
    trainer.fit(clearance_model, datamodule)
    
    # clearance_model.eval()
    # trainer.test(clearance_model, datamodule)

    ## -- connecting result data to rdkit feature range -- ##
    dict_result = {"SMILES": df_testData["SMILES"], "predict": clearance_model.test_result['preds'], "Clint": clearance_model.test_result['labels']}
    df_result = pd.DataFrame.from_dict(dict_result)

    df_result = pd.concat([df_result, df_testRdkitLabel[df_testRdkitLabel.columns[1:]]], axis=1)

    return df_result, clearance_model.valid_log, clearance_model.test_log, df_trainData, df_testData


def main(config=None):
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        ##-- hyper param config file Load --##
        config = DictX(config)

        if not os.path.exists(config.log_path):
            os.makedirs(config.log_path)

        logger = TensorBoardLogger(save_dir=config.log_path, name=project_name)
        

        outputs_list = list()
        path_name, _ = os.path.splitext(args.config)
        resultData_path = f"{path_name}_Commonsetting/"

        df_result, valid_log, test_log, df_trainData, df_testData = train_function(config, logger)
        outputs_list.append([df_result, valid_log, test_log, df_trainData, df_testData, config])
                
        print("---------------Chart Boundary setting!!--------------------")
        variance_boundary = 0.5
        for index, output in enumerate(outputs_list):
            variance_list =[abs(data - output[0]['Clint'][idx])/output[0]['predict'].mean() for idx, data in enumerate(output[0]['predict'])]
            output[0]["variance"] = variance_list
            
            if index ==  0:
                df_duplPos = output[0][output[0]["variance"] > variance_boundary]
            else:
                duplPos_temp = output[0][output[0]["variance"] > variance_boundary]
                df_duplPos = df_duplPos[df_duplPos["SMILES"].isin(duplPos_temp["SMILES"])]
            

        for output_result, output_devlog, output_testlog, output_traindata, output_testdata, output_options in outputs_list:
            print("---------------Save prediction Results!!--------------------")
            save_result(output_options, resultData_path, output_result, output_devlog, output_testlog, output_traindata, output_testdata, df_duplPos)
    
    except Exception as e:
        print(e)


if __name__ == "__main__":
    #-- wandb Sweep Hyper Param Tuning --##
    parser = ArgumentParser()
    parser.add_argument("--gpus", help="gpus", type=str, default="0,1")
    parser.add_argument("--config", help="config_file", type=str, default="1_config_sweep.json")
    args = parser.parse_args()

    config_file = 'config/1_config_train.json'
    config = load_hparams(config_file)
    project_name = config["name"]
    main(config)
        