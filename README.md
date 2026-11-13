# CLintPred — clearance_ft

화합물 구조와 단백질 정보를 이용한 내재 청소율(간 효소가 약물을 제거하는 능력) 예측 코드.
현재 `clearance_ft` 폴더 기준의 예측·학습 코드 백업.

## 파일 구성

| 경로 | 역할 |
| --- | --- |
| `main_prediction.py` | 모델 불러오기, 단일 화합물 예측, 결과 역변환 |
| `main_trainer.py` | 학습과 평가 결과 저장 흐름 |
| `modules/` | 약물·단백질 모델과 데이터 읽기 |
| `utils/` | 데이터 전처리, 평가 지표, 그림 저장 |
| `config/0_predict.json` | 예측용 모델·입력 길이·실행 설정 |
| `config/1_config_train.json` | 학습 설정 |
| `config/1_config_sweep.json` | 여러 학습 설정 후보 |
| `checkpoint/*_hparams.json` | 저장된 가중치에 맞는 모델 구성 |
| `_data_merging.ipynb` | 데이터 병합 작업 노트 |
| `requirement.txt` | 실행 코드에서 사용하는 외부 패키지 목록 |

기존 구성 유지: `modules/`는 모델, `utils/`는 전처리·평가·그림 저장 역할로 분리.
`checkpoint/`와 `dataset/`는 코드에서 사용하는 상대 경로이므로 실행 전 해당 위치에 별도 파일 준비.

## 설치

내부 코드에서 `clearance_ft` 이름으로 서로를 불러오므로 단독 사용 시에도 해당 이름으로 복제.

```bash
git clone https://github.com/hskang0906/CLintPred.git clearance_ft
cd clearance_ft
python -m pip install -r requirement.txt
```

- 기존 `pytorch-lightning==1.8.4` 유지: 학습 코드에서 `strategy='dp'`(여러 그래픽 처리 장치에 계산을 나누는 방식)와 이전 버전의 학습 종료 처리 방식 사용.
- `torch>=1.13`: `torch.load(..., weights_only=True)`(저장된 가중치만 읽는 옵션) 사용 기준.
- Python·CUDA(그래픽 처리 장치용 계산 환경)·나머지 패키지의 전체 버전 조합은 미검증. `requirement.txt`는 모든 버전을 고정한 환경 파일이 아닌 직접 의존 패키지 목록.
- 새 약물·단백질 결합값 계산 경로는 `accelerator='gpu'`(그래픽 처리 장치 사용)로 고정. 해당 경로 실행 시 호환되는 NVIDIA 장치와 PyTorch 환경 필요.
- Kaleido(그림 파일 생성 도구) 1.x 사용 시 Chrome 또는 Chromium 별도 필요: [공식 그림 저장 안내](https://plotly.com/python/static-image-export/).

## 별도 준비 파일

가중치와 데이터 파일은 저장소에 미포함.

| 위치 | 필요한 내용 |
| --- | --- |
| `checkpoint/biomarker.pt` | 약물·단백질 결합 예측 가중치 |
| `checkpoint/main.pt` | 청소율 예측 가중치 |
| `checkpoint/biomarker_hparams.json` | 결합 예측 모델 구성; 저장소에 포함 |
| `checkpoint/main_hparams.json` | 청소율 모델 구성; 저장소에 포함 |
| `dataset/liver_aas.csv` | 단백질 기준 데이터; `name`, `aas`, `type` 열 필요 |
| `dataset/<train_dataType>/train.csv` | 학습용 화합물 데이터; 학습 시 필요 |

- 가중치·모델 구성·단백질 순서의 일치 필요. 현재 청소율 모델 구성의 `feature_length`(추가 입력값 개수)는 31.
- 화합물 학습 데이터의 기본 열: `SMILES`(분자 구조 문자열), `Clint`(청소율 정답값).
- 추가 입력값 사용 시 `utils/preprocess.py`에 정의된 물성 열도 필요.
- 모델 식별자: `seyonec/PubChem10M_SMILES_BPE_450k`, `Rostlab/prot_bert_bfd`. Hugging Face에서 읽는 토크나이저(문자열을 모델 입력용 숫자로 바꾸는 도구)·모델 설정은 인터넷 연결 또는 기존 로컬 저장본 필요.

## 함수 호출로 예측

위 파일과 실행 환경 준비 후 저장소 폴더에서 실행. 상위 폴더를 Python 검색 경로에 추가하여 내부 `clearance_ft` 참조 지원.

```bash
PYTHONPATH="$(dirname "$PWD")${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
from clearance_ft.main_prediction import load_clearance_models, run_clearance

models = load_clearance_models()
result = run_clearance("CCO", preloaded_models=models, output_dir="./results")
print(result)
PY
```

- `load_clearance_models()`는 `config/0_predict.json`과 모델 구성 파일을 읽으며 인자 없이 호출.
- `run_clearance()`는 `SMILES`(분자 구조 문자열), `predict`(예측 청소율), `Clint`(동일 예측값) 열을 가진 표 반환. 표 파일 저장은 호출 측에서 수행.
- `run_clearance()`에서 `expm1`(로그 변환된 값을 원래 크기로 되돌리는 계산) 적용.
- `output_dir`는 중간 결합 예측값 저장 위치 지정; 최종 표 자동 저장 인자는 아님.

## 현재 실행 경로의 제약

- `main_prediction.py` 직접 실행 경로: 인자를 받지 않는 `load_clearance_models()`에 설정을 전달하고 입력 분자도 `None`으로 전달. 현재 상태에서는 위 함수 호출 경로 사용.
- `main_trainer.py`: 실제 설정 읽기 경로는 `config/1_config_train.json`으로 고정; `--config`는 읽을 설정 파일을 변경하지 않음.
- 학습 코드의 `predict_Biomarker()` 호출 인자 수가 정의와 불일치. 결합 예측값을 새로 생성하는 학습 경로는 수정 필요.
- 학습 후 평가 호출이 주석 처리된 상태에서 평가 결과를 참조하므로 현재 학습 스크립트의 전체 완료는 보장되지 않음.
- 문서 수정 과정에서 패키지 설치·실제 모델 예측·학습은 수행하지 않음. 실행 시간·메모리 사용량도 미검증.

## 백업 제외 범위

- 모델 가중치·직렬화 파일(저장된 모델 객체), 데이터셋(학습·평가용 데이터 파일).
- `fedavg` 관련 파일, `report/`, `reports/`, `log/`, `logs/`, `lightning_logs/`, `results/`.
- 실행 중 생성된 Python 임시 파일과 로컬 환경 변수 파일.
- 제외 규칙: `.gitignore`. 과거 Git 이력에 이미 포함된 데이터는 최신 파일 목록에서 제외되더라도 이전 이력에 잔존.

## 패키지 근거

- [PyTorch 1.13.0 가중치 읽기 구현](https://github.com/pytorch/pytorch/blob/v1.13.0/torch/serialization.py)
- [PyTorch Lightning 1.8.4 공식 의존 패키지](https://github.com/Lightning-AI/lightning/blob/1.8.4/requirements/pytorch/base.txt)
