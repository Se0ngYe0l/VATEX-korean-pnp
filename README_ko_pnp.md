# 한국어 Pick+Place VATEX — 학습 & 인퍼런스 가이드

VATEX(WACV 2025 referring segmentation 모델)를 한국어 입력 전용으로 확장하고,
로봇 pick-and-place 작업을 위해 "집을 물건(pick)"과 "놓을 위치/물건(place)"을
**동시에** 분할하도록 만든 작업 기록입니다. 원본 VATEX 사용법은
[README.md](README.md)를 참고하고, 이 문서는 이 레포에 추가된
한국어/pick+place 관련 기능만 다룹니다.

> **중요한 제약**: 명령(인퍼런스 입력)은 절대 실시간 번역하지 않습니다.
> 번역은 학습 데이터를 만들 때(NLLB로 오프라인 1회) 사용했을 뿐, 실제
> 추론 시점에는 한국어 문장이 번역 없이 다국어 CLIP 텍스트 인코더에
> 그대로 들어갑니다.

## 다운로드

- **체크포인트**: [huggingface.co/oSYPark/VATEX-korean-pnp](https://huggingface.co/oSYPark/VATEX-korean-pnp)
  (`roborefit_finetune_ko/model_final.pth`, `roborefit_pnp_finetune_ko/model_final.pth`)
- **OCID-Ref 한국어 번역 어노테이션**: [huggingface.co/datasets/oSYPark/ocidref-ko](https://huggingface.co/datasets/oSYPark/ocidref-ko)
  (JSON만 제공 — 원본 이미지는 라이선스 문제로 재배포하지 않으니 공식 OCID
  페이지에서 따로 받아야 함)
- RoboRefit, RoboGround 데이터는 각 원 저작자의 공식 배포처를 이용하세요
  (라이선스 문제로 이 레포/HF에서는 재배포하지 않습니다).

---

## 1. 환경

- conda 환경 이름: `vatex` (Python 3.9.25)
- 노트북 인퍼런스를 쓰려면 `ipykernel`이 설치돼 있어야 하고, 커널 이름은
  `vatex`로 등록돼 있습니다 (`Python (vatex)`로 표시됨). 새 환경에서라면:
  ```bash
  conda run -n vatex pip install ipykernel
  conda run -n vatex python -m ipykernel install --user --name vatex --display-name "Python (vatex)"
  ```

---

## 2. 무엇이 바뀌었나 (원본 VATEX 대비)

| 영역 | 변경 내용 |
|---|---|
| CLIP 백본 | `openai/clip-vit-base-patch16` → 다국어 `xlm-roberta-base-ViT-B-32` (LAION5B `laion5b_s13b_b90k`, open_clip) |
| 명사구 추출 | 영어 spaCy 의존 파서 → Kiwi 기반 한국어 형태소 분석 (`vatex/utils/noun_phrase_ko.py`) |
| 학습 데이터 | 영어 expression에 `expressions_ko` 필드를 추가 (NLLB-200으로 오프라인 1회 번역) |
| 설정 플래그 | `cfg.MODEL.MULTILINGUAL_CLIP.*`, `cfg.INPUT.USE_KOREAN_EXPRESSIONS` |
| pick+place | RoboGround(RoboCasa 시뮬레이션)에서 pick/place 이중 마스크 데이터를 직접 추출해 조인트 학습 |

영어 전용 경로(원본 코드)는 그대로 남아 있고, 위 플래그들이 꺼져 있으면
원본과 동일하게 동작합니다.

---

## 3. 데이터 준비 (이미 완료되어 있음 — 재현 시에만 필요)

1. **RoboRefit 한국어 번역** (실제 촬영 사진 기반 referring segmentation):
   `utils/translate_expressions_ko.py` — NLLB-200-distilled-1.3B + 카테고리
   글로서리(`utils/ko_category_glossary.py`)로 오역(예: cone→콩고) 교정 후
   `instances_roborefit_{train,testA,testB}_ko.json`에 `expressions_ko` 필드로 저장.

2. **RoboGround PnP 추출** (RoboCasa 시뮬레이션, pick+place 이중 마스크):
   `utils/extract_roboground_pnp.py` — hdf5 데모 파일에서 첫 프레임 RGB +
   segmentation mask(1=pick, 2=place) + `ep_meta`의 언어 지시문을 직접 추출.
   `utils/translate_roboground_pnp.py` — pick 짧은 표현만 NLLB로 번역하고,
   place 명사는 고정 글로서리(`PLACE_INFO`)를 사용, 전체 문장은 번역 없이
   템플릿(`"{pick}{particle} 집어서 {place}에 {verb}"`)으로 직접 조립.

3. **데이터셋 등록**: `vatex/data/datasets/builtin.py`에
   `roborefit_*_ko`, `roboground_pnp_{train,test}_ko` 로 등록되어 있음.

---

## 4. 학습

체크포인트는 다음 순서로 이어서(warm-start) 학습되었습니다:

```
results/roborefit_finetune_fixed/model_final.pth   (영어 RoboRefit 파인튜닝, 원본)
        │  MODEL.WEIGHTS로 warm-start, resizer/clip_sim_resizer만 새로 초기화
        ▼
results/roborefit_finetune_ko/model_final.pth       (한국어 RoboRefit 전용, mIoU 90.58%)
        │  MODEL.WEIGHTS로 warm-start
        ▼
results/roborefit_pnp_finetune_ko/model_final.pth   (RoboRefit+RoboGround PnP 조인트, PnP mIoU 61.64%)
```

### Step 1 — 한국어 RoboRefit 단독 파인튜닝

```bash
python train_net_video.py \
    --config-file configs/roborefit/swin/swin_base_ko.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS results/roborefit_finetune_fixed/model_final.pth \
    OUTPUT_DIR results/roborefit_finetune_ko
```

### Step 2 — RoboRefit + RoboGround PnP 조인트 파인튜닝 (pick+place)

```bash
python train_net_video.py \
    --config-file configs/roborefit/swin/swin_base_ko_pnp.yaml \
    --num-gpus 1 \
    MODEL.WEIGHTS results/roborefit_finetune_ko/model_final.pth \
    OUTPUT_DIR results/roborefit_pnp_finetune_ko
```

- 중간에 끊기면 `--resume` 플래그를 추가해 마지막 체크포인트부터 재개합니다.
- `swin_base_ko.yaml`의 `SOLVER.IMS_PER_BATCH=2`는 VRAM 과부하(24GB GPU 기준
  98.6% 사용 → 컴퓨터 불안정)를 겪은 뒤 줄인 값입니다. 더 큰 GPU라면
  배치를 올리고 `BASE_LR`도 비례해서 올리는 게 낫습니다.
- `CHECKPOINT_PERIOD: 500`로 짧게 잡혀 있는 것도 같은 이유(중간에 재부팅된
  적 있음)로, 안정적인 환경이라면 늘려도 됩니다.

### 진행 상황 확인
```bash
bash utils/check_ko_training.sh
```
iteration/ETA/loss/GPU 사용량/체크포인트 목록/로그 tail을 한번에 보여줍니다.

---

## 5. 평가

```bash
python train_net_video.py \
    --config-file configs/roborefit/swin/swin_base_ko_pnp.yaml \
    --num-gpus 1 --eval-only \
    MODEL.WEIGHTS results/roborefit_pnp_finetune_ko/model_final.pth \
    DATASETS.TEST '("roboground_pnp_test_ko",)' \
    OUTPUT_DIR results/roborefit_pnp_finetune_ko
```

RoboRefit 단독 성능이 조인트 학습 후에도 유지되는지(catastrophic forgetting
여부)는 `DATASETS.TEST '("roborefit_testA_ko",)'`로 바꿔서 같은 방식으로
확인할 수 있습니다 — 아직 실행하지 않은 검증입니다.

---

## 6. 인퍼런스 (실사진 + 한국어 문장 테스트)

### 핵심: 압축 문장은 반드시 pick/place로 분리 후 넣어야 함

VATEX는 "표현 1개 → 마스크 1개"인 **단일 타겟** 모델입니다. 학습 데이터를
보면 pick 어노테이션과 place 어노테이션 둘 다 전체 문장
(`"갈색 물체를 집어서 접시에 놓아줘"` 같은)이 **완전히 동일한 문자열**로
들어있는데, 이 문장을 모델에 그대로 두 번 넣으면 입력이 같으니 **항상
똑같은 마스크만 반환**됩니다(직접 확인: `identical: True`). 그래서
`vatex/utils/split_pnp_ko.py`의 `split_pnp_instruction_ko()`가 Kiwi
품사 태그로 압축 문장을 pick(을/를로 끝나는 구)과 place(에로 끝나는 구,
"위/안/옆" 같은 상대위치 명사는 제거)로 먼저 분리하고, 그 두 구를
**각각 따로** 모델에 넣습니다.

```python
from vatex.utils.split_pnp_ko import split_pnp_instruction_ko

split_pnp_instruction_ko("캔을 노란 바구니에 넣어줘")
# -> ("캔", "노란 바구니")
```

### 방법 A — 노트북 (추천, 반복 테스트에 편함)

`notebooks/infer_real_photo.ipynb`

1. 위 두 셀(설정 / 모델 로드)을 한 번만 실행 (커널: `Python (vatex)`).
2. 맨 아래 셀의 `IMAGE_PATH`, `INSTRUCTION`만 바꿔가며 반복 실행 →
   원본 이미지와 pick(빨강)/place(초록) 예측이 바로 인라인으로 표시됩니다.

### 방법 B — 스크립트 (파일로 저장, 배치 처리용)

`utils/infer_real_photos.py` — 상단의 `SAMPLES` 리스트에
`(이미지 경로, 한국어 지시문)` 튜플을 넣고 실행:

```bash
cd /home/seongyeol/VATEX
conda run -n vatex python utils/infer_real_photos.py
```

결과는 `results/real_photo_visualizations/real_XX.png`에 저장됩니다.

### 인퍼런스 시 주의할 점

- `height`/`width`는 반드시 **augmentation(리사이즈) 이전** 원본 이미지
  크기를 넘겨야 합니다. 리사이즈 후 크기를 넘기면 예측 마스크가 원본
  해상도 사진 위에서 어긋나고 프레임 밖으로 번지는 버그가 발생합니다
  (두 인퍼런스 코드 모두 이미 올바르게 처리돼 있음, 새로 짤 때만 주의).

---

## 7. 현재 성능 / 알려진 한계

| 체크포인트 | 평가셋 | mIoU |
|---|---|---|
| `roborefit_finetune_ko` | RoboRefit testA (한국어) | 90.58% |
| `roborefit_pnp_finetune_ko` | RoboGround PnP test (한국어) | 61.64% |
| (참고) 영어 베이스라인 | RoboRefit testA | 92.3% |

**실사진 테스트에서 드러난 한계**: 학습 분포(RoboCasa 시뮬레이션 식료품,
RoboRefit 장난감)와 가까운 물체(예: 일반적인 캔, 바구니)는 실사진에서도
pick+place 둘 다 정확했지만, 반짝이는 작은 금속 캔(스팸 통조림)처럼 학습
때 본 적 없는 형태/재질은 pick 인식이 실패했습니다. 즉 현재 병목은
아키텍처보다 **실제 환경 물체·조명에 대한 도메인 격차**입니다.

**개선 아이디어** (우선순위 순, 자세한 내용은 작업 기록 참고):
1. `configs/roborefit/swin/swin_base_ko_pnp.yaml`의 `DATASETS.TRAIN`에
   실사진 기반 `ocidref_train_ko`가 빠져있음 — 추가만 해도 비용 없이
   실물 다양성 보강 가능.
2. 반사/하이라이트 계열 photometric augmentation 강화.
3. 실제 배포 환경(책상/카메라/조명)에서 pick+place 사진 50~100장 수집 후
   이어서 파인튜닝 — 가장 확실하지만 라벨링 비용 발생.

---

## 8. 주요 파일 레퍼런스

| 파일 | 역할 |
|---|---|
| `vatex/vatex_model.py` | 다국어 CLIP 조건부 경로 추가 (`_forward_clip_text_ko` 등) |
| `vatex/utils/noun_phrase_ko.py` | 한국어 단일 명사구 추출 (Kiwi 기반) |
| `vatex/utils/split_pnp_ko.py` | 압축 pick+place 문장 → (pick구, place구) 분리 |
| `vatex/config.py` | `MODEL.MULTILINGUAL_CLIP.*`, `INPUT.USE_KOREAN_EXPRESSIONS` 설정 추가 |
| `vatex/data/dataset_mapper.py` | `expressions_ko` 필드 사용 분기 |
| `vatex/data/datasets/builtin.py` | `*_ko`, `roboground_pnp_*` 데이터셋 등록 |
| `utils/ko_category_glossary.py` | 카테고리명 번역 글로서리 + 조사(을/를 등) 교정 |
| `utils/translate_expressions_ko.py` | RoboRefit/OCID-Ref 한국어 번역 |
| `utils/extract_roboground_pnp.py` | RoboGround hdf5 → pick+place COCO 데이터 추출 |
| `utils/translate_roboground_pnp.py` | RoboGround PnP 한국어 번역 + 문장 템플릿 조립 |
| `utils/visualize_ko_samples.py` | RoboRefit 한국어 예측 시각화 |
| `utils/visualize_pnp_samples.py` | RoboGround PnP 예측 시각화 (pick+place 동시) |
| `utils/infer_real_photos.py` | 실사진 인퍼런스 스크립트 |
| `notebooks/infer_real_photo.ipynb` | 실사진 인퍼런스 노트북 (반복 테스트용) |
| `utils/check_ko_training.sh` | 학습 진행 상황 확인 bash 스크립트 |
