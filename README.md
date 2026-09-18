# 나라장터 입찰공고 법령 위반 모니터링

입찰공고·첨부·등록 메타데이터를 읽고 24개 항목의 위반 여부와 원문 근거를 출력하는 고정 Gemma 추론 파이프라인입니다. 모델을 학습하지 않고 전처리, 판정 기준, 사례 검색, 제약 디코딩을 구성합니다.

최신 제출은 **v5-F**입니다. 제출 번호 **92665**, 접수 시각 **2026-09-18 22:53:15**, 접수 확인 당시 **채점 중**입니다. v4는 **0.3104368405 / 51분 42초**를 기록했습니다. 이 저장소는 실시간 점수 현황을 제공하지 않습니다.

v4 코드는 `submission/`, v5 코드는 `submission_v5/`에 보존했습니다. v5는 B그룹(v10~18)에 사실 8개를 먼저 출력하는 변경만 적용한 실험입니다. 실제 dev F1 비교는 아직 실행하지 못했으며, 자격 문맥 보충·사례 검색 개선의 개별 실험과 결합도 미실행입니다. [v5 검증 기록](docs/v5-validation.md)과 [맥북 이전 안내](docs/mac-migration.md)를 참고하십시오.

## 구조

```text
submission/
  script.py             # 고정 모델 추론, 그룹 병합, CSV 및 실행 감사
  context.py            # 항목별 원문 구절 선택과 정확한 offset 보존
  model/README.md       # 별도로 준비할 로컬 정적 자산 안내
submission_v5/          # 최신 v5-F 제출 코드와 별도 로컬 정적 자산
tools/                  # 자산 생성, 검증, 토큰 검사, ZIP 생성
tests/                  # 원문 정합성·그룹별 근거 범위·실패 처리 테스트
docs/                   # 설계 및 검증 요약
artifacts/              # 제출 접수 기록과 파일 해시 (ZIP 제외)
```

## v4 처리 흐름

1. 공고 하나의 원문과 메타데이터를 읽습니다.
2. A(v1~v8), B(v10~v18), C(v9, v19~v24) 그룹별 관련 구절을 선택합니다.
3. 해당 항목의 적용대상·위반 조건·예외·정상 대비 기준과 관련 부분라벨 사례를 제공합니다.
4. 고정 Gemma를 그룹별로 호출하고 JSON 스키마로 출력 형식을 제한합니다.
5. 각 그룹의 근거 번호를 해당 문서의 원문으로 복원한 뒤 24항목을 합칩니다.
6. `id,v1..v24,e1..e24`의 49열 CSV를 저장합니다. 정상 및 부재탐지 5항목의 근거는 빈칸입니다.

모델은 한 번만 적재하며 공고 간 예측·통계를 공유하지 않습니다. v4 입력 예산은 그룹별 6,000 / 6,500 / 6,000토큰, 생성 상한은 그룹별 512토큰입니다. v5는 v4 원문·사례 선택 후 B 사실 지시를 추가합니다(별도 입력 여유 1,536토큰). B 생성 상한은 768, A/C는 512입니다.

## 공개 범위

설명회 자료·정리본, 베이스라인 노트북·코드, 대회 원본 데이터, 가상환경, 모델·토크나이저 캐시, 생성 사례·검토 자료, 예측 결과 및 제출 ZIP은 포함하지 않습니다. 원본 작업 폴더의 자료는 삭제하지 않았습니다.

대회는 참가자 생성 라벨·가공 자료의 공유 경로를 코드 공유 게시판으로 제한합니다. 따라서 이 공개 저장소에는 실행 코드와 자료를 포함하지 않는 요약을 게시합니다. **clone만으로 정적 자산과 결과까지 재현되지는 않습니다.** 자산은 허용된 경로로 별도 확보해야 합니다. [공식 규칙](https://dacon.io/competitions/official/236754/overview/rules)

## 로컬 준비

Python 3.11 이상 환경에서 프로젝트 루트를 기준으로 실행합니다. 실제 평가 서버는 별도 고정 환경을 사용합니다.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

대회에서 내려받은 자료를 다음 위치에 배치합니다. `open/baseline/`은 Git에 포함하지 않습니다.

```text
open/dev.jsonl.gz
open/dev_labels.csv
open/data/항목표.json
open/data/법령패키지/...
open/data/test.jsonl.gz
```

이어서 [정적 자산 안내](submission/model/README.md)에 따라 `submission/model/`의 로컬 파일을 복원합니다. 테스트는 해당 대회 자료와 자산이 있어야 실행됩니다.

```bash
python tools/make_dev_folds.py
python -m pytest tests -q
python submission/script.py --mock --input open/dev.jsonl.gz --data-dir open/data --output analysis/mock_dev
python tools/validate_submission.py analysis/mock_dev/mock_submission.csv --input open/dev.jsonl.gz
python submission_v5/script.py --mock --input open/dev.jsonl.gz --data-dir open/data --output analysis/mock_v5
```

`--mock`는 모델과 토큰 수를 모의 처리하는 연결 검사입니다. 점수·실제 토큰 예산·GPU 실행시간을 검증하지 않습니다.

## 실제 모델 실행 및 패키징

평가 환경에 준비된 모델 경로를 `PPS_MODEL_DIR`, 입력 자료 폴더를 `PPS_DATA_DIR`, 출력 폴더를 `PPS_OUTPUT_DIR`로 설정합니다.

```bash
python submission/script.py
python tools/package_submission.py
python tools/package_submission.py --source submission_v5 --name submit_v5.zip
```

패키징 기본 결과는 로컬 `artifacts/submit_v4.zip`, 두 번째 명령은 `artifacts/submit_v5.zip`입니다. v5 실제 실행 진입점은 `python submission_v5/script.py`입니다. 두 버전 모두 NVIDIA/vLLM 평가 환경용이며 macOS MLX 실행기는 아직 없습니다. 기본 서버 패키지를 사용하며 이 저장소의 개발 의존성 파일을 제출 환경에 설치하지 않습니다. 실행 중 모델 다운로드·외부 API 호출은 하지 않습니다.

`--single-pass`는 24항목 동시 판정 비교용, `--examples 0`은 사례 없는 비교용입니다. `--exclude-ids`는 검색에서 제외할 ID들의 JSON 배열 파일을 받습니다.

## 검증 및 한계

- 로컬 전체 테스트 **51개 통과**: 기존 44개와 v5 사실·근거 범위·재시도 검사 7개.
- 패키지 모의 실행 **200건 형식 오류 0**.
- 고정 토크나이저 **600요청** 검사: 제공 정답 근거 **54/54**, 추가 검토 구절 **12/12** 보존.
- 입력 토큰 총량 v4 **3,486,071** → v5 **3,593,871**(약 **3.09% 증가**). v4/v5의 선택 원문·사례·user 메시지는 600요청 모두 동일합니다.
- v4 서버 점수와 시간은 확인했지만 실제 고정 모델의 로컬 dev F1과 v5 서버 결과는 미확인입니다.

근거 보존율은 분류 성능이 아닙니다. v5 점수 상승이나 2시간 내 실행을 보장하지 않습니다. 검토 주체는 AI 에이전트이며 사람 법률전문가의 검수로 표시하지 않습니다. [기존 v4 설계 기록](docs/design-and-validation.md)은 v4 접수 당시의 기록이며, 최신 상태는 위 내용과 v5 검증 기록을 기준으로 합니다.

## 개발 도구

| 도구 | 용도 |
|---|---|
| `build_assets.py` | 제공 dev 근거로 기존 양성 사례 자산 생성; 기존 자산을 덮어쓰므로 백업 후 사용 |
| `build_v4_examples.py` | 로컬 `review_specs.json`에 명시한 검토 구절을 검증·변환 |
| `make_dev_folds.py` | 공개 dev 내부 진단용 분할 생성 |
| `profile_v4.py` | 로컬 `analysis/fixed_tokenizer/`의 고정 토크나이저로 실제 입력 검사 |
| `profile_v5.py` | v4/v5 원문·사례 동일성 및 v5 토큰·근거 보존 검사; 모델 추론 없음 |
| `profile_prompts.py` | 단일 패스 입력 진단용 과거 도구 |
| `measure_context_coverage.py` | 별도 `open/baseline/script.py`를 준비했을 때만 가능한 과거 비교 도구 |
| `validate_submission.py` | CSV 검증 및 선택적 항목별 F1 계산 |
| `package_submission.py` | 명시적 허용 목록으로 제출 ZIP 생성 |

대회 자료와 제3자 모델에는 각 제공자의 조건이 적용됩니다. 저장소의 MIT 라이선스가 해당 자료의 재배포 권한을 부여하지는 않습니다.
