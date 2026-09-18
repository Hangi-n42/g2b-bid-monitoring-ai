# 맥북으로 작업 이전

2026-09-18 기준. M4 Max 36GB의 실제 모델 실행은 아직 검증하지 않았다. 현재 Git의 두 추론 코드는 vLLM용이며 MLX용 모델 실행 경로는 다음 작업이다. 저장소를 복제하는 것만으로 맥에서 실제 추론이 시작되지는 않는다.

## Git 외에 별도로 복사할 파일

개인 장치 간 전송으로 기존 작업 폴더의 상대경로를 그대로 복원한다. 아래 자료는 공개 Git에 추가하지 않는다.

| 자료 | Windows 원본 위치 | 맥 저장소 내 위치 / 용도 |
|---|---|---|
| 대회 원본 | `open/` | `open/` 전체. dev·정답·항목표·법령·데이터 스키마·테스트 자료 보존 |
| v4 정적 자산 | `submission/model/*.json` | 동일 경로, 네 JSON |
| v5 정적 자산 | `submission_v5/model/*.json` | 동일 경로, 네 JSON |
| 비공개 검토 선택 목록 | `.publish/g2b-bid-monitoring-ai/submission/model/review_specs.json` | **`submission/model/review_specs.json`**. 공개용 검토 사례 생성기에 필요 |
| 고정 토크나이저 | `analysis/fixed_tokenizer/` | 동일 경로. 토큰 비교 재현에 필요 |
| 비교·분석 기록 | `analysis/` | 동일 경로. 원문·예시 포함 기록이 있으므로 비공개 유지 |
| 제출 원본 | `artifacts/submit_v4.zip`, `artifacts/submit_v5.zip`, `artifacts/submit_89803.zip` | 동일 경로. 실제 제출물과 v3/v4/v5 비교 보존 |
| 대회 안내·설명회·baseline | 기존 Markdown·노트북·설명회 폴더, `open/baseline/` | 참고용으로 별도 보관. Git에서는 계속 제외 |

`analysis/` 전체를 옮기면 토크나이저도 함께 포함된다. `artifacts/` 전체를 보관해도 되지만 재추출 폴더 등 중복이 있다. 최소 실행에는 `open/`, 두 버전의 모델 폴더 JSON, 토큰 검사에는 `analysis/fixed_tokenizer/`가 필요하다. 생성 사례를 재작성하는 경우에만 `review_specs.json`이 필요하지만 작업 연속성을 위해 함께 보존한다.

Git은 소스 줄바꿈을 LF로 정규화한다. 코드 동작이 같아도 맥에서 새로 압축한 ZIP 해시는 실제 제출 당시의 해시와 달라질 수 있다. 과거 제출물의 바이트 단위 재현에는 원본 ZIP을 보관하고, 새 패키징은 새 manifest로 관리한다.

## 옮기지 않아도 되는 것

- `.venv/`, `venv/`, `.tools/python/`: Windows 환경이므로 맥에서 새 가상환경을 만든다.
- `__pycache__/`, `.pytest_cache/`: 재생성 가능한 캐시.
- `.publish/` 전체: Git에서 다시 clone하되 위의 `review_specs.json`만 별도 확보한다.
- 계정 토큰·`.env`·브라우저 프로필: 복사 묶음에 넣지 말고 맥에서 필요한 계정만 다시 인증한다.

확인했던 Windows Gemma 캐시에는 설정·토크나이저만 있고 모델 가중치는 없었다. 캐시를 복사해도 실제 모델이 생기지 않는다. 고정 리비전의 가중치 확보와 MLX 변환·실행 지원을 별도로 준비해야 한다.

## 맥에서 우선 실행할 검사

```bash
git clone https://github.com/Hangi-n42/g2b-bid-monitoring-ai.git
cd g2b-bid-monitoring-ai
# 위 비공개 자료를 동일 상대경로에 복원한 다음 실행
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python tools/make_dev_folds.py
python -m pytest tests -q
python submission_v5/script.py --mock --input open/dev.jsonl.gz --data-dir open/data --output analysis/mac_mock_v5
python tools/validate_submission.py analysis/mac_mock_v5/mock_submission.csv --input open/dev.jsonl.gz
python tools/profile_v5.py
```

위 절차는 형식·원문·토큰 검증이며 실제 F1 측정이 아니다. 개발 의존성에는 MLX/vLLM이 없다. Python 패키지 설치와 실제 맥 실행 성공은 별도로 확인한다.

## 다음 실제 모델 작업

1. 대회 고정 리비전 `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`를 기준으로 모델 변환·로딩을 검증한다.
2. MLX 실행기를 별도 구현하되 원문 선택·사례·가이드·채점 로직을 유지한다. 구조화 출력 제약의 차이도 점검한다.
3. 8bit, 요청 1개부터 실제 제출 입력 길이까지 메모리·스왑·생성 길이·처리시간을 측정한다. 36GB에서 안정적으로 처리된다고 아직 보장할 수 없다.
4. 동일 맥 설정에서 v4와 v5-F의 dev 예측을 확보해 항목별 TP·FP·FN을 비교한다. 같은 8bit라도 서버의 per-channel INT8과 MLX 양자화는 같지 않으므로 서버 점수·시간 재현으로 표현하지 않는다.
5. 자격 문맥 보충과 사례 검색 개선은 분리해서 평가하고 효과가 확인된 후보만 결합한다.
