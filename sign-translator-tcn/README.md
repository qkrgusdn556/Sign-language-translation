# 노트북에서 내려받아 실행

이 폴더에는 현재 웹 프로그램과 학습 데이터·모델이 포함되어 있습니다. `data/`는 Git LFS로 관리합니다. API 키와 Python 설치 라이브러리는 포함하지 않습니다.

Git for Windows(Git LFS 포함)와 Python 3.12 64비트·Python Launcher를 설치한 뒤 아래 순서로 진행하세요.

```powershell
git lfs install
git clone https://github.com/qkrgusdn556/Sign-language-translation.git
cd Sign-language-translation/sign-translator-tcn
git lfs pull
powershell -NoProfile -File .\scripts\setup.ps1
.\run_tcn.bat
```

실시간 화면: http://127.0.0.1:5194/frontend/live.html

YOLO 모델은 `data/yolo-studio/runs` 안 같은 실행의 `fingerspell.onnx`와 `labels.json`을 함께 연결합니다. Gemini 키는 새 컴퓨터의 화면에서 별도로 설정합니다. 추론에 NVIDIA GPU가 필수는 아니며 모델 추가 학습 없이 기존 모델을 사용할 수 있습니다.

GitHub의 Download ZIP은 저장소 설정에 따라 LFS 원본 대신 포인터만 포함할 수 있으므로 위 clone + lfs pull 절차를 사용하세요. 새 노트북의 카메라·처리 속도는 실제 환경에서 확인해야 합니다.

---

# 실시간 수어 인식 실험 프로젝트

실행: **run_tcn.bat** → http://127.0.0.1:5194/

카메라·문장 전용 화면: http://127.0.0.1:5194/frontend/live.html (아바타 제거, 모델 설정은 별도 창)

YOLO 학습실: `powershell -NoProfile -File scripts/start-yolo-studio.ps1` → http://127.0.0.1:5196/

| 폴더 | 내용 |
| --- | --- |
| frontend | 웹 화면, 관절 추적, DTW·지문자 판단, 카메라·문장 전용 화면, YOLO 학습실 화면 |
| backend | 웹 서버, 저장 API, Gemini·SLRT 연결, TCN 실행, 학습실 서버 |
| training | TCN·YOLO 모델 정의 및 학습 작업 |
| tools | 평가, 진단, 다른 PC로 이전하는 백업·복원 도구 |
| scripts | 설치, 서버 실행, API 키 설정 |
| docs | 사용 설명서, 수집 규약, 공동 학습 및 이전 안내 |
| tests | 자동 회귀 테스트 |
| data | 개인 샘플, 영상, 모델 — 기존 위치 유지 |
| vendor | 외부 라이브러리 |
| reports | 평가 결과 |
| logs | 실행 로그 |
| .build | 로컬 검증 산출물 및 변경 전 코드 스냅샷 |

Python 모듈의 공통 경로는 `project_paths.py`에서 관리합니다. 모든 서버와 학습 작업은 현재 작업 디렉터리와 관계없이 이 프로젝트의 `data`를 사용합니다.

- [사용 설명서](docs/USER-GUIDE.md)
- [다른 PC 설치·이전](docs/PORTABLE-SETUP.md)
- [영상 수집 규약](docs/RECORDING-CONTRACT.md)
- [공동 학습](docs/SHARED-LEARNING.md)

백업: `py -3.12 tools/portable_backup.py export sign-translator-transfer.zip`

테스트: `python -m unittest discover -s tests` / `npm test`

기존 `/app.js` 등의 웹 주소는 새 `/frontend/` 주소로 연결됩니다. Python을 직접 실행할 때는 `python backend/web_server.py --port 5194`처럼 새 파일 경로를 사용하세요.

TCN 연속 입력 연결과 실제 연속 수어 인식 정확도는 별개입니다. 현재 모델의 연결 동작·비수어 구간 학습 한계를 고려해야 합니다.
