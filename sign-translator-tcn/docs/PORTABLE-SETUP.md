# 다른 Windows 컴퓨터에서 실행

## 포함 범위

현재 웹 앱, YOLO 학습실, DTW 샘플, YOLO 모델 및 학습 사진/영상은 이전 가능합니다. 별도 SLRT 프로젝트는 이 백업에 포함되지 않으며 아래 절차로 추가 이전해야 합니다. 기존 실험 TCN 가중치가 없으면 새 컴퓨터에서도 재학습이 필요합니다.

## 현재 컴퓨터에서 백업

학습과 녹화, 샘플 저장을 멈춘 뒤 실행합니다. 기존 백업은 덮어쓰지 않습니다.

```powershell
py -3.12 tools/portable_backup.py export sign-translator-transfer.zip
```

이 ZIP에는 개인 사진/영상과 학습 모델이 포함됩니다. USB나 본인만 접근 가능한 비공개 저장소로 옮기세요. 설치 라이브러리, API 키 환경변수, 임시 파일/로그는 포함하지 않습니다. 별도로 저장한 비밀 파일을 data에 넣지 마세요.

## 새 컴퓨터에서 설치

1. Python 3.12 64비트와 Python Launcher를 설치합니다.
2. `tools/portable_backup.py` 파일과 ZIP을 새 컴퓨터의 같은 폴더에 옮깁니다. 아래 명령은 옮긴 파일이 있는 폴더에서 실행하며, 비어 있는 대상 폴더로 복원합니다.

```powershell
py -3.12 portable_backup.py restore sign-translator-transfer.zip C:\SignTranslator
cd C:\SignTranslator
powershell -NoProfile -File .\scripts/setup.ps1
```

복원 시 모든 파일의 SHA-256을 검사합니다. 기존 파일이 있는 폴더에는 복원하지 않습니다. PowerShell 실행 정책이 스크립트를 차단하면 조직 정책을 확인하세요. 설치는 인터넷과 충분한 디스크 공간이 필요하며 PyTorch 등으로 다운로드가 클 수 있습니다. GPU는 필수가 아니지만 CPU 학습은 느릴 수 있습니다. GPU 지원은 설치된 PyTorch/CUDA/드라이버 구성에 따라 다릅니다.

## API 키 한 번 설정

```powershell
powershell -NoProfile -File .\scripts/set-gemini-key.ps1
```

키는 화면에 출력하지 않고 Windows 사용자 환경변수에 저장합니다. 환경변수는 암호화 저장소가 아닙니다. 키를 GitHub에 올리지 마세요. 기존 서버가 실행 중이면 종료 후 다시 실행해야 적용됩니다. 페이지의 텍스트 Google 전송 동의는 별도로 체크합니다.

## 실행

- `run_tcn.bat`: http://127.0.0.1:5194/
- `powershell -NoProfile -File .\scripts/start-yolo-studio.ps1`: http://127.0.0.1:5196/
- 브라우저에서 카메라를 허용합니다. 인터넷은 MediaPipe·ONNX·3D 라이브러리 로딩과 Gemini에 필요합니다.
- YOLO는 `data/yolo-studio/runs`의 같은 학습 기록에 있는 `fingerspell.onnx`, `labels.json`을 함께 연결합니다. 새로고침 시 재연결이 필요합니다.
- 서버 실행 창은 숨겨집니다. 포트가 사용 중이면 다른 프로그램을 종료하지 않고 안내만 출력합니다. 종료할 때는 작업 관리자에서 해당 명령줄의 서버 프로세스를 확인하세요.

## SLRT 별도 이전

SLRT 원본 프로젝트의 코드, requirements, data, models를 별도로 옮겨 그 프로젝트의 설치 절차를 따릅니다. `.venv`는 복사하지 말고 다시 만듭니다. 새 컴퓨터에서 실제 경로로 설정하세요.

```powershell
[Environment]::SetEnvironmentVariable('SLRT_PROJECT_PATH', 'C:\SlrtProject', 'User')
[Environment]::SetEnvironmentVariable('SLRT_PYTHON', 'C:\SlrtProject\.venv\Scripts\python.exe', 'User')
```

SLRT 미설치 상태에서도 DTW·YOLO 기능은 사용할 수 있습니다. SLRT 자료 전달 옵션은 끄고 사용하세요. 이전 영상 모델을 옮겼다고 연속 수어 번역 정확도가 보장되지는 않습니다.

## 검증 범위

백업/복원과 기존 코드 회귀 테스트를 실행합니다. 새로운 실제 Windows 컴퓨터에서 드라이버·카메라·GPU까지 확인한 것은 아니므로 해당 환경에서 최종 확인해야 합니다.
