# 진행 기록 (작업 노트)

> 다음 세션에서 바로 이어갈 수 있도록 지금까지 한 일과 다음 할 일을 정리한 문서.

## 목표

강화학습 로봇 영상처럼, **이미지 분류 AI(ResNet-50)가 이미지를 한 장씩 판단하는 과정**을
브라우저에서 슬라이드쇼로 보여주고, **적대적 공격(adversarial attack) 전/후** 판단이
어떻게 무너지는지 시각적으로 비교한다. (컴퓨터비전 보안 주제)

## 완료한 것 (2026-10-21)

- [x] 기존 [`KMUTMU/ComputerVision`](https://github.com/KMUTMU/ComputerVision) 레포 분석
  - 데이터: Imagenette / ImageNet val (`data/images/` + `data/labels.csv`)
  - 모델: ImageNet 사전학습 **ResNet-50** (torchvision)
  - 공격: FGSM / I-FGSM / MI-FGSM (`adversarial/attacks.py`), LPAA (`lpaa.py`)
- [x] **`precompute.py`** — 로컬 GPU에서 위 모델/공격을 재사용해 이미지별
      clean/adversarial 예측 + PNG를 `web/data/`에 덤프
- [x] **`web/`** — 서버 불필요한 정적 대시보드
  - 재생/일시정지/이전/다음, 속도 조절, **공격 ON/OFF 토글**
  - 키보드: `Space` 재생, `←/→` 이동, `A` 공격토글
  - 원본 vs 공격 이미지 비교 + top-k 확률 막대 + 전체 요약
- [x] **`make_demo_data.py`** — GPU 없이 레이아웃 미리보기용 데모 데이터 생성
- [x] 데이터 로직 92개 단위 검증 통과
- [x] 로컬에서 실제 실행 성공 ✅
  - I-FGSM, 30장: **공격 전 정확도 90% → 공격 후 0%** (원래 맞힌 것 중 27장이 속음)
  - 예: `goose → ostrich`, `meerkat → mongoose` 로 오분류
  - 브라우저에서 슬라이드쇼 + 공격 토글 정상 확인

## 레포 상태 / 주의점

- 작업 브랜치: **`feature/live-visualization`**  (PR: https://github.com/KMUTMU/Visualization_ComputerVision/pull/1)
- ⚠ 레포가 완전히 비어 있어서 첫 푸시 브랜치가 자동으로 기본 브랜치가 됐었음.
  `main`을 새로 만들어 정리했고 PR을 올림. **GitHub 설정에서 기본 브랜치를 `main`으로
  변경 필요** (에이전트 권한으로는 변경 불가). PR 머지 후 바꿔도 됨.
- 로컬 경로: `C:\GITHUB\Visualization_ComputerVision` (conda env `cv`)

## 실행 방법 (로컬, Windows Anaconda Prompt)

```cmd
conda activate cv
cd C:\GITHUB\Visualization_ComputerVision
python precompute.py --cv-repo C:\GITHUB\ComputerVision --attack ifgsm --num 30
cd web
python -m http.server 8000
```
→ 브라우저에서 `http://localhost:8000` 열고 ▶ 재생, 공격 ON 토글

## 다음에 할 수 있는 것 (아이디어)

- [ ] **학습 과정 타임라인 모드**: 로봇 영상처럼 epoch이 지날수록 점점 정확해지는 모습
      (현재는 학습 끝난 모델 기준)
- [ ] **공격 종류 드롭다운**: 웹에서 fgsm/ifgsm/mifgsm 바로 전환
      (precompute 때 공격별로 데이터를 각각 만들어 두고 웹에서 선택)
- [ ] **ε(노이즈 세기) 비교**: 8/255, 16/255, 32/255 생성해 비교
- [ ] **GitHub Pages 배포**: 링크만으로 공유
- [ ] PR #1 머지 + 기본 브랜치 main 으로 변경
