# AI 이미지 판단 라이브 시각화 (공격 전/후 비교)

강화학습 로봇 영상처럼, **이미지 분류 AI(ResNet-50)가 이미지를 한 장씩 보고 판단하는 과정**을
브라우저에서 "재생" 버튼으로 슬라이드쇼처럼 보여주고,
**적대적 공격(adversarial attack)을 넣었을 때 판단이 어떻게 무너지는지** 시각적으로 비교하는 데모입니다.

> 보안(adversarial robustness) 연구용 시각화입니다.
> 모델/공격 코드는 기존 [`KMUTMU/ComputerVision`](https://github.com/KMUTMU/ComputerVision)
> 레포의 `adversarial/` (ResNet-50 + FGSM/I-FGSM/MI-FGSM)를 그대로 재사용합니다.

## 무엇을 보여주나

- **왼쪽**: 원본 이미지 → AI의 판단(클래스 + 확신도), 정답 여부
- **가운데**: 공격이 추가한 노이즈(사람 눈엔 거의 안 보임, 증폭 표시)
- **오른쪽**: 공격당한 이미지 → AI의 판단이 어떻게 바뀌었는지 (속았는지/버텼는지)
- **아래**: 상위 후보 확률 막대, 그리고 전체 요약(공격 전/후 정확도, 속은 비율)
- **재생/일시정지/이전/다음**, 속도 조절, **공격 ON/OFF 토글** (키보드: `Space` 재생, `←/→` 이동, `A` 공격토글)

## 구조

```
Visualization_ComputerVision/
├─ precompute.py       # [로컬 GPU] ComputerVision 모델/공격으로 결과(JSON+PNG) 생성
├─ make_demo_data.py   # [GPU 불필요] 레이아웃 미리보기용 가짜 데이터 생성
└─ web/                # 순수 웹 대시보드 (서버 불필요, 정적 파일)
   ├─ index.html
   ├─ style.css
   ├─ app.js
   └─ data/            # precompute.py / make_demo_data.py 가 채움
      ├─ results.json
      └─ images/*.png
```

**왜 이렇게 나눴나?**
실제 모델 추론과 공격 생성은 GPU가 필요하므로 **로컬(RTX 노트북)에서 한 번만** 돌려
결과를 `web/data/` 에 덤프하고, 웹은 그 정적 데이터를 읽어 재생만 합니다.
→ 서버 없이 `index.html` 하나로 어디서든(또는 GitHub Pages) 볼 수 있습니다.

## 사용법

### 1. 실제 데이터 생성 (로컬 GPU, 권장)

ComputerVision 레포에 데이터(`data/images/` + `data/labels.csv`)가 준비돼 있어야 합니다.
(준비 방법은 ComputerVision 레포의 `adversarial/README.md` 참고)

```cmd
conda activate cv
cd C:\GITHUB\Visualization_ComputerVision
python precompute.py --cv-repo C:\GITHUB\ComputerVision --attack ifgsm --num 30
```

- `--attack` : `fgsm` / `ifgsm` / `mifgsm`
- `--num`    : 시각화할 이미지 수
- 공격 파라미터는 논문 기본값(ε=16/255, T=10, α=1.6/255, μ=1.0). 필요시 `--epsilon` 등으로 변경

### 2. 브라우저로 보기

`web/` 폴더를 로컬 서버로 열어야 합니다 (JSON 로드는 `file://` 에서 막힘):

```cmd
cd web
python -m http.server 8000
```

브라우저에서 `http://localhost:8000` → **▶ 재생** 을 누르면 한 장씩 넘어갑니다.
오른쪽 **공격 ON** 토글을 켜면 공격당한 이미지의 판단이 나타납니다.

### (옵션) GPU 없이 레이아웃만 미리보기

```cmd
python make_demo_data.py      # 가짜 데이터 생성 (실제 추론 아님)
cd web && python -m http.server 8000
```

## 실험 아이디어

- `--attack` 을 `fgsm` → `ifgsm` → `mifgsm` 으로 바꿔 생성하고, 각각의 "속은 비율"을 비교
- `--epsilon` 을 `8/255`, `16/255`, `32/255` 로 바꿔 노이즈 세기에 따른 변화 관찰
- 같은 이미지에서 원본은 "개"로, 공격 후엔 엉뚱한 클래스로 바뀌는 사례 찾기
