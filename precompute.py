"""
웹 시각화용 데이터 생성 스크립트 (로컬 GPU에서 실행)

하는 일:
  - ComputerVision 레포의 Imagenette/ImageNet 이미지를 한 장씩 ResNet-50에 넣어
    (1) 공격 전(clean) 예측, (2) 공격 후(adversarial) 예측을 계산하고
  - 각 이미지의 원본/공격 PNG와 예측 결과(JSON)를 web/data/ 에 덤프합니다.

왜 이렇게 하나?
  - 실제 모델 추론 + 공격 생성은 GPU가 필요하므로 로컬(RTX 노트북)에서 한 번만 돌리고,
  - 그 결과(이미지 + JSON)를 웹 대시보드가 읽어 "한 장씩 넘기며" 보여줍니다.
    (서버 없이 index.html 만 열면 됩니다.)

전제:
  - 이 스크립트는 ComputerVision 레포의 adversarial/ 폴더 옆에서 import 할 수 있어야 합니다.
    --cv-repo 로 ComputerVision 레포 경로를 지정하세요 (attacks.py, data_utils.py 사용).
  - 데이터는 그 레포 기준 data/images/ + data/labels.csv 에 준비되어 있어야 합니다.

사용 예 (Windows Anaconda Prompt):
    conda activate cv
    cd C:\GITHUB\Visualization_ComputerVision
    python precompute.py --cv-repo C:\GITHUB\ComputerVision --attack ifgsm --num 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image


def resolve_cv_repo(cv_repo):
    """ComputerVision 레포의 adversarial/ 를 import 경로에 추가."""
    adv_dir = os.path.join(cv_repo, "adversarial")
    if not os.path.isdir(adv_dir):
        sys.exit(
            f"[에러] '{adv_dir}' 가 없습니다.\n"
            f"--cv-repo 로 ComputerVision 레포 경로를 정확히 지정하세요.\n"
            f"예: python precompute.py --cv-repo C:\\GITHUB\\ComputerVision")
    sys.path.insert(0, adv_dir)
    # data/ 상대경로를 CV 레포 기준으로 쓰도록 작업 디렉터리 이동
    os.chdir(cv_repo)
    return adv_dir


def main():
    p = argparse.ArgumentParser(description="웹 시각화용 데이터 생성")
    p.add_argument("--cv-repo", type=str, required=True,
                   help="ComputerVision 레포 경로 (attacks.py/data_utils.py 사용)")
    p.add_argument("--attack", type=str, default="ifgsm",
                   choices=["fgsm", "ifgsm", "mifgsm"],
                   help="사용할 공격 (lpaa는 전이성용이라 제외)")
    p.add_argument("--num", type=int, default=30, help="시각화할 이미지 수")
    p.add_argument("--epsilon", type=float, default=16 / 255)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=1.6 / 255)
    p.add_argument("--momentum", type=float, default=1.0)
    p.add_argument("--out", type=str, default=None,
                   help="출력 폴더 (기본: 이 스크립트 옆 web/data)")
    p.add_argument("--topk", type=int, default=3, help="확률 상위 몇 개를 저장할지")
    args = p.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out or os.path.join(script_dir, "web", "data")
    img_out = os.path.join(out_dir, "images")
    os.makedirs(img_out, exist_ok=True)

    resolve_cv_repo(args.cv_repo)

    # CV 레포의 코드 재사용
    from torchvision import models  # noqa: E402
    from attacks import build_attack  # noqa: E402
    from data_utils import get_loader, get_normalizer  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    print("=" * 64)
    print(f"웹 시각화 데이터 생성  |  공격: {args.attack.upper()}")
    print(f"ε={args.epsilon:.4f} (~{args.epsilon*255:.0f}/255)  "
          f"T={args.steps}  α={args.alpha:.4f}  μ={args.momentum}")
    print(f"장치: {device} ({gpu_name})  |  대상: {args.num}장")
    print("=" * 64)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.eval().to(device)
    for prm in model.parameters():
        prm.requires_grad_(False)

    classes = models.ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
    normalize = get_normalizer(device)
    attack = build_attack(args.attack, args.epsilon, args.steps,
                          args.alpha, args.momentum)
    loader = get_loader(batch_size=1)

    def topk_probs(logits, k):
        probs = torch.softmax(logits, dim=1)[0]
        vals, idxs = probs.topk(k)
        return [
            {"label": classes[i.item()], "index": int(i.item()),
             "prob": round(float(v.item()), 4)}
            for v, i in zip(vals, idxs)
        ]

    def save_png(tensor, path):
        arr = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        Image.fromarray((arr * 255).astype(np.uint8)).save(path)

    records = []
    done = 0
    start = time.time()

    for images, labels in loader:
        if done >= args.num:
            break
        images = images.to(device)
        labels = labels.to(device)

        # 공격 전 예측
        with torch.no_grad():
            clean_logits = model(normalize(images))
        clean_top = topk_probs(clean_logits, args.topk)
        clean_pred = clean_top[0]["index"]

        # 공격 생성 + 공격 후 예측
        adv = attack(model, images, labels, normalize)
        with torch.no_grad():
            adv_logits = model(normalize(adv))
        adv_top = topk_probs(adv_logits, args.topk)
        adv_pred = adv_top[0]["index"]

        # 섭동(노이즈) 시각화: 아주 작으므로 보기 좋게 정규화
        noise = (adv[0] - images[0]).detach().cpu()
        nmin, nmax = noise.min(), noise.max()
        noise_vis = (noise - nmin) / (nmax - nmin + 1e-8)

        base = f"{done:03d}"
        save_png(images[0], os.path.join(img_out, f"{base}_orig.png"))
        save_png(adv[0], os.path.join(img_out, f"{base}_adv.png"))
        save_png(noise_vis, os.path.join(img_out, f"{base}_noise.png"))

        true_label = int(labels.item())
        records.append({
            "id": done,
            "true_label": true_label,
            "true_name": classes[true_label],
            "orig_img": f"images/{base}_orig.png",
            "adv_img": f"images/{base}_adv.png",
            "noise_img": f"images/{base}_noise.png",
            "clean": {
                "pred_index": clean_pred,
                "pred_name": classes[clean_pred],
                "correct": clean_pred == true_label,
                "topk": clean_top,
            },
            "adv": {
                "pred_index": adv_pred,
                "pred_name": classes[adv_pred],
                "correct": adv_pred == true_label,
                "fooled": (clean_pred == true_label) and (adv_pred != true_label),
                "topk": adv_top,
            },
        })

        tag = "FOOLED" if records[-1]["adv"]["fooled"] else (
            "robust" if records[-1]["adv"]["correct"] else "·")
        print(f"[{done+1}/{args.num}] {classes[true_label][:18]:18s} "
              f"clean={classes[clean_pred][:15]:15s} "
              f"-> adv={classes[adv_pred][:15]:15s}  {tag}")
        done += 1

    # 요약 통계
    n = len(records)
    clean_correct = sum(r["clean"]["correct"] for r in records)
    adv_correct = sum(r["adv"]["correct"] for r in records)
    fooled = sum(r["adv"]["fooled"] for r in records)
    elapsed = time.time() - start

    meta = {
        "attack": args.attack,
        "attack_params": {
            "epsilon": args.epsilon, "epsilon_255": round(args.epsilon * 255, 1),
            "steps": args.steps, "alpha": args.alpha, "momentum": args.momentum,
        },
        "model": "ResNet-50 (ImageNet pretrained)",
        "device": gpu_name,
        "count": n,
        "summary": {
            "clean_accuracy": round(100.0 * clean_correct / max(n, 1), 2),
            "adv_accuracy": round(100.0 * adv_correct / max(n, 1), 2),
            "fooled": fooled,
            "fool_rate_of_correct": round(
                100.0 * fooled / max(clean_correct, 1), 2),
        },
        "records": records,
    }

    out_json = os.path.join(out_dir, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("-" * 64)
    print(f"공격 전 정확도 : {meta['summary']['clean_accuracy']:.2f}%")
    print(f"공격 후 정확도 : {meta['summary']['adv_accuracy']:.2f}%  "
          f"(원래 맞힌 것 중 {fooled}장이 공격에 속음)")
    print(f"소요 시간      : {elapsed:.1f}초")
    print(f"저장 위치      : {out_json}")
    print(f"               {img_out}  ({n*3}개 PNG)")
    print("-" * 64)
    print("이제 web/index.html 을 브라우저로 열면 슬라이드쇼가 재생됩니다.")


if __name__ == "__main__":
    main()
