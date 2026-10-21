"""
데모용 샘플 데이터 생성 (GPU/모델 불필요, 표준 라이브러리만 사용)

실제 모델 추론 없이, 웹 대시보드(web/)의 레이아웃·재생·토글 동작을
미리보기/검증하기 위한 가짜(mock) 데이터를 만든다.

실제 데이터는 precompute.py 로 로컬 GPU에서 생성해야 한다.
이 스크립트는 네트워크나 torch 없이도 돌아가도록 PNG 를 직접 써서 만든다.

사용:
    python make_demo_data.py
"""

import json
import math
import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "web", "data")
IMG = os.path.join(OUT, "images")

# ImageNet 비슷한 데모 클래스 (실제 데이터와 무관, 시각화 검증용)
CLASSES = [
    "golden retriever", "tabby cat", "Egyptian cat", "Labrador retriever",
    "tiger cat", "beagle", "Siamese cat", "pug", "Persian cat", "collie",
]


def write_png(path, pixels, w, h):
    """pixels: [(r,g,b), ...] 길이 w*h, 표준 라이브러리로 PNG 저장."""
    def chunk(typ, data):
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)

    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0
        for x in range(w):
            r, g, b = pixels[y * w + x]
            raw += bytes((r & 255, g & 255, b & 255))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    idat = zlib.compress(bytes(raw), 9)
    with open(path, "wb") as f:
        f.write(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def gradient_img(w, h, seed, noise=0):
    """시드 기반의 부드러운 그라디언트 이미지 (+옵션 노이즈)."""
    px = []
    rng = (seed * 2654435761) & 0xffffffff
    def rnd():
        nonlocal rng
        rng = (1103515245 * rng + 12345) & 0x7fffffff
        return rng / 0x7fffffff
    base_r, base_g, base_b = int(rnd() * 200) + 30, int(rnd() * 200) + 30, int(rnd() * 200) + 30
    for y in range(h):
        for x in range(w):
            fx, fy = x / w, y / h
            r = int(base_r + 60 * math.sin(fx * 6 + seed))
            g = int(base_g + 60 * math.sin(fy * 6 + seed * 1.3))
            b = int(base_b + 60 * math.sin((fx + fy) * 4 + seed * 0.7))
            if noise:
                r += int((rnd() - 0.5) * noise)
                g += int((rnd() - 0.5) * noise)
                b += int((rnd() - 0.5) * noise)
            px.append((max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))
    return px


def noise_img(w, h, seed):
    px = []
    rng = (seed * 40503) & 0xffffffff
    def rnd():
        nonlocal rng
        rng = (1103515245 * rng + 12345) & 0x7fffffff
        return rng / 0x7fffffff
    for _ in range(w * h):
        v = int(rnd() * 255)
        px.append((v, int(rnd() * 255), 255 - v))
    return px


def main():
    os.makedirs(IMG, exist_ok=True)
    W = H = 96
    N = 12
    records = []
    clean_correct = adv_correct = fooled = 0

    for i in range(N):
        true_label = i % len(CLASSES)
        # 데모: clean 은 대부분 정답, adv 는 대부분 틀리게
        clean_ok = (i % 5 != 0)  # 20% 는 원래도 오답
        clean_pred = true_label if clean_ok else (true_label + 3) % len(CLASSES)
        adv_fooled = clean_ok and (i % 4 != 0)  # 맞힌 것 중 다수가 속음
        adv_pred = (true_label + 2) % len(CLASSES) if adv_fooled else clean_pred

        base = f"{i:03d}"
        write_png(os.path.join(IMG, f"{base}_orig.png"), gradient_img(W, H, i + 1), W, H)
        write_png(os.path.join(IMG, f"{base}_adv.png"), gradient_img(W, H, i + 1, noise=18), W, H)
        write_png(os.path.join(IMG, f"{base}_noise.png"), noise_img(W, H, i + 1), W, H)

        def topk(primary, conf):
            others = [c for c in range(len(CLASSES)) if c != primary][:2]
            rest = (1 - conf)
            return [
                {"label": CLASSES[primary], "index": primary, "prob": round(conf, 4)},
                {"label": CLASSES[others[0]], "index": others[0], "prob": round(rest * 0.7, 4)},
                {"label": CLASSES[others[1]], "index": others[1], "prob": round(rest * 0.3, 4)},
            ]

        clean_correct += clean_ok
        adv_ok = adv_pred == true_label
        adv_correct += adv_ok
        fooled += adv_fooled

        records.append({
            "id": i, "true_label": true_label, "true_name": CLASSES[true_label],
            "orig_img": f"images/{base}_orig.png",
            "adv_img": f"images/{base}_adv.png",
            "noise_img": f"images/{base}_noise.png",
            "clean": {
                "pred_index": clean_pred, "pred_name": CLASSES[clean_pred],
                "correct": clean_pred == true_label,
                "topk": topk(clean_pred, 0.88 if clean_ok else 0.55),
            },
            "adv": {
                "pred_index": adv_pred, "pred_name": CLASSES[adv_pred],
                "correct": adv_ok, "fooled": adv_fooled,
                "topk": topk(adv_pred, 0.79 if adv_fooled else 0.6),
            },
        })

    meta = {
        "attack": "ifgsm (DEMO/mock)",
        "attack_params": {"epsilon": 16 / 255, "epsilon_255": 16.0, "steps": 10,
                          "alpha": 1.6 / 255, "momentum": 1.0},
        "model": "ResNet-50 (DEMO 데이터 — 실제 추론 아님)",
        "device": "mock", "count": N,
        "summary": {
            "clean_accuracy": round(100 * clean_correct / N, 2),
            "adv_accuracy": round(100 * adv_correct / N, 2),
            "fooled": fooled,
            "fool_rate_of_correct": round(100 * fooled / max(clean_correct, 1), 2),
        },
        "records": records,
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"데모 데이터 생성 완료: {OUT}")
    print(f"  이미지 {N*3}개, results.json")
    print("  web/ 를 로컬 서버로 열어 미리보기:  cd web && python -m http.server 8000")


if __name__ == "__main__":
    main()
