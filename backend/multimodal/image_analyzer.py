"""
이미지 분석 + self-consistency verification → image_metadata.json 생성

입력: backend/data/raw/{defects,equipment,diagram}/
출력: backend/data/processed/image_metadata.json
실행: python -m backend.multimodal.image_analyzer
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image

from backend.multimodal.metadata_utils import (
    infer_error_codes,
    needs_review,
    normalize_detected_component,
    normalize_error_codes,
    parse_json_response,
    resolve_graph_links,
)
from backend.multimodal.prompts import PROMPTS, VERIFIER_PROMPT

log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
IMAGE_ROOT  = BACKEND_DIR / "data" / "raw"
OUTPUT_PATH = BACKEND_DIR / "data" / "processed" / "image_metadata.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CATEGORY_DIRS: dict[str, str] = {
    "defects":   "defects",
    "equipment": "equipment",
    "diagrams":  "diagram",
}
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SLEEP_BETWEEN  = 0.3
MAX_LONG_SIDE  = 1024


@dataclass(frozen=True)
class Config:
    api_key:  str
    base_url: str
    model:    str


def load_env(start: Path) -> None:
    for directory in [start, *start.parents]:
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return


def get_config() -> Config:
    load_env(Path(__file__).resolve().parent)
    return Config(
        api_key  = os.getenv("GATEWAY_API_KEY", "").strip(),
        base_url = os.getenv("GATEWAY_BASE_URL", "").strip(),
        model    = os.getenv("MULTIMODAL_MODEL", "gemini-2.5-pro").strip(),
    )

# image helpers
def encode_image(path: Path) -> tuple[str, str]:
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if max(w, h) > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / max(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode(), "image/jpeg"


# VLM calls
def call_vlm(client: OpenAI, model: str, b64: str, prompt: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
        max_tokens=900,
        temperature=0.1,
    )
    content = response.choices[0].message.content if response.choices else None
    if not content:
        raise ValueError(f"빈 응답. finish_reason={response.choices[0].finish_reason!r}")
    return parse_json_response(content)


def analyze_image(
    client: OpenAI,
    model: str,
    image_path: Path,
    category: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    b64, _ = encode_image(image_path)

    analysis = call_vlm(client, model, b64, PROMPTS[category])
    analysis["detected_component"] = normalize_detected_component(
        analysis.get("detected_component"), category
    )
    analysis["related_error_codes"] = normalize_error_codes(
        analysis.get("related_error_codes", [])
    )
    if not analysis["related_error_codes"]:
        analysis["related_error_codes"] = infer_error_codes(analysis, category)

    verification = None
    try:
        verifier_prompt = VERIFIER_PROMPT.format(
            analysis_json=json.dumps(analysis, ensure_ascii=False, indent=2)
        )
        verification = call_vlm(client, model, b64, verifier_prompt)
    except Exception as e:
        log.warning("verifier failed for %s: %s", image_path.name, e)

    return analysis, verification


# result builders
def build_entry(
    image_path: Path,
    category: str,
    analysis: dict[str, Any],
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    error_codes = analysis.get("related_error_codes", [])
    confidence  = float(verification.get("confidence_score", -1)) if verification else -1

    return {
        "image_id":         image_path.stem,
        "file_path":        str(image_path),
        "category":         category,
        "analysis":         analysis,
        "verification":     verification,
        "confidence_score": confidence,
        "graph_links":      resolve_graph_links(error_codes),
        "needs_review":     needs_review(analysis, verification),
    }


def collect_images() -> list[tuple[Path, str]]:
    images: list[tuple[Path, str]] = []
    for category, folder_name in CATEGORY_DIRS.items():
        folder = IMAGE_ROOT / folder_name
        if not folder.exists():
            print(f"[SKIP] 폴더 없음: {folder}")
            continue
        files = sorted(f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTS)
        print(f"[FOUND] {category:10s}: {len(files)}장")
        images.extend((f, category) for f in files)
    return images


# main
def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    cfg = get_config()
    if not cfg.api_key:
        raise RuntimeError("GATEWAY_API_KEY가 없습니다. .env를 확인하세요.")

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    images = collect_images()
    if not images:
        print("이미지를 찾을 수 없습니다. 아래 경로를 확인하세요:")
        for folder_name in CATEGORY_DIRS.values():
            print(f"  {IMAGE_ROOT / folder_name}/")
        return

    total = len(images)
    print(f"\n총 {total}장  →  {cfg.model}")
    print("─" * 62)

    results: list[dict[str, Any]] = []
    failed:  list[dict[str, str]] = []

    for i, (img_path, category) in enumerate(images, 1):
        print(f"[{i:02d}/{total}] {img_path.name:<36} ({category})", end=" ... ", flush=True)
        try:
            analysis, verification = analyze_image(client, cfg.model, img_path, category)
            entry = build_entry(img_path, category, analysis, verification)
            results.append(entry)

            OUTPUT_PATH.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )

            score = f"{entry['confidence_score']:.2f}" if entry["confidence_score"] >= 0 else "n/a"
            flag  = "⚠ review" if entry["needs_review"] else "OK     "
            label = analysis.get("severity") or analysis.get("condition") or "?"
            print(f"{flag}  [conf:{score}] [{label}]")

        except Exception as e:
            print(f"FAIL — {e}")
            failed.append({"file": str(img_path), "error": str(e)})

        if i < total:
            time.sleep(SLEEP_BETWEEN)

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    review_list = [r for r in results if r["needs_review"]]
    low_conf    = [r for r in results if 0 <= r["confidence_score"] < 0.6]

    print("\n" + "─" * 62)
    print(f"완료     : {len(results)}장 성공  /  {len(failed)}장 실패")
    print(f"저장     : {OUTPUT_PATH}")
    print(f"검토 필요 : {len(review_list)}장  (low-confidence: {len(low_conf)}장)")

    if review_list:
        print("\n── needs_review ──")
        for r in review_list:
            score = f"{r['confidence_score']:.2f}" if r["confidence_score"] >= 0 else "n/a"
            print(f"  [{score}] {r['image_id']:<40} ({r['category']})")

    if failed:
        print("\n── 실패 ──")
        for item in failed:
            print(f"  {item['file']}  →  {item['error']}")


if __name__ == "__main__":
    main()