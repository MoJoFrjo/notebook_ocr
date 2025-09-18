import os
import shutil
import requests
import base64
import json
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageFilter

# === Config ===
BASE_DIR   = r"C:\Users\austi\Desktop\Projects\notebook_ocr"
INPUT_DIR  = Path(BASE_DIR) / "input_pics"
OUTPUT_DIR = Path(BASE_DIR) / "output_txt"
ARCHIVE_DIR= Path(BASE_DIR) / "archive"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3.2-vision:11b"

MAX_WIDTH        = 2000
AUTO_SPLIT_COLS  = True
NUM_CTX          = 9000
NUM_PREDICT      = 1400

for folder in (INPUT_DIR, OUTPUT_DIR, ARCHIVE_DIR):
    folder.mkdir(exist_ok=True)

# ---------- imaging helpers ----------

def resize_image(image_path: Path, max_width: int = MAX_WIDTH) -> Path:
    with Image.open(image_path) as img:
        if img.width > max_width:
            ratio = max_width / img.width
            new_h = int(img.height * ratio)
            img = img.resize((max_width, new_h), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
        tmp = image_path.parent / f"resized_{image_path.name}"
        img.save(tmp, format="JPEG", quality=90)
        return tmp

def split_columns(preprocessed_path: Path) -> list[tuple[str, Path]]:
    """Return list of (label, image_path) for left/right columns."""
    if not AUTO_SPLIT_COLS:
        return [("Full Page", preprocessed_path)]
    with Image.open(preprocessed_path) as img:
        w, h = img.size
        left_box  = (0, 0, w // 2, h)
        right_box = (w // 2, 0, w, h)
        left_img  = img.crop(left_box)
        right_img = img.crop(right_box)

        left_path  = preprocessed_path.parent / f"left_{preprocessed_path.name}"
        right_path = preprocessed_path.parent / f"right_{preprocessed_path.name}"
        left_img.save(left_path,  format="JPEG", quality=90)
        right_img.save(right_path, format="JPEG", quality=90)
        return [("Left Column", left_path), ("Right Column", right_path)]

def shorten_filename(path: Path, max_total_len: int = 240, stem_keep: int = 120) -> Path:
    s = str(path)
    if len(s) <= max_total_len:
        return path
    ext  = path.suffix
    stem = path.stem[:max(8, stem_keep - len(ext))]
    return path.with_name(stem + ext)

# ---------- text cleanup ----------

def clean_text(s: str) -> str:
    if "[END]" in s:
        s = s.split("[END]", 1)[0]
    idx = s.find("\nNote:")
    if idx != -1:
        s = s[:idx]
    lines = [ln.rstrip() for ln in s.splitlines()]
    out, last, repeats = [], None, 0
    for ln in lines:
        if ln and ln == last:
            repeats += 1
            if repeats >= 2:
                continue
        else:
            repeats = 0
        out.append(ln)
        last = ln
    return "\n".join([ln for ln in out if ln.strip()]).strip()

# ---------- core OCR ----------

def ocr_one_image(path: Path) -> str:
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "You are a STRICT OCR transcription agent. Extract ONLY the exact visible handwritten/printed text.\n"
        "Rules:\n"
        "• Copy text verbatim (spelling, punctuation, capitalization). Do NOT rephrase.\n"
        "• Do NOT summarize, infer, or invent missing text. If unclear, write '??'.\n"
        "• Preserve structure exactly as written on the page:\n"
        "   - Each main numbered item (1, 2, 3, etc.) must be on its own line.\n"
        "   - Any indented or sub-items beneath a number must be placed on their own lines, prefixed with '- '.\n"
        "   - Preserve fractions (¼, ½, ⅛) and musical notation symbols as written.\n"
        "   - Do not merge separate sub-items into one line.\n"
        "• If the page has two columns, output the entire left column first (top to bottom), then the right column (top to bottom).\n"
        "• Output plain text only (no Markdown, no headers, no extra symbols).\n"
        "• End the output with the token [END] and nothing after."
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "options": {
            "temperature": 0,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "repeat_penalty": 1.2,
            "penalty_last_n": 256
        },
        "stream": False
    }

    resp = requests.post(OLLAMA_URL, json=payload, timeout=600)
    resp.raise_for_status()

    try:
        data = resp.json()
    except Exception:
        data = json.loads(resp.text.strip().splitlines()[-1])

    return data.get("response", "")

def transcribe_image(image_path: Path) -> str:
    print(f"📷 OCR: {image_path.name}")
    tmp = resize_image(image_path)
    parts = split_columns(tmp)

    sections = []
    try:
        for label, p in parts:
            part_txt = ocr_one_image(p)
            cleaned = clean_text(part_txt)
            sections.append(f"### {label}\n{cleaned}\n")
    finally:
        for _, p in parts + [("tmp", tmp)]:
            try:
                if Path(p).exists():
                    Path(p).unlink()
            except Exception:
                pass

    return "\n".join(sections).strip()

# ---------- batch ----------

def process_notebook(notebook_path: Path):
    notebook_name = notebook_path.name
    out_dir = OUTPUT_DIR / notebook_name
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        list(notebook_path.glob("*.jpg")) +
        list(notebook_path.glob("*.jpeg")) +
        list(notebook_path.glob("*.png")) +
        list(notebook_path.glob("*.JPG")) +
        list(notebook_path.glob("*.JPEG")) +
        list(notebook_path.glob("*.PNG"))
    )

    if not files:
        print(f"⚠️ No images found in {notebook_name}")
        return

    for i, img in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing {img.name}")
        text = transcribe_image(img)
        if not text:
            print(f"⚠️ Empty OCR for {img.name}")
            continue

        txt_path = out_dir / f"{notebook_name}_Page-{i:03}.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        archived = ARCHIVE_DIR / f"{notebook_name}_{datetime.now():%Y-%m-%d_%H-%M-%S}_{img.name}"
        archived = shorten_filename(archived)
        shutil.move(str(img), archived)
        print(f"✅ Saved {txt_path.name}")

    print(f"📓 Finished notebook: {notebook_name}")

if __name__ == "__main__":
    notebooks = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    if not notebooks:
        print("⚠️ No notebooks found. Place images inside input_pics/<notebook_name>/")
    else:
        for nb in notebooks:
            process_notebook(nb)
        print("🏁 OCR complete.")
