import os
import shutil
import requests
import base64
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from PIL import Image  # 👈 auto-resize

# === Config ===
BASE_DIR = r"C:\Users\austi\Desktop\Projects\notebook_ocr"
INPUT_DIR = Path(BASE_DIR) / "input_pics"   # 📂 Put notebooks here (each in its own folder)
OUTPUT_DIR = Path(BASE_DIR) / "output_md"
ARCHIVE_DIR = Path(BASE_DIR) / "archive"

# Ollama API
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2-vision:11b"

# Ensure folders exist
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
ARCHIVE_DIR.mkdir(exist_ok=True)


# === Auto-start Ollama if not running ===
def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434/api/tags", timeout=2)
        print("✅ Ollama service is already running.")
    except Exception:
        print("⚡ Starting Ollama service...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(5)


ensure_ollama_running()


def resize_image(image_path: Path, max_width: int = 2000) -> Path:
    """Resize image if too wide. Returns resized temp file path."""
    with Image.open(image_path) as img:
        if img.width > max_width:
            ratio = max_width / float(img.width)
            new_height = int(img.height * ratio)
            resized = img.resize((max_width, new_height), Image.LANCZOS)
            temp_path = image_path.parent / f"resized_{image_path.name}"
            resized.save(temp_path, format="JPEG", quality=90)
            return temp_path
    return image_path


def clean_transcript(text: str) -> str:
    """Remove duplicate lines but allow all headings/content."""
    lines = text.splitlines()
    cleaned, seen = [], set()
    for line in lines:
        if line.strip() and line not in seen:
            cleaned.append(line)
            seen.add(line)
    return "\n".join(cleaned).strip()


def transcribe_image(image_path: Path) -> str:
    """Send an image to Ollama vision model and return transcript text."""
    print(f"📷 Sending {image_path.name} to {MODEL}...")

    resized_path = resize_image(image_path)
    with open(resized_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    # 🔹 Refined OCR prompt with double “do not repeat”
    prompt = (
        "You are an OCR transcription agent. Transcribe this handwritten notebook page into Markdown for Obsidian.\n"
        "Rules:\n"
        "- Copy text exactly as written, once only. Do NOT repeat, invent, or summarize.\n"
        "- Use # or ## for headings if clearly shown.\n"
        "- Keep numbered lists as 1. 2. 3. and use - for bullet points. Indent sub-items.\n"
        "- If handwriting is unclear, write '??'.\n"
        "- Preserve arrows (→ or ->).\n"
        "- Mark drawings/sketches as [[Drawing: description]].\n"
        "- Stop when the page ends. Output only valid Markdown.\n"
        "- Do not repeat content. Do not repeat content."
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "images": [img_b64],
        "options": {"temperature": 0, "num_ctx": 7500},  # 🔹 updated context window
        "stream": False,
    }

    for attempt in range(1, 4):  # 3 attempts
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=600)
            response.raise_for_status()

            raw_lines = response.text.strip().splitlines()
            last_line = raw_lines[-1]
            data = json.loads(last_line)

            transcript = data.get("response", "").strip()
            if not transcript:
                print(f"⚠️ No transcript returned for {image_path.name} (attempt {attempt}/3)")
                continue

            return clean_transcript(transcript)

        except Exception as e:
            print(f"⚠️ Error for {image_path.name} (attempt {attempt}/3): {e}")
            time.sleep(5)

    print(f"❌ Failed to process {image_path.name} after 3 attempts.")
    return ""


def process_notebook(notebook_path: Path):
    """Process all images in one notebook folder."""
    notebook_name = notebook_path.name
    output_folder = OUTPUT_DIR / notebook_name
    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in notebook_path.rglob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    total = len(files)
    print(f"📓 Notebook '{notebook_name}': Found {total} page(s).")

    for i, file in enumerate(files, start=1):
        print(f"[{i}/{total}] Processing {file.name}...")

        transcript = transcribe_image(file)
        if not transcript:
            print(f"❌ Skipping {file.name}")
            continue

        # Numbered file names for clarity
        page_num = str(i).zfill(3)
        md_filename = f"{notebook_name}_Page-{page_num}.md"
        md_path = output_folder / md_filename

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {notebook_name} - Page {page_num}\n\n")
            f.write("## Transcript\n")
            f.write(transcript + "\n\n")
            f.write("## Original Image\n")
            f.write(f"![[{file.name}]]\n")

        # Copy image next to markdown
        shutil.copy(file, output_folder / file.name)

        # Archive original
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        archived_file = ARCHIVE_DIR / f"{notebook_name}_{timestamp}_{file.name}"
        shutil.move(str(file), archived_file)

        print(f"✅ Saved {md_filename} and archived {file.name}.")


if __name__ == "__main__":
    notebooks = [d for d in INPUT_DIR.iterdir() if d.is_dir()]
    if not notebooks:
        print("⚠️ No notebooks found in input_pics/. Place each notebook in its own folder.")
    else:
        for nb in notebooks:
            process_notebook(nb)
        print("\n🏁 All notebooks processed.")
