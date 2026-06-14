"""
Project 3: Multimodal Image Generation Studio (Free Version - Pollinations.ai)
DecodeLabs - Generative AI Internship

No API key needed. Pipeline:
1. Build request URL with prompt + resolution params
2. Send request with split timeouts (connect/read)
3. Retry with exponential backoff + jitter on failures
4. Stream image bytes safely to disk (chunked, memory-safe)
5. Verify file integrity with Pillow's load()
"""

import os
import time
import random
import requests
from urllib.parse import quote
from PIL import Image

BASE_URL = "https://image.pollinations.ai/prompt/"

ASPECT_RATIO_MAP = {
    "square": (1024, 1024),
    "landscape": (1344, 768),
    "portrait": (768, 1344),
}

MAX_RETRIES = 4
CONNECT_TIMEOUT = 3.05
READ_TIMEOUT = 90
CHUNK_SIZE = 65536


def build_url(prompt, aspect="square", seed=None):
    width, height = ASPECT_RATIO_MAP.get(aspect, (1024, 1024))
    encoded_prompt = quote(prompt)
    url = f"{BASE_URL}{encoded_prompt}?width={width}&height={height}&nologo=true"
    if seed is not None:
        url += f"&seed={seed}"
    return url


def download_image(url, filename):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(
                url,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            )

            if resp.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                return True

            if resp.status_code in (429, 503):
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"[RETRY] Server busy ({resp.status_code}). "
                      f"Waiting {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue

            print(f"[ERROR] Unexpected status {resp.status_code}")
            return False

        except requests.exceptions.ConnectTimeout:
            print("[NETWORK FAILURE] Could not connect. Failing fast.")
            return False

        except requests.exceptions.ReadTimeout:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"[INFERENCE TIMEOUT] Server slow. "
                  f"Retrying in {wait:.1f}s (attempt {attempt}/{MAX_RETRIES})...")
            time.sleep(wait)
            continue

        except requests.exceptions.RequestException as e:
            print(f"[NETWORK ERROR] {e}")
            return False

    print("[FAILED] Max retries exceeded.")
    return False


def verify_image(filename):
    try:
        img = Image.open(filename)
        img.load()
        print(f"[SUCCESS] Verified image saved -> {filename} ({img.size[0]}x{img.size[1]})")
        return True
    except Exception as e:
        print(f"[CORRUPT IMAGE] Integrity check failed: {e}")
        os.remove(filename)
        return False


def main():
    prompt = input("Enter image prompt: ").strip()
    print("Aspect ratio options: square / landscape / portrait")
    aspect = input("Aspect ratio (default square): ").strip().lower() or "square"

    url = build_url(prompt, aspect=aspect)
    filename = "generated_image.png"

    print("\nGenerating image... (this can take 10-30 seconds)")
    if download_image(url, filename):
        verify_image(filename)
    else:
        print("Image generation failed.")


if __name__ == "__main__":
    main()
