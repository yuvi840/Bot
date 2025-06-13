import asyncio
import torch
from pathlib import Path
from playwright.async_api import async_playwright
from PIL import Image
import io
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or a custom trained model

async def solve_hcaptcha(page):
    print("[*] Navigating to hCaptcha page...")
    await page.goto("https://democaptcha.com/demo-form-eng/hcaptcha.html")
    await page.wait_for_timeout(2000)

    print("[*] Switching to hCaptcha iframe...")
    frame = None
    for f in page.frames:
        if "hcaptcha" in f.url:
            frame = f
            break

    if not frame:
        print("[❌] hCaptcha frame not found.")
        return

    checkbox = await frame.wait_for_selector("#checkbox", timeout=10000)
    await checkbox.click()
    print("[+] Checkbox clicked. Waiting for challenge...")

    await page.wait_for_timeout(5000)

    # Switch to challenge iframe
    challenge_frame = None
    for f in page.frames:
        if "hcaptcha.com" in f.url and "sitekey" in f.url:
            challenge_frame = f
            break

    if not challenge_frame:
        print("[❌] Challenge iframe not found.")
        return

    print("[*] Capturing image tiles...")
    tiles = await challenge_frame.query_selector_all("div.task-image .image")

    for idx, tile in enumerate(tiles):
        bbox = await tile.bounding_box()
        if not bbox:
            continue

        screenshot = await page.screenshot(clip=bbox)
        img = Image.open(io.BytesIO(screenshot)).convert('RGB')
        
        # 🧠 Run YOLOv5 detection
        results = model(img, size=640)
        labels = results.xyxyn[0][:, -1].tolist()
        names = results.names
        print(f"[{idx}] Detected objects: {[names[int(l)] for l in labels]}")

        # Example: Click if bus is detected
        if any(names[int(l)] == "bus" for l in labels):
            await tile.click()
            print(f"✅ Clicked tile {idx} (bus detected)")

    # Submit
    submit_button = await challenge_frame.query_selector("button.button-submit")
    if submit_button:
        await submit_button.click()
        print("🎉 Submitted answer!")

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=["--no-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()

        try:
            await solve_hcaptcha(page)
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
