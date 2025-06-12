import asyncio
import torch
from yolov5 import YOLOv5
import cv2
import numpy as np
from playwright.async_api import async_playwright
import os
import time

# Load pretrained YOLOv5s model
yolo = YOLOv5("yolov5s.pt", device="cpu")  # Or 'cuda' if using GPU

# Target class names from COCO that might be used in HCaptcha
label_map = {
    "bicycle": "bike",
    "bus": "bus",
    "car": "car",
    "motorcycle": "motorcycle",
    "traffic light": "traffic light",
    "fire hydrant": "hydrant",
}

# Helper to get label from challenge prompt
def map_prompt_to_label(prompt_text):
    prompt_text = prompt_text.lower()
    for coco, readable in label_map.items():
        if readable in prompt_text:
            return coco
    return None

async def solve_hcaptcha():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # GUI visible
        context = await browser.new_context()
        page = await context.new_page()

        # Test HCaptcha demo page
        await page.goto("https://accounts.hcaptcha.com/demo")

        # Wait for iframe
        frame = await page.frame_locator("iframe").first.content_frame()
        await frame.click("div[role='checkbox']")

        print("[*] Waiting for challenge...")
        await page.wait_for_timeout(3000)  # Wait for HCaptcha challenge to load

        # Switch to challenge iframe
        challenge_frame = page.frame(name=lambda name: name and "hcaptcha" in name)
        if not challenge_frame:
            print("[-] Challenge iframe not found.")
            return

        prompt_text = await challenge_frame.locator(".prompt-text").inner_text()
        print(f"[+] Prompt: {prompt_text}")

        target_label = map_prompt_to_label(prompt_text)
        if not target_label:
            print("[-] Target object not supported in model.")
            return

        print(f"[+] Looking for: {target_label}")

        tiles = await challenge_frame.locator(".task-image .image").all()
        os.makedirs("tiles", exist_ok=True)

        for i, tile in enumerate(tiles):
            box = await tile.bounding_box()
            await page.screenshot(path=f"tiles/tile_{i}.png", clip=box)

        # Detect objects in each tile
        to_click = []
        for i in range(len(tiles)):
            img_path = f"tiles/tile_{i}.png"
            results = yolo.predict(img_path)
            labels = results[0]["class_name"]

            if target_label in labels:
                print(f"[✓] Detected '{target_label}' in tile {i}")
                to_click.append(i)
            else:
                print(f"[ ] Tile {i} skipped")

        # Click matching tiles
        for i in to_click:
            await tiles[i].click()
            await asyncio.sleep(0.5)

        print("[*] Clicked all detected tiles.")

        await asyncio.sleep(10)
        await browser.close()

asyncio.run(solve_hcaptcha())
