import asyncio
import base64
import io
from PIL import Image
from playwright.async_api import async_playwright
import torch
import numpy as np

# Load YOLOv5s model (make sure yolov5s.pt is in the same directory or installed from torch hub)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

def map_prompt_to_label(prompt: str):
    prompt = prompt.lower()
    if "bicycle" in prompt:
        return "bicycle"
    elif "bus" in prompt:
        return "bus"
    elif "motorcycle" in prompt:
        return "motorcycle"
    elif "traffic light" in prompt:
        return "traffic light"
    elif "fire hydrant" in prompt:
        return "fire hydrant"
    elif "crosswalk" in prompt:
        return "crosswalk"
    elif "truck" in prompt:
        return "truck"
    else:
        return None

def detect_target(image: Image.Image, target_label: str) -> bool:
    """Detects if the given label exists in the image using YOLOv5"""
    results = model(image)
    detected = False

    for *box, conf, cls in results.xyxy[0]:  # xyxy format
        label = model.names[int(cls)]
        if label == target_label:
            detected = True
            break
    return detected

async def solve_hcaptcha():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Use True for headless
        context = await browser.new_context()
        page = await context.new_page()

        print("[*] Navigating to hCaptcha demo page...")
        await page.goto("https://accounts.hcaptcha.com/demo")

        # Wait for iframe
        await page.wait_for_timeout(3000)
        frame_locator = page.frame_locator("iframe")
        frame_element = await frame_locator.element_handle()
        challenge_frame = await frame_element.content_frame()

        if not challenge_frame:
            print("[-] Challenge iframe not found.")
            return

        # Get prompt text
        prompt_text = await challenge_frame.locator(".prompt-text").text_content()
        print(f"[+] Prompt: {prompt_text}")

        # Wait for images to load
        await asyncio.sleep(2)

        # Get all challenge images
        image_elements = await challenge_frame.query_selector_all('div.task-image .image')
        print(f"[+] Found {len(image_elements)} challenge images.")

        target_label = map_prompt_to_label(prompt_text)

        if not target_label:
            print(f"[-] Target '{prompt_text}' not supported by YOLO model.")
            return

        for i, element in enumerate(image_elements):
            style = await element.get_attribute("style")
            if not style or "data:image" not in style:
                print(f"[-] Skipping image {i}, no base64 data.")
                continue

            # Extract base64 image
            try:
                b64data = style.split("url(\"")[1].split("\")")[0].split(",")[1]
                img_bytes = base64.b64decode(b64data)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                print(f"[-] Error decoding image {i}: {e}")
                continue

            if detect_target(img, target_label):
                print(f"[✓] Clicking on image {i} as it matches '{target_label}'")
                await element.click()
            else:
                print(f"[ ] Image {i} does not match.")

        print("[✓] Done. Attempted hCaptcha solve.")
        await asyncio.sleep(3)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(solve_hcaptcha())
