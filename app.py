from starlette.applications import Starlette
from starlette.responses import HTMLResponse
from starlette.routing import Route
import uvicorn
import sys
import os
import io
import aiohttp
import asyncio
import base64
from PIL import Image
import importlib
import torch
import cv2
from SimpleHRNet import SimpleHRNet

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()
app = Starlette()
model = SimpleHRNet(48, 17, "weights/pose_hrnet_w48_384x288.pth")

@app.route("/", methods = ["GET"])
async def homepage(request):
    return HTMLResponse(
            """
            <h1> Pose Estimation Demo (HRNet) </h1>
            <p> Upload your image with a human body and receive the image with keypoints </p>
            <form action="/upload" method = "post" enctype = "multipart/form-data">
                <u> Select picture to upload: </u> <br> <p>
                1. <input type="file" name="file"><br><p>
                2. <input type="submit" value="Upload">
            </form>
            <br>
            <br>
            <u> Submit picture URL </u>
            <form action = "/classify-url" method="get">
                1. <input type="url" name="url" size="60"><br><p>                
		2. <input type="submit" value="Upload">
            </form>
            """)
def predict_image_from_bytes(bytes):
        #load byte data into a stream
        img_file = io.BytesIO(bytes)
        #encoding the image in base64 to serve in HTML
        img_pil = Image.open(img_file)
        img_pil.save("img.jpg", format="JPEG")
        img_uri = base64.b64encode(open("img.jpg", 'rb').read()).decode('utf-8')
        #make inference on image and return an HTML respons
        img = cv2.imread("img.jpg", cv2.IMREAD_COLOR)
        pts = model.predict(img)
        import numpy as np
        from misc.visualization import draw_points_and_skeleton, joints_dict
        #img = cv2.imread('image.jpg',0) # reads image 'opencv-logo.png' as grayscale
        person_ids = np.arange(len(pts), dtype=np.int32)
        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
                img = draw_points_and_skeleton(img, pt, joints_dict()['coco']['skeleton'], person_index=pid,
                                          points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                          points_palette_samples=10)
        saved_image = cv2.imwrite('pointed.jpg',img)
        saved_image_uri = base64.b64encode(open("pointed.jpg", 'rb').read()).decode('utf-8')
        return HTMLResponse(
                """
                <html>
                <figure class = "figure">
                <img src="data:image/png;base64, %s" class = "figure-img">
                </figure>
                </html>
                """ %(saved_image_uri))

@app.route("/upload", methods = ["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)

@app.route("/classify-url", methods = ["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)

if __name__ == "__main__":
    if "serve" in sys.argv:
        port = int(os.environ.get("PORT", 8008))
        uvicorn.run(app, host = "0.0.0.0", port = port)
