import os
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
import base64
import time
import uuid
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request, Body, WebSocket, WebSocketDisconnect
import requests
import shutil

from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List  # Import List from typing module
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import json
from pydantic import BaseModel, Field
from openai import OpenAI  # OpenAI Python library to make API calls
import yaml
from dotenv import load_dotenv

from gtts import gTTS
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import subprocess
from threading import Lock, Thread



app = FastAPI()

# Allow CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# fixed for the system
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
base_path = str(BASE_DIR)
configuration_file = str(BASE_DIR / "config.yaml")
UPLOAD_DIRECTORY = str(BASE_DIR / "uploads")
EMBEDD_IMG_DIRECTORY = str(BASE_DIR / "embedding_images")
CATALOG_DIRECTORY = str(BASE_DIR / "embedding_images" / "embedding_images")
STATIC_DIRECTORY = str(BASE_DIR / "static")
DATA_DIRECTORY = str(BASE_DIR / "data")
CHAT_DATA_FILE = str(BASE_DIR / "data" / "chat_messages.json")
REWARDS_DATA_FILE = str(BASE_DIR / "data" / "rewards.json")
REDUCED_FPS = 4
max_workers = 8  # Set the desired number of parallel threads
PARALLEL_PROCESSING = 1


def load_configuration(config_path=configuration_file):
    try:
        # Load default configuration
        with open(config_path, 'r') as default_file:
            configuration = yaml.safe_load(default_file)
        return configuration
    except FileNotFoundError as e:
        print(f"ERROR: Fail to load configuration from:{config_path}. error:{e}. Make sure the file paths are correct.")
        return {}
    except yaml.YAMLError as e:
        print(f"ERROR loading YAML file:{config_path}. error:{e}")
        return {}


def load_parameter(param,config_dict):
    if param in config_dict.keys():
        print(f"{param} loaded")
        return config_dict[param]
    else:
        print(f"ERROR: fail to load param:{param}")
        return None

# load the configuration
config = load_configuration()
if not config:
    print(f"ERROR: failed to load configuration from file:{configuration_file}")
else:
    print(f"SUCCESS: loaded configuration from file:{configuration_file}")

# load the configurtion parameter
openai_api_key = os.environ.get("OPENAI_API_KEY") or load_parameter("openai_api_key",config)
number_of_similar_images = load_parameter("number_of_similar_images",config) or 4
# base_path = load_parameter("base_path",config)
custom_params = load_parameter("custom_params",config) or {"model": "gpt-4o-mini"}

custom_yolo_model = BASE_DIR / "best.pt"
fallback_yolo_model = BASE_DIR / "yolov8s.pt"
yolo_model_file = str(custom_yolo_model if custom_yolo_model.exists() else fallback_yolo_model)
embeddings_file = str(BASE_DIR / "embedding.json")
common_qna_file = str(BASE_DIR / "qna" / "common.txt")
qna_directory = str(BASE_DIR / "qna")
prompt_file = str(BASE_DIR / "prompt.json")

# Serve local assets and generated files.
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(STATIC_DIRECTORY, exist_ok=True)
os.makedirs(DATA_DIRECTORY, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIRECTORY), name="uploads")
app.mount("/embedding_images", StaticFiles(directory=EMBEDD_IMG_DIRECTORY), name="embedding_images")
app.mount("/catalog", StaticFiles(directory=CATALOG_DIRECTORY), name="catalog")
app.mount("/static", StaticFiles(directory=STATIC_DIRECTORY), name="static")

# In-memory storage for metadata
storage = {
    "cropped_images": {}, ## cropped_images[idx] = {"class_name":class_name,"score":score, "cropped_image_path":cropped_image_path}
    "similar_images": {}, ## similar_images[ids]  = {"image_file_name":image_file_name,"distance":distance, "prompt" :prompt, "masked_image_path": masked_image_path}
}

data_lock = Lock()
video_rooms = {}


class ChatMessageIn(BaseModel):
    sender: str = Field(..., min_length=1, max_length=40)
    text: str = Field(..., min_length=1, max_length=1000)


class RewardCommentIn(BaseModel):
    author: str = Field(..., min_length=1, max_length=40)
    text: str = Field(..., min_length=1, max_length=500)


def utc_ms():
    return int(time.time() * 1000)


def safe_key(value, default="general"):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip()).strip("-").lower()
    return cleaned[:80] or default


def clean_text(value, max_length):
    cleaned = re.sub(r"\s+", " ", str(value or "").strip())
    return cleaned[:max_length]


def load_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default


def save_json_file(path, data):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def default_rewards_state():
    return {
        "challenges": {
            "wednesday-pink": {"id": "wednesday-pink", "title": "On Wednesdays We Wear Pink"},
            "met-calm": {"id": "met-calm", "title": "The Met Calm"},
            "street-fit": {"id": "street-fit", "title": "Street Fit Check"},
        },
        "uploads": [],
    }


def public_upload_path(path):
    try:
        relative = Path(path).resolve().relative_to(Path(UPLOAD_DIRECTORY).resolve())
        return f"/uploads/{relative.as_posix()}"
    except ValueError:
        return None

labels = ["sunglasses","hat","jacket","shirt","pants","shorts","skirt","dress","bag", "shoes", "item"]
label_aliases = {
    "sunglass": "sunglasses",
    "shoe": "shoes",
    "handbag": "bag",
    "backpack": "bag",
}
fashion_label_set = set(labels) - {"item"}


def normalize_class_name(class_name):
    class_name = str(class_name).lower().strip()
    return label_aliases.get(class_name, class_name)


def infer_class_from_filename(path):
    stem = Path(path).stem.lower()
    for label in fashion_label_set:
        if stem.startswith(label) or f"_{label}" in stem:
            return label
    if stem.startswith("sunglass"):
        return "sunglasses"
    if stem.startswith("shoe"):
        return "shoes"
    return "item"

def populate_label_dict():
    label_dict = {}

    for item in labels:
        label_dict[item] = 0

    return label_dict


## embeddings = [{"vector":[512 normalized values],"image_file_name":<image_file_name>, "prompt":<detail description of the image>},...]
def get_similar_images(embeddings,input_images,prompts,number_of_similar_images):
    ensure_ml_resources()
    
    print(f"        ===> get_similar_images(embeddings, input_images, prompts, {number_of_similar_images})")

    similar_images = {}
    idx = 0
    for img in input_images:
        # Extract embedding for the input file
        input_embedding = extractor(img[0])
        print(f"======== similar image for image:{img[0]} ==========")
        sm = similar_matches(input_embedding,embeddings,number_of_similar_images,img[1])
        for item in sm:
            similar_images[idx] = {
                                    "image_file_name": f"/catalog/{item['image_file_name']}",
                                    "distance": item['distance'],
                                    "prompt": prompts[item['image_file_name']],
                                    "class_name": img[1],
                                    }
            print(f"class_name={img[1]}, file={item['image_file_name']}, cos dis={item['distance']}, prompt={prompts[item['image_file_name']]}")
            idx += 1
    print("============================================================================")
    return similar_images


def similar_matches(input_embedding, embeddings, number_of_suggestions, class_name=None):

    print(f"    ===> similar_matches(input_embedding, embeddings, {number_of_suggestions}, {class_name})")

    normalized_class = normalize_class_name(class_name) if class_name else None
    searchable_embeddings = embeddings
    if normalized_class and normalized_class != "item":
        prefix = "sunglasses" if normalized_class == "sunglasses" else normalized_class
        searchable_embeddings = [
            embedding for embedding in embeddings
            if normalize_class_name(embedding["image_file_name"].split("_")[0]) == prefix
        ] or embeddings

    # Extract vectors from embeddings
    embed_list = [e['vector'] for e in searchable_embeddings]

    # Calculate cosine similarities
    distances = cosine_similarity([input_embedding], embed_list)[0]

    # Get the indices of the N smallest distances (N best scores)
    best_indices = np.argsort(distances)[::-1][:number_of_suggestions]  # Sort in descending order and take top N

    # Create a list of dictionaries with the best matches
    best_matches = [
        {
            "image_file_name": searchable_embeddings[i]['image_file_name'],
            "distance": distances[i],
        }
        for i in best_indices
    ]
    return best_matches


class FeatureExtractor:
    def __init__(self, modelname):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        # Load the pre-trained model
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.model.eval()

        # Get the input size required by the model
        self.input_size = self.model.default_cfg["input_size"]

        config = resolve_data_config({}, model=modelname)
        # Get the preprocessing function provided by TIMM for the model
        self.preprocess = create_transform(**config)

    def __call__(self, imagepath):
        import torch

        # Preprocess the input image
        input_image = Image.open(imagepath).convert("RGB")  # Convert to RGB if needed
        input_image = self.preprocess(input_image)

        # Convert the image to a PyTorch tensor and add a batch dimension
        input_tensor = input_image.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Extract the feature vector
        feature_vector = output.squeeze().numpy()

        return normalize(feature_vector.reshape(1, -1), norm="l2").flatten()


## embeddings = [{"image_file_name":<image_file_name>, "prompt":<detail description of the image>},...]
def load_prompts(prompt_file):
    prompts = {}
    try:
        with open(prompt_file, 'r') as f:
            # Check if the file is empty
            if os.stat(prompt_file).st_size == 0:
                print(f"Warning: File '{embeddings_file}' is empty. loading empty prompts")
                return {}, []
            prompts_list = json.load(f)

            for pmt in prompts_list:
                image_file_name = pmt['image_file_name']
                prompt = pmt['prompt']
                prompts[image_file_name] = prompt

        return prompts, prompts_list

    except FileNotFoundError:
        print(f"Error: File '{prompt_file}' not found. loading empty prompts")
        return {}, []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{prompt_file}': {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading prompts from '{prompt_file}': {e}")
        return None, None


## embeddings = [{"vector":[512 normalized values],"image_file_name":<image_file_name>, "prompt":<detail description of the image>},...]
def load_embeddings(embeddings_file):
    try:
        with open(embeddings_file, 'r') as f:
            # Check if the file is empty
            if os.stat(embeddings_file).st_size == 0:
                print(f"Warning: File '{embeddings_file}' is empty. loading empty embedds")
                return []
            embeddings = json.load(f)
        return embeddings
    except FileNotFoundError:
        print(f"Error: File '{embeddings_file}' not found. loading empty embedds")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{embeddings_file}': {e}")
        return None
    except Exception as e:
        print(f"An error occurred while loading embeddings from '{embeddings_file}': {e}")
        return None


def convert_to_high_quality_png(input_image_path, output_image_path, target_size=(1024, 1024), max_file_size=4 * 1024 * 1024):

    print(f"    ===> convert_to_high_quality_png({input_image_path},{output_image_path},{target_size},{max_file_size/(1024*1024)} Mb)")
    # Open the input image
    image = Image.open(input_image_path).convert("RGBA")
    
    # Resize the image while maintaining aspect ratio
    image.thumbnail(target_size, Image.LANCZOS)
    
    # Create a new image with a white background and paste the resized image onto it
    background = Image.new('RGBA', target_size, (255, 255, 255, 0))
    image_width, image_height = image.size
    offset = ((target_size[0] - image_width) // 2, (target_size[1] - image_height) // 2)
    background.paste(image, offset)
    
    background.save(output_image_path, format='PNG', optimize=True)

    file_size = os.path.getsize(output_image_path)
    current_size = target_size

    # PNG does not have a JPEG-style quality setting. If a file is too large,
    # reduce canvas size gradually while preserving the transparent square shape.
    while file_size > max_file_size and min(current_size) > 256:
        current_size = (int(current_size[0] * 0.9), int(current_size[1] * 0.9))
        resized = background.resize(current_size, Image.LANCZOS)
        resized.save(output_image_path, format='PNG', optimize=True)
        file_size = os.path.getsize(output_image_path)
    
    return output_image_path


# Function to crop and save detected objects and create an annotated image
def generate_crop_mask_files(image_path, output_dir):
    ensure_ml_resources()

    print(f"    ===> generate_crop_mask_files({image_path},{output_dir})")

    cropped_images = {}
    masked_images = {}

    # Load image
    original_image = Image.open(image_path).convert("RGB")
    image_np = np.array(original_image)

    # Perform inference
    results = detection_model(image_np)
    results = results[0]  # YOLO returns a list of results, we take the first one

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a drawing context
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    font = ImageFont.load_default()  # Load a default font

    # Extract the original file name and extension
    base_name = os.path.basename(image_path)
    file_name, file_extension = os.path.splitext(base_name)

    if len(results.boxes) == 0:
        class_name = infer_class_from_filename(image_path)
        cropped_image_path = os.path.join(output_dir, f"{file_name}_crop_{class_name}_0.png")
        original_image.save(cropped_image_path, format="PNG")
        convert_to_high_quality_png(cropped_image_path, cropped_image_path,target_size=(150, 150), max_file_size=4 * 1024 * 1024)
        cropped_images[0] = {"class_name":class_name,"score":0.0, "cropped_image_path":cropped_image_path}

        annotated_image_path = os.path.join(output_dir, f"{file_name}_an.png")
        draw.rectangle([0, 0, original_image.width - 1, original_image.height - 1], outline="green", width=2)
        draw.text((8, 8), f"{class_name} fallback", fill="blue", font=font)
        annotated_image.save(annotated_image_path)

        masked_image_path = os.path.join(output_dir, f"{file_name}_mask_{class_name}_0.png")
        mask = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        mask.save(masked_image_path)
        masked_images[0] = {"class_name":class_name,"score":0.0, "masked_image_path":masked_image_path}

        print(f"No YOLO boxes found. Fallback crop saved at: {cropped_image_path}")
        return cropped_images, masked_images

    # Iterate through detected objects and generate the cropped image file
    for idx, box in enumerate(results.boxes):
        # Get bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])
        class_idx = int(box.cls[0])
        class_name = normalize_class_name(detection_model.names[class_idx])  # Get the class name
        score = box.conf[0].item()  # Get the confidence score

        # Crop the image using bounding box from the original image
        cropped_image = original_image.crop((xmin, ymin, xmax, ymax))

        # Save the cropped image without modifications
        cropped_image_path = os.path.join(output_dir, f"{file_name}_crop_{class_name}_{idx}{file_extension}")
        cropped_image.save(cropped_image_path, format="PNG")

        ## convert the cropped image to 150 x 150 because the embedding image is also 150 x 150
        convert_to_high_quality_png(cropped_image_path, cropped_image_path,target_size=(150, 150), max_file_size=4 * 1024 * 1024)

        cropped_images[idx] = {"class_name":class_name,"score":score, "cropped_image_path":cropped_image_path}

        # Draw the bounding box on the annotated image
        draw.rectangle([xmin, ymin, xmax, ymax], outline="green", width=2)
        text = f"{class_name} {score:.2f}"
        draw.text((xmin, ymin - 10), text, fill="blue", font=font)

        print(f"Cropped object saved at: {cropped_image_path}")

    # Save the annotated image
    annotated_image_path = os.path.join(output_dir, f"{file_name}_an{file_extension}")
    annotated_image.save(annotated_image_path)
    print(f"Annotated image saved at: {annotated_image_path}")

    # Iterate through detected objects and generate the masks files
    for idx, box in enumerate(results.boxes):
        # Get bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

        # Adjust coordinates to fit within image bounds
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(original_image.width, xmax)
        ymax = min(original_image.height, ymax)

        # Create a blank mask image for the current object
        mask = Image.new("RGBA", original_image.size, (0, 0, 0, 255))  # Opaque black mask

        # Draw a filled polygon as the mask for the detected object
        draw = ImageDraw.Draw(mask)
        points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        draw.polygon(points, fill=(0, 0, 0, 0))  # Transparent fill inside polygon

        # Save the mask with a unique filename based on class and index
        class_idx = int(box.cls[0])
        class_name = normalize_class_name(detection_model.names[class_idx])
        score = box.conf[0].item()  # Get the confidence score

        mask_name = f"{file_name}_mask_{class_name}_{idx}{file_extension}"
        masked_image_path = os.path.join(output_dir, mask_name)

        masked_images[idx] = {"class_name":class_name,"score":score, "masked_image_path":masked_image_path}

        mask.save(masked_image_path)

        print(f"Mask saved at: {masked_image_path}")

    return cropped_images, masked_images


def validate_embeddings_prompts_matching(embeddings,prompts_list):
    prompts_keys = [e['image_file_name'] for e in prompts_list]
    embeddings_keys = [e['image_file_name'] for e in embeddings]
    return prompts_keys == embeddings_keys


detection_model = None
embeddings = load_embeddings(embeddings_file) or []
prompts, prompts_list = load_prompts(prompt_file)
prompts = prompts or {}
prompts_list = prompts_list or []
extractor = None
resources_ready = False
resources_loading = False
resources_error = None
resources_lock = Lock()

rt = validate_embeddings_prompts_matching(embeddings, prompts_list)
if not rt:
    print("Error: embeddings and prompts don't 1:1 ERROR !!!")
else:
    print("SUCCESS: embedding file names map to prompt map files")


def ensure_ml_resources():
    global detection_model, embeddings, prompts, prompts_list, extractor
    global resources_ready, resources_loading, resources_error

    if resources_ready:
        return

    with resources_lock:
        if resources_ready:
            return

        resources_loading = True
        resources_error = None
        try:
            from ultralytics import YOLO

            detection_model = YOLO(yolo_model_file)
            if not embeddings:
                embeddings = load_embeddings(embeddings_file) or []
            if not prompts_list:
                prompts, prompts_list = load_prompts(prompt_file)
                prompts = prompts or {}
                prompts_list = prompts_list or []
            extractor = FeatureExtractor("resnet50")

            rt = validate_embeddings_prompts_matching(embeddings, prompts_list)
            if not rt:
                print("Error: embeddings and prompts don't 1:1 ERROR !!!")
            else:
                print("SUCCESS: embedding file names map to prompt map files")

            resources_ready = True
        except Exception as exc:
            resources_error = str(exc)
            print(f"ERROR: ML resources failed to load: {exc}")
            raise
        finally:
            resources_loading = False


def warm_ml_resources():
    try:
        ensure_ml_resources()
    except Exception:
        pass


@app.on_event("startup")
async def start_background_model_warmup():
    if os.environ.get("WARM_ML_ON_STARTUP", "").lower() in {"1", "true", "yes"}:
        Thread(target=warm_ml_resources, daemon=True).start()


client = OpenAI(api_key=openai_api_key) if openai_api_key else None
if client:
    print("SUCCESS: OpenAI client configured")
else:
    print("WARNING: OPENAI_API_KEY is not configured. Try-on generation and AI Q&A will be disabled.")


@app.get("/")
async def index():
    return FileResponse(str(Path(STATIC_DIRECTORY) / "index.html"))


@app.get("/api/health")
async def health():
    catalog_count = len([p for p in Path(CATALOG_DIRECTORY).glob("*.png")]) if Path(CATALOG_DIRECTORY).exists() else 0
    model_classes = {normalize_class_name(name) for name in detection_model.names.values()} if detection_model else set()
    fashion_detector_loaded = len(model_classes & fashion_label_set) >= 5
    return {
        "status": "ok",
        "openai_configured": client is not None,
        "yolo_model": Path(yolo_model_file).name,
        "yolo_model_present": Path(yolo_model_file).exists(),
        "ml_ready": resources_ready,
        "ml_loading": resources_loading,
        "ml_error": resources_error,
        "fashion_detector_loaded": fashion_detector_loaded,
        "embeddings": len(embeddings or []),
        "prompts": len(prompts_list or []),
        "catalog_images": catalog_count,
        "similar_images_per_detection": number_of_similar_images,
    }


@app.get("/api/chat/rooms/{room_id}/messages")
async def list_chat_messages(room_id: str, limit: int = 100):
    room_key = safe_key(room_id)
    limit = max(1, min(limit, 300))
    with data_lock:
        data = load_json_file(CHAT_DATA_FILE, {})
        messages = data.get(room_key, [])[-limit:]
    return {"room": room_key, "messages": messages}


@app.post("/api/chat/rooms/{room_id}/messages")
async def create_chat_message(room_id: str, message: ChatMessageIn):
    room_key = safe_key(room_id)
    text = clean_text(message.text, 1000)
    sender = clean_text(message.sender, 40)
    if not text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    item = {
        "id": uuid.uuid4().hex,
        "room": room_key,
        "sender": sender,
        "text": text,
        "created_at": utc_ms(),
    }

    with data_lock:
        data = load_json_file(CHAT_DATA_FILE, {})
        room_messages = data.setdefault(room_key, [])
        room_messages.append(item)
        data[room_key] = room_messages[-500:]
        save_json_file(CHAT_DATA_FILE, data)

    return item


@app.get("/api/rewards")
async def get_rewards():
    with data_lock:
        data = load_json_file(REWARDS_DATA_FILE, default_rewards_state())
    return data


@app.post("/api/rewards/uploads")
async def create_reward_upload(
    challenge_id: str = Form(...),
    author: str = Form("Guest"),
    title: str = Form(""),
    image_url: str = Form(""),
    file: UploadFile | None = File(None),
):
    challenge_key = safe_key(challenge_id)
    author = clean_text(author, 40) or "Guest"
    title = clean_text(title, 120)
    image_url = clean_text(image_url, 1000)

    stored_file_url = None
    if file and file.filename:
        content_type = file.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Only image uploads are supported for challenges.")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            suffix = ".png"
        rewards_dir = Path(UPLOAD_DIRECTORY) / "rewards"
        rewards_dir.mkdir(parents=True, exist_ok=True)
        stored_path = rewards_dir / f"{uuid.uuid4().hex}{suffix}"
        bytes_written = 0
        with open(stored_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > 8 * 1024 * 1024:
                    raise HTTPException(status_code=400, detail="Image upload must be under 8 MB.")
                buffer.write(chunk)
        stored_file_url = public_upload_path(stored_path)

    if not stored_file_url and not image_url:
        raise HTTPException(status_code=400, detail="Upload an image file or paste an image URL.")

    upload = {
        "id": uuid.uuid4().hex,
        "challenge_id": challenge_key,
        "author": author,
        "title": title,
        "image_url": stored_file_url or image_url,
        "likes": 0,
        "comments": [],
        "created_at": utc_ms(),
    }

    with data_lock:
        data = load_json_file(REWARDS_DATA_FILE, default_rewards_state())
        data.setdefault("challenges", default_rewards_state()["challenges"])
        data.setdefault("uploads", [])
        if challenge_key not in data["challenges"]:
            data["challenges"][challenge_key] = {
                "id": challenge_key,
                "title": challenge_key.replace("-", " ").title(),
            }
        data["uploads"].insert(0, upload)
        data["uploads"] = data["uploads"][:300]
        save_json_file(REWARDS_DATA_FILE, data)

    return upload


@app.post("/api/rewards/uploads/{upload_id}/like")
async def like_reward_upload(upload_id: str):
    with data_lock:
        data = load_json_file(REWARDS_DATA_FILE, default_rewards_state())
        for upload in data.get("uploads", []):
            if upload["id"] == upload_id:
                upload["likes"] = int(upload.get("likes", 0)) + 1
                save_json_file(REWARDS_DATA_FILE, data)
                return upload
    raise HTTPException(status_code=404, detail="Upload not found.")


@app.post("/api/rewards/uploads/{upload_id}/comments")
async def comment_reward_upload(upload_id: str, comment: RewardCommentIn):
    item = {
        "id": uuid.uuid4().hex,
        "author": clean_text(comment.author, 40) or "Guest",
        "text": clean_text(comment.text, 500),
        "created_at": utc_ms(),
    }
    if not item["text"]:
        raise HTTPException(status_code=400, detail="Comment cannot be empty.")

    with data_lock:
        data = load_json_file(REWARDS_DATA_FILE, default_rewards_state())
        for upload in data.get("uploads", []):
            if upload["id"] == upload_id:
                upload.setdefault("comments", []).append(item)
                upload["comments"] = upload["comments"][-100:]
                save_json_file(REWARDS_DATA_FILE, data)
                return upload
    raise HTTPException(status_code=404, detail="Upload not found.")


@app.websocket("/ws/video/{room_id}/{peer_id}")
async def video_signaling(websocket: WebSocket, room_id: str, peer_id: str):
    room_key = safe_key(room_id)
    peer_key = safe_key(peer_id, default=uuid.uuid4().hex)
    await websocket.accept()

    peers = video_rooms.setdefault(room_key, {})
    peers[peer_key] = websocket

    async def broadcast(payload, include_self=False):
        stale = []
        for target_peer, target_socket in video_rooms.get(room_key, {}).items():
            if not include_self and target_peer == peer_key:
                continue
            try:
                await target_socket.send_json(payload)
            except RuntimeError:
                stale.append(target_peer)
        for stale_peer in stale:
            video_rooms.get(room_key, {}).pop(stale_peer, None)

    await broadcast({"type": "peer-joined", "from": peer_key, "peers": list(peers.keys())})
    try:
        while True:
            payload = await websocket.receive_json()
            if not isinstance(payload, dict):
                continue
            payload["from"] = peer_key
            payload["room"] = room_key
            target = payload.get("to")
            if target:
                target_socket = video_rooms.get(room_key, {}).get(safe_key(target))
                if target_socket:
                    await target_socket.send_json(payload)
            else:
                await broadcast(payload)
    except WebSocketDisconnect:
        pass
    finally:
        video_rooms.get(room_key, {}).pop(peer_key, None)
        await broadcast({"type": "peer-left", "from": peer_key})
        if not video_rooms.get(room_key):
            video_rooms.pop(room_key, None)


def extract_frames(video_path: str, output_dir: str):
    import cv2

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_{i}.png")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
    cap.release()
    return frames, fps


# def extract_frames(video_path: str, output_dir: str, frame_skip: int = 2):
#     cap = cv2.VideoCapture(video_path)
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frames = []

#     for i in range(0, frame_count, frame_skip):
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_path = os.path.join(output_dir, f"{os.path.basename(video_path)}_{i}.png")
#         cv2.imwrite(frame_path, frame)
#         frames.append(frame_path)
    
#     cap.release()
#     print(f"Extracted {len(frames)} frames")
#     return frames, fps

def extract_frame_number(filename):
    match = re.search(r'_(\d+)\.png', filename)
    if match:
        return int(match.group(1))
    else:
        return -1

def sort_frames(filenames):
    return sorted(filenames, key=extract_frame_number)


def merge_frames_to_video(frames: list, output_video_path: str, fps: float):
    import cv2

    sorted_frames = sort_frames(frames)

    frame = cv2.imread(sorted_frames[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for frame_path in sorted_frames:
        video.write(cv2.imread(frame_path))
    video.release()


# def merge_frames_to_video(frames: list, output_video_path: str, fps: float, frame_skip: int = 2):
#     adjusted_fps = fps / frame_skip
#     frame = cv2.imread(frames[0])
#     height, width, layers = frame.shape
#     video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), adjusted_fps, (width, height))
#     for frame_path in frames:
#         video.write(cv2.imread(frame_path))
#     video.release()

def append_elements(dict1, dict2):
    # Find the maximum key in dict1
    max_key = max(dict1.keys()) if dict1 else -1
    
    # Create a new dictionary to hold the merged results
    merged_dict = dict1.copy()
    
    # Add elements from dict2 to the merged_dict with new keys
    for i, value in enumerate(dict2.values(), start=max_key + 1):
        merged_dict[i] = value
        
    return merged_dict


def reencode_video(input_path: str, output_path: str):
    command = [
        'ffmpeg', '-y' , '-i', input_path, '-c:v', 'libx264', '-preset', 'fast', '-pix_fmt', 'yuv420p', '-movflags', 'faststart', output_path
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Video re-encoded to {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error re-encoding video: {e}")


def process_frame(frame_path, upload_directory, storage):
    frame_path_sqr = f"{frame_path}_sqr.png"
    convert_to_high_quality_png(frame_path, frame_path_sqr)
    cropped_images_temp, _ = generate_crop_mask_files(frame_path_sqr, upload_directory)
    storage["cropped_images"] = append_elements(storage["cropped_images"], cropped_images_temp)
    file_name_frame_path_sqr, _ = os.path.splitext(frame_path_sqr)
    return f"{file_name_frame_path_sqr}_an.png"

def process_video(input_file_path, upload_directory, storage, file, max_workers=None):
    print("Running YOLO8 on frames of a video ...")
    frames, fps = extract_frames(input_file_path, upload_directory)
    annotated_frames = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_frame = {executor.submit(process_frame, frame, upload_directory, storage): frame for frame in frames}
        
        for future in as_completed(future_to_frame):
            try:
                annotated_frame = future.result()
                annotated_frames.append(annotated_frame)
            except Exception as exc:
                print(f'Frame processing generated an exception: {exc}')

    output_video_path = os.path.join(upload_directory, f"{file.filename}_an.mp4")
    merge_frames_to_video(annotated_frames, output_video_path, fps)

    reencoded_video_path = os.path.join(upload_directory, f"{file.filename}_reencoded.mp4")
    reencode_video(output_video_path, reencoded_video_path)
    return reencoded_video_path

    # return output_video_path


def reduce_fps(input_path, output_path, target_fps):
    from moviepy.editor import VideoFileClip

    # Load the original video
    video = VideoFileClip(input_path)
    
    # Set the desired FPS
    video = video.set_fps(target_fps)
    
    # Write the result to a new file with the specified codec
    video.write_videofile(output_path, codec='libx264')


@app.post("/detect_classes")
async def detect_classes(file: UploadFile = File(...)):

    global storage

    print(f"===> detect_classes({file.filename})")

    classes_with_ids = {}
    # clear and copy to global storage every time this API is called
    storage = {"cropped_images": {},"similar_images": {}}

    content_type = file.content_type or ""
    input_file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())

    if content_type.startswith('image'):
        print("Running YOLO8 on single image ...")
        # client/server path
        # input_file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
        input_file_path_sqr = os.path.join(UPLOAD_DIRECTORY, f"{file.filename}_sqr.png")

        # convert the input file to png image of 1024 x 1024 (required for Dall-e for editing)
        convert_to_high_quality_png(input_file_path, input_file_path_sqr)

        # this function will call the yolo model and crop the detected images
        cropped_images_temp, _  = generate_crop_mask_files(input_file_path_sqr, UPLOAD_DIRECTORY)
        storage["cropped_images"].update(cropped_images_temp)

        # for handling detection of multiple same class
        labels_dict = populate_label_dict()

        for idx in storage["cropped_images"]:
            class_name = storage["cropped_images"][idx]["class_name"]
            score = storage["cropped_images"][idx]["score"]
            labels_dict.setdefault(class_name, 0)
            if labels_dict[class_name]:
                display_class_name = f"{class_name}_{labels_dict[class_name]} ({round(score,4)})"
            else:
                display_class_name = f"{class_name} ({round(score,4)})"

            classes_with_ids[display_class_name] = idx
            labels_dict[class_name] +=1

    elif content_type.startswith('video'):
        input_file_path_reduced_fps = os.path.join(UPLOAD_DIRECTORY, f"{file.filename}_reduced_fps.mp4")
        reduce_fps(input_file_path, input_file_path_reduced_fps, REDUCED_FPS)

        if PARALLEL_PROCESSING:
            print("Running YOLO8 on frames of a video in parallel ...")
            # max_workers = 20  # Set the desired number of parallel threads
            output_video_path = process_video(input_file_path_reduced_fps, UPLOAD_DIRECTORY, storage, file, max_workers)
        else:
            print("Running YOLO8 on frames of a video in sequential ...")
            frames, fps = extract_frames(input_file_path_reduced_fps, UPLOAD_DIRECTORY)
            annotated_frames = []
            for frame_path in frames:
                frame_path_sqr = f"{frame_path}_sqr.png"
                convert_to_high_quality_png(frame_path, frame_path_sqr)
                cropped_images_temp, _ = generate_crop_mask_files(frame_path_sqr, UPLOAD_DIRECTORY)

                storage["cropped_images"] = append_elements(storage["cropped_images"],cropped_images_temp)

                file_name_frame_path_sqr, _ = os.path.splitext(frame_path_sqr)
                annotated_frames.append(f"{file_name_frame_path_sqr}_an.png")

            output_video_path = os.path.join(UPLOAD_DIRECTORY, f"{file.filename}_an.mp4")
            merge_frames_to_video(annotated_frames, output_video_path, fps)

            reencoded_video_path = os.path.join(UPLOAD_DIRECTORY, f"{file.filename}_reencoded.mp4")
            reencode_video(output_video_path, reencoded_video_path)


        # for handling detection of multiple same class
        labels_dict = populate_label_dict()
        print(storage["cropped_images"])

        for idx in storage["cropped_images"]:
            class_name = storage["cropped_images"][idx]["class_name"]
            score = storage["cropped_images"][idx]["score"]
            labels_dict.setdefault(class_name, 0)
            if not labels_dict[class_name]:
                display_class_name = f"{class_name} ({round(score,4)})"
                classes_with_ids[display_class_name] = idx
            labels_dict[class_name] +=1
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")


    if content_type.startswith('image'):
        response_image_path = f"/uploads/{file.filename}_sqr_an.png"
        file_type = "image"
    else:
        # response_image_path = f"/uploads/{file.filename}_an.mp4"
        response_image_path = f"/uploads/{file.filename}_reencoded.mp4"
        file_type = "video"

    print(f"<=== detect_classes({response_image_path},{classes_with_ids})")

    # if file.content_type.startswith('image'):
    #     response_image_path = "/uploads/3.jpg_sqr_an.png"
    #     file_type = "image"
    # else:
    #     response_image_path = "/uploads/xyz.mp4_an.mp4"
    #     file_type = "video"

    # classes_with_ids = {}

    return JSONResponse(content={"image": response_image_path, "file_type": file_type, "classes": classes_with_ids})


@app.post("/fetch_similar_images")
async def fetch_similar_images(selected_ids: List[int]):

    global storage
    global prompts


    print(f"===> fetch_similar_images({selected_ids})")

    input_images = []
    input_classes = []
    for _id in selected_ids:
        input_images.append((storage["cropped_images"][_id]["cropped_image_path"],storage["cropped_images"][_id]["class_name"]))
        input_classes.append(storage["cropped_images"][_id]["class_name"])

    input_classes_str = ",".join(input_classes)
    print(f"++++ fetch_similar_images for classes:{input_classes_str}")

    similar_images_temp = get_similar_images(embeddings,input_images,prompts,number_of_similar_images)

    # clear and refresh the global storage every time this API is called    
    storage["similar_images"] =  {}
    storage["similar_images"].update(similar_images_temp)

    return JSONResponse(content={"similar_images": storage["similar_images"]})


@app.post("/imerse_image")
# async def imerse_image(selected_image_ids: List[int]):
async def imerse_image(selected_image_id: int = Form(...), file: UploadFile = File(...)):

    global storage

    print(f"===> imerse_image({selected_image_id,file.filename})")

    if client is None:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")

    if selected_image_id not in storage["similar_images"]:
        raise HTTPException(status_code=404, detail="Similar image not found. Run detection and choose a catalog item first.")

    # full input file path
    input_file_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    input_file_path_sqr = os.path.join(UPLOAD_DIRECTORY, f"{file.filename}_sqr.png")

    # Save the uploaded file
    with open(input_file_path, "wb") as buffer:
        buffer.write(await file.read())

    # convert the input file to png image of 1024 x 1024 (required for Dall-e for editing)
    convert_to_high_quality_png(input_file_path, input_file_path_sqr)

    # this function will call the yolo model and crop the detected images
    _, masked_images_temp  = generate_crop_mask_files(input_file_path_sqr, UPLOAD_DIRECTORY)

    original_class = storage["similar_images"][selected_image_id]["class_name"]
    selected_image = storage["similar_images"][selected_image_id]["image_file_name"]

    print(f"get the prompt for this image:{selected_image}")

    input_masked_image_path = None

    your_classes = list(set(image_info["class_name"] for image_info in masked_images_temp.values()))

    for key in masked_images_temp.keys():
        if masked_images_temp[key]["class_name"] == original_class:
            input_masked_image_path = masked_images_temp[key]["masked_image_path"]
            break

    print(f"classes detected in the try out image:{your_classes}")

    try:
        if input_masked_image_path is None:
            print("The class of dress you selected is not yoloed in your image over which you are trying the dress.")
            return JSONResponse(content={"image_details": None})
           
        input_image = input_file_path_sqr
        masked_image_path = input_masked_image_path
        prompt = storage["similar_images"][selected_image_id]["prompt"]

        print(input_image)
        print(masked_image_path)
        print(prompt)

        with open(input_image, "rb") as image_file, open(masked_image_path, "rb") as mask_file:
            edit_response = client.images.edit(
                model=os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1"),
                image=image_file,
                mask=mask_file,
                prompt=prompt,
                size="1024x1024",
            )

        output_name = f"{Path(input_file_path_sqr).name}_imerse.png"
        imerse_image_file = os.path.join(UPLOAD_DIRECTORY, output_name)

        print(imerse_image_file)

        image_data = edit_response.data[0]
        if getattr(image_data, "b64_json", None):
            with open(imerse_image_file, "wb") as buffer:
                buffer.write(base64.b64decode(image_data.b64_json))
            return JSONResponse(content={"image_details": f"/uploads/{output_name}"})

        if getattr(image_data, "url", None):
            response = requests.get(image_data.url, stream=True)
            if response.status_code == 200:
                with open(imerse_image_file, "wb") as buffer:
                    response.raw.decode_content = True
                    shutil.copyfileobj(response.raw, buffer)

                return JSONResponse(content={"image_details": f"/uploads/{output_name}"})

        raise HTTPException(status_code=500, detail="Image edit API returned no image data")

    except Exception as e:
        print(f"error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while calling image editing API: {e}")


@app.post("/ask_me", response_model=dict)
async def ask_me(selected_image_id: int = Body(...), text: str = Body(...)):

    if client is None:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")

    if selected_image_id not in storage["similar_images"]:
        raise HTTPException(status_code=404, detail="Image not found.")

    image_file_path = storage["similar_images"][selected_image_id]["image_file_name"]
    prompt = storage["similar_images"][selected_image_id]["prompt"]

    # Extract just the file name from the file path
    image_file_name = os.path.basename(image_file_path)

    # Read the Q&A file associated with the image
    common_qna_text, common_file = read_common_qna_file(common_qna_file)
    qna_text, qna_file = read_qna_file(image_file_name)

    print(f"Retrieving review question: {text}")
    print(f"Retrieving review question for prompt: {prompt}")
    print(f"From general question from common QnA file: {common_file}")
    print(f"From specific question for file from: {qna_file}")

    if common_qna_text or qna_text:
        messages = [
            {"role": "user", "content": f"The production description of this fashion item is: {prompt}"},
            {"role": "user", "content": f"Here are some previous questions and answers for this product: {common_qna_text}\n{qna_text}"},
            {"role": "user", "content": f"Based on the product description and previous Q&A, could you briefly answer the following question: {text}"},
        ]

        answer, error_msg = get_information_openai(client, messages,custom_params)

        if error_msg:
            print(f"Chat GPT returned error: {error_msg}")
            answer = "System failed to get an answer."
    else:
        answer = "No information available."

    # Return JSONResponse with the answer
    return {"answer": answer}


@app.post("/ask_me_voice", response_model=dict)
async def ask_me_voice(selected_image_id: int = Body(...), file: UploadFile = File(...)):

    if client is None:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY is not configured.")

    with open(file.filename, "wb") as buffer:
        buffer.write(file.file.read())

    audio_input = open(file.filename, "rb")

    # Decode audio
    text = convert_audio_to_text(client,audio_input)

    if selected_image_id not in storage["similar_images"]:
        raise HTTPException(status_code=404, detail="Image not found.")

    image_file_path = storage["similar_images"][selected_image_id]["image_file_name"]
    prompt = storage["similar_images"][selected_image_id]["prompt"]

    # Extract just the file name from the file path
    image_file_name = os.path.basename(image_file_path)

    # Read the Q&A file associated with the image
    common_qna_text, common_file = read_common_qna_file(common_qna_file)
    qna_text, qna_file = read_qna_file(image_file_name)

    print(f"Retrieving review question: {text}")
    print(f"Retrieving review question for prompt: {prompt}")
    print(f"From general question from common QnA file: {common_file}")
    print(f"From specific question for file from: {qna_file}")

    if common_qna_text or qna_text:
        messages = [
            {"role": "user", "content": f"The production description of this fashion item is: {prompt}"},
            {"role": "user", "content": f"Here are some previous questions and answers for this product: {common_qna_text}\n{qna_text}"},
            {"role": "user", "content": f"Based on the product description and previous Q&A, could you briefly answer the following question. Keep responses under 20 words.: {text}"},
        ]

        answer, error_msg = get_information_openai(client, messages,custom_params)

        if error_msg:
            print(f"Chat GPT returned error: {error_msg}")
            answer = "System failed to get an answer."
    else:
        answer = "No information available."

    # Convert chat response to audio
    convert_text_to_speech(answer)

    response_audio_file = f"uploads/output.mp3"

    # Return JSONResponse with the answer
    return FileResponse(response_audio_file, media_type='audio/wav')
    # return {"answer": answer}



# Open AI - Whisper
# Convert audio to text
def convert_audio_to_text(client,audio_file):
  try:
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    message_text = transcript.text
    return message_text
  except Exception as e:
    return "this information is not available"


# Convert text to speech
def convert_text_to_speech(message):
    try:
        tts = gTTS(text=message, lang='en')
        tts.save(f"./uploads/output.mp3")
        # with open("output.mp3", "rb") as audio_file:
        #     audio_data = audio_file.read()
        # os.remove("output.mp3")  # Clean up the temporary file
        # return audio_data
    except Exception as e:
        print(f"An error occurred: {e}")
        # return


# Function to check and read the Q&A text file
def read_common_qna_file(common_qna_file):

    print(f"Fetching Common Q&A from file: {common_qna_file}")

    if os.path.exists(common_qna_file):
        with open(common_qna_file, 'r') as file:
            return file.read().strip(), common_qna_file
    else:
        return None, None

# Function to check and read the Q&A text file
def read_qna_file(image_file_name):

    qna_file_path = f"{qna_directory}/{image_file_name.split('.')[0]}.txt"

    print(f"Fetching Q&A from file: {qna_file_path}")

    if os.path.exists(qna_file_path):
        with open(qna_file_path, 'r') as file:
            return file.read().strip(), qna_file_path
    else:
        return None, None


def get_information_openai(client,messages,custom_params):

    print("Calling get_information_openai with messages:", messages)

    try:
        response = client.chat.completions.create(**custom_params, messages=messages)
        response_content = response.choices[0].message.content
        print("ChatGPT returned successfully:", response_content)
        return response_content, None

    except Exception as e:
        print(f"ChatGPT returned unsuccessfully: {e}")
        return None, f"{e}"
