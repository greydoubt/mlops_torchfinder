import os
import argparse
import zipfile
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from preprocessing.transform import get_transforms
from deep_learning.model import Model
from pathfinding.astar import AStarPathFinder
from utils.io import get_files_with_extension, load_image, save_image
from utils.visualization import draw_path_on_image


def infer_on_image(image_path, model_path, output_dir, device):
    # Load image
    image = load_image(image_path)

    # Load model
    model = Model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # Transform image
    transform = get_transforms()
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    output = output.squeeze().cpu().numpy()

    # Find path
    path_finder = AStarPathFinder()
    path = path_finder.find_path(output)

    # Draw path on image
    drawn_image = draw_path_on_image(image, path)

    # Save original image and drawn image
    basename = os.path.basename(image_path)
    name, ext = os.path.splitext(basename)
    save_image(image, os.path.join(output_dir, f"{name}_original{ext}"))
    save_image(drawn_image, os.path.join(output_dir, f"{name}_path{ext}"))


def infer_on_folder(folder_path, model_path, output_dir, device):
    image_paths = get_files_with_extension(folder_path, (".jpg", ".jpeg", ".png"))
    for image_path in tqdm(image_paths, desc="Processing images"):
        infer_on_image(image_path, model_path, output_dir, device)


def infer_on_zip(zip_path, model_path, output_dir, device):
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        image_paths = [
            f for f in zip_file.namelist() if f.endswith((".jpg", ".jpeg", ".png"))
        ]
        for image_path in tqdm(image_paths, desc="Processing images"):
            with zip_file.open(image_path) as f:
                image = Image.open(f).convert("RGB")
                # Save temporary image to disk and infer on it
                temp_path = os.path.join(output_dir, "temp.jpg")
                image.save(temp_path)
                infer_on_image(temp_path, model_path, output_dir, device)
                # Remove temporary image from disk
                os.remove(temp_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer path on an image or folder of images.")
    parser.add_argument("input_path", help="Path to an image file, folder of images, or zip file.")
    parser.add_argument("model_path", help="Path to the trained model file.")
    parser.add_argument("--output-dir", default="output", help="Path to the output directory.")
    parser.add_argument("--device", default="cpu", help="Device to run inference on (default: cpu).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.isfile(args.input_path):
        infer_on_image(args.input_path, args.model_path, args.output_dir, args.device)
    elif os.path.isdir(args.input_path):
        infer_on_folder(args.input_path, args.model_path, args.output_dir, args.device)
    elif os.path.isfile(args.input_path) and args.input_path.endswith(".zip"):
        infer_on_zip(args.input_path, args.model_path,
