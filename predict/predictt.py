import os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from src.data_pipeline.data_loader import load_data
from src.models.base.resnet import ResNet50FeatureExtractor

# ---------------- Configuration ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 3

# תיקיית שורש
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "saved_model.pth")

# נתיבי תמונות וקבצי תוצאה
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "PREDICT", "predict_photos")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "PREDICT", "predict_results")

# צבעים לפי דירוג
COLORS = [(0, 128, 0), (255, 165, 0), (255, 0, 0)]  # ירוק, כתום, אדום

# ---------------- Helpers ----------------
def get_color(rank):
    if rank < len(COLORS):
        return COLORS[rank]
    return (0,0,0)

def create_result_image(image, predictions, save_path):
    """יוצר תמונה סופית עם הכלב במרכז והטקסט למטה"""
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except:
        font = ImageFont.load_default()

    # Canvas קבוע
    canvas_width = 500
    canvas_height = 600

    # שמירת יחס גובה-רוחב של התמונה
    img_ratio = image.width / image.height
    max_img_height = 400
    max_img_width = 400
    if img_ratio > 1:  # רחבה
        new_w = min(max_img_width, image.width)
        new_h = int(new_w / img_ratio)
    else:  # גבוהה
        new_h = min(max_img_height, image.height)
        new_w = int(new_h * img_ratio)

    image_resized = image.resize((new_w, new_h))

    # יצירת Canvas חדש
    new_img = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    x_offset = (canvas_width - new_w)//2
    y_offset = 20
    new_img.paste(image_resized, (x_offset, y_offset))

    # ציור הטקסט
    draw = ImageDraw.Draw(new_img)
    spacing = 10
    y_start = y_offset + new_h + 20

    for idx, (cls, prob) in enumerate(predictions):
        text = f"{cls}: {prob:.1f}%"
        color = get_color(idx)

        bbox = draw.textbbox((0,0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        x = (canvas_width - text_w)//2
        draw.text((x, y_start + idx * (text_h + spacing)), text, fill=color, font=font)

    new_img.save(save_path)

def predict_image(feature_extractor, classifier, image_path, classes, transform):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    feature_extractor.eval()
    classifier.eval()
    with torch.no_grad():
        features = feature_extractor(input_tensor).view(input_tensor.size(0), -1)
        output = classifier(features)
        probs = torch.softmax(output, dim=1).squeeze()

    top_probs, top_idx = torch.topk(probs, TOP_K)
    top_classes = [classes[i] for i in top_idx.tolist()]
    top_probs = (top_probs * 100).tolist()
    return image, list(zip(top_classes, top_probs))

# ---------------- Main ----------------
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load classes
    _, _, _, classes = load_data()

    # Feature extractor
    feature_extractor = ResNet50FeatureExtractor().to(DEVICE)

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # דגימת תמונה ראשונה כדי לקבוע input_dim
    image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".png",".jpg",".jpeg"))]
    if not image_files:
        print(f"❌ No images found in {INPUT_FOLDER}")
        return

    first_image = Image.open(os.path.join(INPUT_FOLDER, image_files[0])).convert("RGB")
    input_tensor = transform(first_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sample_features = feature_extractor(input_tensor).view(input_tensor.size(0), -1)
        input_dim = sample_features.size(1)

    # Classifier
    classifier = torch.nn.Linear(input_dim, len(classes)).to(DEVICE)
    classifier.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

    # Predict
    print(f"\n🚀 Starting prediction on folder: {INPUT_FOLDER}\n")
    for filename in image_files:
        image_path = os.path.join(INPUT_FOLDER, filename)
        image, predictions = predict_image(feature_extractor, classifier, image_path, classes, transform)
        save_path = os.path.join(OUTPUT_FOLDER, filename)
        create_result_image(image, predictions, save_path)
        print(f"✅ {filename} → {[f'{cls}: {prob:.1f}%' for cls, prob in predictions]}")

    print(f"\n🎉 All results saved in: {OUTPUT_FOLDER}\n")

if __name__ == "__main__":
    main()
