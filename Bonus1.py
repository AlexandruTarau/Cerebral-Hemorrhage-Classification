import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from RSNADataset import RSNADataset
from Preprocess import Preprocess


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_model(num_classes=6):
    model = models.efficientnet_b0(weights=None)
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, old_conv.out_channels, 3, stride=2, padding=1, bias=False)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with open("config.json", "r") as f:
    CONFIG = json.load(f)

random_seed(CONFIG['seed'])

# Load model
model_path = "../models_etapa2_final/FinalModel_fixed/fold_2/best_model.pth"
model = get_model()
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state'])
model = model.to(device)
model.eval()

# Init GradCAM
output_dir = "../models_etapa2_final/FinalModel_fixed/fold_2/gradcam_results"
os.makedirs(output_dir, exist_ok=True)

class_names = ['any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']
counts = {cls: {'correct': 0, 'wrong': 0} for cls in class_names}
target_count = 5
threshold = 0.42

target_layers = [model.features[-1]]  # Last layer
cam = GradCAM(model=model, target_layers=target_layers)

# Test dataset
transform = Preprocess(config_path='config.json', training=False)

test_dataset = RSNADataset(
    data_dir=CONFIG['test_images'],
    csv_file=CONFIG['test_csv'],
    transform=transform
)

# Generate the images
for i in tqdm(range(len(test_dataset))):
    # Check if we got all necessary images
    total_collected = sum([c['correct'] + c['wrong'] for c in counts.values()])
    if total_collected >= len(class_names) * target_count * 2:
        break

    image_tensor, target_labels = test_dataset[i]
    input_batch = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)
        preds = torch.sigmoid(output).squeeze().cpu().numpy()

    targets = target_labels.numpy()
    pred_labels = (preds > threshold).astype(int)

    for idx, cls in enumerate(class_names):
        is_correct = (pred_labels[idx] == targets[idx])
        status = 'correct' if is_correct else 'wrong'

        # Save image if needed
        if counts[cls][status] < target_count:
            with torch.enable_grad():
                cam_targets = [ClassifierOutputTarget(idx)]
                grayscale_cam = cam(input_tensor=input_batch, targets=cam_targets)[0, :]

                # Normalize image
                img_np = image_tensor.squeeze().numpy()
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                rgb_img = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                rgb_img = np.float32(rgb_img) / 255

                vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

                file_name = f"{cls}_{status}_{counts[cls][status]}.png"
                cv2.imwrite(os.path.join(output_dir, file_name), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

                counts[cls][status] += 1