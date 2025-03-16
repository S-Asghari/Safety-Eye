### Install & Import the Requirements

!pip install fiftyone
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone.core.labels import Classification

!pip install roboflow
from roboflow import Roboflow

import os
import shutil
import json

### Load the Data

rf = Roboflow(api_key="NpeBlgiGKRnkLj4kIL2C")
project = rf.workspace("safewalkbd").project("safewalkbd-l8jbn")
version = project.version(9)
dataset = version.download("coco")

dataset_dir = "/content/SafeWalkBD-9"

for split in ["train", "valid", "test"]:
    split_path = os.path.join(dataset_dir, split)
    data_path = os.path.join(split_path, "data")

    # Create 'data' folder if it doesn't exist
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Move all image files into 'data/' subfolder
    for file in os.listdir(split_path):
        if file.endswith(".jpg") or file.endswith(".png"):  # Adjust for other image formats if needed
            shutil.move(os.path.join(split_path, file), os.path.join(data_path, file))

print("Folder structure updated successfully.")

# Load the training dataset
train_dataset = fo.Dataset.from_dir(
    dataset_dir=f"{dataset_dir}/train",
    dataset_type=fo.types.COCODetectionDataset,
    labels_path=f"{dataset_dir}/train/_annotations.coco.json"
)

# Load the validation dataset
valid_dataset = fo.Dataset.from_dir(
    dataset_dir=f"{dataset_dir}/valid",
    dataset_type=fo.types.COCODetectionDataset,
    labels_path=f"{dataset_dir}/valid/_annotations.coco.json"
)

# Load the test dataset
test_dataset = fo.Dataset.from_dir(
    dataset_dir=f"{dataset_dir}/test",
    dataset_type=fo.types.COCODetectionDataset,
    labels_path=f"{dataset_dir}/test/_annotations.coco.json"
)

# Launch FiftyOne app for visualization
session = fo.launch_app(train_dataset)


# Load annotation JSON file
with open(f"{dataset_dir}/train/_annotations.coco.json", "r") as f:
    data = json.load(f)

# Print top-level keys
print(data.keys())

# Extract category names
category_names = [category["name"] for category in data["categories"] if category["name"] != 'car-vehicle-dog-animal-curb-wall']

# Print the list of category names
print(category_names)

### Test Zero-Shot Model Performance on Test Set

model = foz.load_zoo_model(
    "zero-shot-detection-transformer-torch",
    name_or_path="google/owlvit-base-patch32",
    classes=category_names,
)

# for category in category_names:
#     test_dataset.apply_model(
#         model,
#         label_field=category,  # Apply the model for each category
#         confidence_thresh=0.5,
#         batch_size=16
#     )

test_dataset.apply_model(
    model,
    label_field="Obstacle",
    confidence_thresh=0.5,
    batch_size=16
)

session = fo.launch_app(test_dataset)

with open(f"{dataset_dir}/test/_annotations.coco.json", "r") as f:
    coco_annotations = json.load(f)

# Create ground_truth and predicted labels
for sample in test_dataset:
    # Extract filename
    image_filename = sample.filepath.split("/")[-1]

    # Find image_id
    image_id = next((img["id"] for img in coco_annotations["images"] if img["file_name"] == image_filename), None)
    if image_id is None:
        print(f"Image ID not found for {image_filename}")
        continue

    # Check if any annotation has category_id = 3 (Obstacle)
    has_obstacle_gt = any(ann["image_id"] == image_id and ann["category_id"] == 3 for ann in coco_annotations["annotations"])
    sample["ground_truth"] = 1 if has_obstacle_gt else 0

    # Check model predictions (assuming the model is stored in label field "Obstacle")
    predicted_obstacles = [det for det in sample.detections.detections if det.label == "Obstacle"]
    sample["has_obstacle"] = 1 if predicted_obstacles else 0

    sample.save()

# Convert integer labels to Classification objects and store in new fields
for sample in test_dataset:
    sample["ground_truth_cls"] = Classification(label=str(sample["ground_truth"]))
    sample["has_obstacle_cls"] = Classification(label=str(sample["has_obstacle"]))
    sample.save()

# Evaluate the model using classification metrics
results = test_dataset.evaluate_classifications(
    pred_field="has_obstacle_cls",  # New field for model predictions
    gt_field="ground_truth_cls",    # New field for ground truth labels
    eval_key="obstacle_eval",       # Name for storing results
)

# Print evaluation results
print("Classification report:")
print(results.report())
