import os
import json

def process_labels(json_file_path, img_width, img_height, train_images, val_images, train_labels_dir, val_labels_dir):
    # Read the JSON file for annotations
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Dictionary to store image labels
    image_labels = {}

    # Collect annotations
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']
        x_min, y_min, width, height = bbox

        # Convert to YOLO format (normalize by image size)
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height

        # Collect labels for each image
        if image_id not in image_labels:
            image_labels[image_id] = []
        image_labels[image_id].append(f"{category_id} {x_center} {y_center} {norm_width} {norm_height}")

    # Save label files in the appropriate folder
    for image_id, labels in image_labels.items():
        # Format the image name
        image_name = f"{image_id:04d}.jpg"  # Example: 2882 -> 2882.jpg

        # Determine if the image is in train or val set
        if image_name in train_images:
            label_file_path = os.path.join(train_labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
        elif image_name in val_images:
            label_file_path = os.path.join(val_labels_dir, f"{os.path.splitext(image_name)[0]}.txt")
        else:
            continue  # Skip if the image is not found in either set

        # Write labels to file
        with open(label_file_path, 'w') as label_file:
            for label in labels:
                label_file.write(label + '\n')

    print(f"Processed labels and saved to {train_labels_dir} and {val_labels_dir}.")
