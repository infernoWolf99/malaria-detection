from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_loader import DataPrep


def visualize_boxes(images, targets, class_mapping=None, max_images=4):
    """
    Visualizes images alongside their respective ground truth bounding boxes.
    """

    if isinstance(images, torch.Tensor) and len(images.shape) == 3:
        images = [images]
    if isinstance(targets, dict):
        targets = [targets]

    num_to_show = min(len(images), max_images)
    fig, axes = plt.subplots(num_to_show, 2, figsize=(10, 5 * num_to_show))

    if num_to_show == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_to_show):
        img = images[i].detach().cpu()
        target = targets[i]
        ax = axes[i]

        img_np = img.permute(1, 2, 0).numpy()

        if img_np.min() < 0 or img_np.max() <= 1.0:
            img_np = np.clip(img_np, 0, 1)

        ax_clean = axes[i, 0]
        ax_clean.imshow(img_np)
        ax_clean.axis("off")
        if i == 0:
            ax_clean.set_title("Raw Blood Smear Image", fontsize=12, pad=10)

        # 2. Right Plot: Image with Bounding Boxes
        ax_boxes = axes[i, 1]
        ax_boxes.imshow(img_np)
        ax_boxes.axis("off")
        if i == 0:
            ax_boxes.set_title("Ground Truth Annotations", fontsize=12, pad=10)

        boxes = target["boxes"].detach().cpu().numpy()
        labels = target["labels"].detach().cpu().numpy()

        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin

            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor="red" if label == 1 else "magenta",
                facecolor="none",
            )
            ax_boxes.add_patch(rect)

            class_name = (
                class_mapping.get(int(label), f"Class {label}")
                if class_mapping
                else f"Class {label}"
            )
            ax_boxes.text(
                xmin,
                ymin - 4,
                class_name,
                color="white",
                fontsize=9,
                weight="bold",
                bbox=dict(
                    facecolor="red" if label == 1 else "magenta", alpha=0.6, pad=1
                ),
            )

    plt.tight_layout()
    plt.show()

data_dir = Path("./data/trophozoite-wbc")

t_wbc_data = DataPrep(
    name="Trophozoite plus WBC dataset",
    root_path=data_dir,
    batch_size=2,
    num_workers=2,
)

train, val, test = t_wbc_data.build_loaders()
images, boxes = next(iter(train))
label_map = {1: "Trophozoite", 2: "WBC"}


visualize_boxes(images=images, targets=boxes, class_mapping=label_map, max_images=5)
