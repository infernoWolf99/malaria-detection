from pathlib import Path

import torch, os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MainDataset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transforms = transforms

        # List all image files
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load Image
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Match with corresponding YOLO label text file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    # YOLO lines look like: "0 0.512 0.341 0.12 0.18"
                    class_id, x_center, y_center, w, h = map(
                        float, line.strip().split()
                    )

                    # Convert normalized x_center, y_center, w, h to absolute xmin, ymin, xmax, ymax
                    xmin = (x_center - w / 2) * width
                    ymin = (y_center - h / 2) * height
                    xmax = (x_center + w / 2) * width
                    ymax = (y_center + h / 2) * height

                    box_w = xmax - xmin
                    box_h = ymax - ymin

                    # Only keep the box if it has a real width and height (greater than 1 pixel)
                    if box_w > 1.0 and box_h > 1.0:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(int(class_id) + 1)
                    else:
                        # Log it quietly so you know if your dataset has corrupted annotations
                        print(
                            f"Dropped degenerate box in {img_name}: w={box_w:.2f}, h={box_h:.2f}"
                        )

        # Handle images that might have no Trophozoites (empty backgrounds)
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # Structure target dictionary exactly how PyTorch RetinaNet wants it
        target = {"boxes": boxes, "labels": labels}

        # Convert PIL image to Tensor basic processing
        if self.transforms:
            image = self.transforms(image)
        else:
            from torchvision.transforms import functional as F

            image = F.to_tensor(image)

        return image, target


class DataPrep:
    def __init__(self, name: str, root_path: Path, batch_size: int = 4, pin_memory: bool = False, num_workers: int= 2):
        self.root_path = root_path
        self.batch_size = batch_size
        self.name = name
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def build_datasets(self):
        # in case only the dataset is needed else where
        train_dataset = MainDataset(
            img_dir=os.path.join(self.root_path, "images", "train"),
            label_dir=os.path.join(self.root_path, "labels", "train"),
        )

        val_dataset = MainDataset(
                img_dir=os.path.join(self.root_path, "images", "val"),
                label_dir=os.path.join(self.root_path, "labels", "val"),
            )

        test_dataset = MainDataset(
                img_dir=os.path.join(self.root_path, "images", "test"),
                label_dir=os.path.join(self.root_path, "labels", "test"),
            )

        self._print_dataset_details(self.name, train_dataset, val_dataset, test_dataset)

        return train_dataset, val_dataset, test_dataset

    def _print_dataset_details(self, name, train, val, test):
        print(f'{"=" * 30} {name} {"=" * 30}')
        print(f'Train Size: {len(train)}')
        print(f"Val Size: {len(val)}")
        print(f"Test Size: {len(test)}")
        print(f"Loaded a total of :  {len(train) + len(train) + len(train)} ")

    def build_loaders(self):
        collator = lambda x: tuple(zip(*x))

        train_set, val_set, test_set = self.build_datasets()

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
            )
        test_loader = DataLoader(
            test_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        return train_loader, val_loader, test_loader
