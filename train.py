"""Train face recognition model on raw data using MindSpore with augmentation."""
import os
import sys
import random
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.dataset import GeneratorDataset

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

DATA_DIR = "raw"
MODEL_DIR = "models"
IMG_SIZE = 112
EMBEDDING_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
AUGMENT_FACTOR = 4  # create 4 augmented copies per image

os.makedirs(MODEL_DIR, exist_ok=True)


def augment_image(img):
    """Apply random augmentations including camera-simulating artifacts."""
    w, h = img.size

    # 1. Brightness variation
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # 2. Contrast variation
    factor = random.uniform(0.7, 1.3)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # 3. Rotation (-12 to 12 degrees)
    angle = random.uniform(-12, 12)
    img = img.rotate(angle, Image.BILINEAR, fillcolor=0)

    # 4. Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # 5. Random zoom (crop and resize)
    zoomed = False
    if random.random() < 0.3:
        scale = random.uniform(0.85, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        left = random.randint(0, max(w - new_w, 1))
        top = random.randint(0, max(h - new_h, 1))
        img = img.crop((left, top, left + new_w, top + new_h))
        zoomed = True

    # 6. Random background padding (simulate loose face detection box)
    if random.random() < 0.4:
        scale = random.uniform(0.75, 0.95)
        new_size = int(IMG_SIZE * scale)
        small = img.resize((new_size, new_size), Image.BILINEAR)
        canvas = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        offset_x = random.randint(0, IMG_SIZE - new_size)
        offset_y = random.randint(0, IMG_SIZE - new_size)
        canvas.paste(small, (offset_x, offset_y))
        img = canvas
    elif not zoomed:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    else:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

    # Convert to numpy [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # 7. Random Gaussian noise (simulate camera sensor noise)
    if random.random() < 0.3:
        sigma = random.uniform(0.01, 0.04)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)

    # 8. Random Gaussian blur (simulate motion blur / poor focus)
    if random.random() < 0.25:
        arr_uint8 = (arr * 255).astype(np.uint8)
        ksize = random.choice([3, 5])
        arr_uint8 = cv2.GaussianBlur(arr_uint8, (ksize, ksize), 0)
        arr = arr_uint8.astype(np.float32) / 255.0

    # Normalize to [-1, 1] and transpose to CHW
    arr = (arr - 0.5) / 0.5
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def load_dataset():
    images, labels = [], []
    dirs = sorted(os.listdir(DATA_DIR))
    dirs = [d for d in dirs if os.path.isdir(os.path.join(DATA_DIR, d))]

    valid_dirs = []
    for d in dirs:
        path = os.path.join(DATA_DIR, d)
        files = [f for f in os.listdir(path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if files:
            valid_dirs.append(d)

    class_to_idx = {name: i for i, name in enumerate(valid_dirs)}

    for name in valid_dirs:
        path = os.path.join(DATA_DIR, name)
        files = [f for f in os.listdir(path)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for fname in files:
            try:
                img = Image.open(os.path.join(path, fname)).convert("RGB")
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

                # Original image
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = (arr - 0.5) / 0.5
                arr = np.transpose(arr, (2, 0, 1))
                images.append(arr)
                labels.append(class_to_idx[name])

                # Augmented copies
                for _ in range(AUGMENT_FACTOR - 1):
                    aug_arr = augment_image(img)
                    images.append(aug_arr)
                    labels.append(class_to_idx[name])
            except Exception:
                pass

    images = np.stack(images)
    labels = np.array(labels, dtype=np.int32)
    indices = np.random.permutation(len(images))
    return images[indices], labels[indices], valid_dirs


class FaceNet(nn.Cell):
    def __init__(self, num_classes, embedding_size=128):
        super().__init__()
        self.features = nn.SequentialCell([
            nn.Conv2d(3, 32, 3, pad_mode="same"), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, pad_mode="same"), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, pad_mode="same"), nn.BatchNorm2d(128),
            nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, pad_mode="same"), nn.BatchNorm2d(256),
            nn.ReLU(), nn.MaxPool2d(2, 2),
        ])
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(keep_prob=0.7)
        self.embedding = nn.Dense(256 * 7 * 7, embedding_size)
        self.l2_norm = ops.L2Normalize(axis=1)

    def construct(self, x, return_embedding=False):
        x = self.features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.embedding(x)
        if return_embedding:
            return self.l2_norm(x)
        return x


class ArcFace(nn.Cell):
    """Additive Angular Margin loss layer (simplified for small datasets)."""
    def __init__(self, num_classes, embedding_size=128, s=16.0, m=0.30):
        super().__init__()
        self.s = s
        self.cos_m = ops.cos(Tensor(m, ms.float32))
        self.sin_m = ops.sin(Tensor(m, ms.float32))
        self.weight = ms.Parameter(
            ops.StandardNormal()((num_classes, embedding_size)),
            name="arcface_weight"
        )

    def construct(self, embedding, label):
        x = ops.L2Normalize(axis=1)(embedding)
        w = ops.L2Normalize(axis=1)(self.weight)
        cos_theta = ops.matmul(x, w.T)
        cos_theta = ops.clip_by_value(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        sin_theta = ops.sqrt(1.0 - cos_theta * cos_theta + 1e-7)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        target_mask = ops.OneHot()(label, cos_theta.shape[1],
                                    Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
        logits = target_mask * cos_theta_m + (1.0 - target_mask) * cos_theta
        return logits * self.s


class TrainOneStep(nn.Cell):
    def __init__(self, network, optimizer):
        super().__init__(auto_prefix=False)
        self.network = network
        self.optimizer = optimizer
        self.weights = optimizer.parameters
        self.grad_fn = ops.value_and_grad(network, grad_position=None, weights=self.weights)

    def construct(self, data, label):
        loss, grads = self.grad_fn(data, label)
        self.optimizer(grads)
        return loss


def create_dataset(images, labels, shuffle=True):
    ds = GeneratorDataset(
        source=[(img, lbl) for img, lbl in zip(images, labels)],
        column_names=["image", "label"],
        shuffle=shuffle,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    return ds


def main():
    print("Loading dataset with augmentation...")
    images, labels, class_names = load_dataset()
    num_classes = len(class_names)
    print(f"Classes: {num_classes} ({class_names})")
    print(f"Samples (with augmentation): {len(images)}")

    split = int(len(images) * 0.85)
    train_imgs, train_lbls = images[:split], labels[:split]
    val_imgs, val_lbls = images[split:], labels[split:]

    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    train_ds = create_dataset(train_imgs, train_lbls)
    val_ds = create_dataset(val_imgs, val_lbls, shuffle=False)

    net = FaceNet(num_classes, EMBEDDING_SIZE)
    arcface = ArcFace(num_classes, EMBEDDING_SIZE, s=24.0, m=0.30)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

    def forward_fn(data, label):
        embedding = net(data)
        logits = arcface(embedding, label)
        return loss_fn(logits, label)

    trainable = net.trainable_params() + arcface.trainable_params()
    steps_per_epoch = max(len(train_imgs) // BATCH_SIZE, 1)
    lr = ms.nn.cosine_decay_lr(min_lr=0.00005, max_lr=0.005,
                                total_step=steps_per_epoch * EPOCHS,
                                step_per_epoch=steps_per_epoch, decay_epoch=EPOCHS)
    optimizer = nn.Adam(trainable, learning_rate=lr, weight_decay=1e-4)
    train_step = TrainOneStep(forward_fn, optimizer)

    print(f"\nTraining {EPOCHS} epochs ({steps_per_epoch} steps/epoch)...")
    net.set_train()
    arcface.set_train()

    best_acc = 0
    for epoch in range(EPOCHS):
        epoch_loss = 0
        count = 0
        for d in train_ds.create_dict_iterator():
            loss = train_step(d["image"], d["label"])
            epoch_loss += loss.asnumpy()
            count += 1

        avg_loss = epoch_loss / count if count else 0

        # Validate
        if (epoch + 1) % 5 == 0 or epoch == 0:
            net.set_train(False)
            arcface.set_train(False)
            correct, total = 0, 0
            for d in val_ds.create_dict_iterator():
                embedding = net(d["image"])
                logits = arcface(embedding, d["label"])
                pred = ops.Argmax(axis=1)(logits)
                correct += (pred == d["label"]).astype(ms.int32).sum().asnumpy()
                total += len(d["label"])
            acc = correct / total * 100 if total > 0 else 0
            marker = " *" if acc > best_acc else ""
            if acc > best_acc:
                best_acc = acc
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | loss={avg_loss:.4f} | val_acc={acc:.1f}%{marker}")
            net.set_train()
            arcface.set_train()

    print(f"\nBest validation accuracy: {best_acc:.1f}%")

    # Save only FaceNet (ArcFace weights are not needed for inference)
    save_path = os.path.join(MODEL_DIR, "facenet_final.ckpt")
    ms.save_checkpoint(net, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
