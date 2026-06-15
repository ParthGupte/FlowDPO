import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics.flip_detector.model import TinyMNISTBinaryCNN
from metrics.flip_detector.dataloader import FlippedMNIST
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# Hyperparameters
# --------------------------------------------------

BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 10

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# --------------------------------------------------
# Datasets / Loaders
# --------------------------------------------------

train_dataset = FlippedMNIST(train=True)
test_dataset = FlippedMNIST(train=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = TinyMNISTBinaryCNN().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
)

# --------------------------------------------------
# Training Loop
# --------------------------------------------------

for epoch in range(EPOCHS):

    # ------------------
    # Train
    # ------------------

    model.train()

    train_loss = 0.0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images).squeeze(1)

        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ------------------
    # Evaluate
    # ------------------

    model.eval()

    test_loss = 0.0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images).squeeze(1)

            loss = criterion(logits, labels)

            test_loss += loss.item()

            probs = torch.sigmoid(logits)

            preds = (probs > 0.5).float()

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    test_loss /= len(test_loader)

    accuracy = accuracy_score(
        all_labels,
        all_preds,
    )

    precision = precision_score(
        all_labels,
        all_preds,
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0,
    )

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Acc: {accuracy:.4f} | "
        f"Prec: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )

# --------------------------------------------------
# Save model
# --------------------------------------------------

torch.save(
    model.state_dict(),
    "src/metrics/flip_detector/saved_checkpoint/tiny_mnist_flip_classifier.pt",
)

print(
    "Saved model to tiny_mnist_flip_classifier.pt"
)