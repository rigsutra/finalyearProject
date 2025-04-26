# ---------------------------------------------
#  EfficientNet‑B0 Transfer Learning Script
# ---------------------------------------------
# Save as efficientnet_b0_train.py
# Requirements: torch>=2.0, torchvision>=0.17, scikit‑learn, seaborn, matplotlib

import os, random, time, argparse
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------ CONFIG --------------
DATA_DIR   = "/content/drive/My Drive/train"   # <-- change me
NUM_CLASSES = 5
IMG_SIZE    = 224   # native for B0
BATCH_SIZE  = 32
LR          = 1e-4
EPOCHS      = 20
PATIENCE    = 5
SEED        = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Reproducibility --------
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ------------- Transforms ------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.RandomPerspective(0.2, p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ------------- Dataset ---------------
full_ds = ImageFolder(DATA_DIR, transform=train_tf)
train_idx, val_idx = train_test_split(range(len(full_ds)), test_size=0.2,
                                     stratify=full_ds.targets, random_state=SEED)
full_ds_val = ImageFolder(DATA_DIR, transform=val_tf)
train_ds, val_ds = Subset(full_ds, train_idx), Subset(full_ds_val, val_idx)

# Weighted sampler (class‑balancing)
class_counts   = np.bincount([full_ds.targets[i] for i in train_idx])
class_weights  = np.where(class_counts==0, 0.0, 1.0/class_counts)
sample_weights = [class_weights[full_ds.targets[i]] for i in train_idx]
sampler        = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=torch.cuda.is_available())
classes = full_ds.classes

# ------------- Model -----------------
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.to(DEVICE)

for n,p in model.named_parameters():
    if "classifier" not in n:
        p.requires_grad_(False)

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ---------- Helper: validation pass ----------
@torch.no_grad()
def validate():
    model.eval(); v_loss=correct=total=0; y_true=[]; y_pred=[]
    for x,y in val_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out,y)
        v_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds==y).sum().item(); total += y.size(0)
        y_true.extend(y.cpu().tolist()); y_pred.extend(preds.cpu().tolist())
    return v_loss/len(val_loader), 100*correct/total, y_true, y_pred

# ------------- Training loop -----------------
best_acc = es_counter = 0
for epoch in range(1, EPOCHS+1):
    if epoch==6:                              # unfreeze backbone
        for p in model.parameters(): p.requires_grad_(True)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=3)

    model.train(); tloss=tcorrect=ttotal=0
    for x,y in train_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(); out=model(x); loss=criterion(out,y)
        loss.backward(); optimizer.step()
        tloss += loss.item(); tcorrect += (out.argmax(1)==y).sum().item(); ttotal += y.size(0)

    vloss, vacc, *_ = validate(); scheduler.step(vloss)
    tacc = 100*tcorrect/ttotal; print(f"E{epoch:02d} | TL {tloss/len(train_loader):.3f} | TA {tacc:.2f}% | VL {vloss:.3f} | VA {vacc:.2f}%")

    if vacc>best_acc:
        best_acc = vacc; torch.save(model.state_dict(), "best_b0.pth"); es_counter=0
    else:
        es_counter += 1; 
        if es_counter>=PATIENCE:
            print("Early stopped"); break

# ------------- Evaluation ---------------
model.load_state_dict(torch.load("best_b0.pth"))
vloss, vacc, y_true, y_pred = validate()
print(f"Best Val‑Acc: {vacc:.2f}%")
cm = confusion_matrix(y_true,y_pred)
plt.figure(figsize=(6,5)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=classes,yticklabels=classes)
plt.xlabel('Pred'); plt.ylabel('True'); plt.title('Confusion Matrix B0'); plt.show()
print(classification_report(y_true,y_pred,target_names=classes))



# """
# disease_train.py

# """

# # ─────────────────────────── 0. Imports ────────────────────────────
# import os, itertools, numpy as np, matplotlib.pyplot as plt, seaborn as sns
# import torch, torch.nn as nn
# from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# torch.backends.cudnn.benchmark = True   # auto‑tune kernels

# # ─────────────────────────── 1. Paths ───────────────────────────────
# DATA_DIR   = r"D:\datasets\leaf_disease\train"   # <-- change to your folder
# OUTPUT_DIR = "./checkpoints"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # ─────────────────────────── 2. Hyper‑params ────────────────────────
# NUM_CLASSES   = 5
# IMG_SIZE      = 128          # 224 for higher accuracy if VRAM allows
# BATCH_SIZE    = 32
# EPOCHS        = 20
# PATIENCE      = 5            # early‑stopping patience
# BASE_LR       = 1e-3         # 0.001
# UNFREEZE_EPOC = 5            # unfreeze backbone after this epoch

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ─────────────────────────── 3. Transforms ──────────────────────────
# train_tf = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(0.2,0.2,0.2,0.1),
#     transforms.RandomPerspective(0.2, p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# val_tf = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])

# # ─────────────────────────── 4. Dataset & loaders ───────────────────
# full_ds   = ImageFolder(DATA_DIR, transform=train_tf)
# val_clone = ImageFolder(DATA_DIR, transform=val_tf)  # identical targets

# train_idx, val_idx = train_test_split(
#     range(len(full_ds)),
#     test_size=0.2,
#     stratify=full_ds.targets,
#     random_state=42
# )

# train_ds = Subset(full_ds, train_idx)
# val_ds   = Subset(val_clone, val_idx)
# classes  = full_ds.classes

# # Weighted sampler for balanced mini‑batches
# cls_cnt        = np.bincount([full_ds.targets[i] for i in train_idx])
# cls_weights    = 1. / cls_cnt
# sample_weights = [cls_weights[full_ds.targets[i]] for i in train_idx]
# sampler        = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
#                           num_workers=2, pin_memory=True)
# val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
#                           num_workers=2, pin_memory=True)

# # ─────────────────────────── 5. Model ───────────────────────────────
# model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
# model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
# model.to(DEVICE)

# # Freeze backbone
# for n, p in model.named_parameters():
#     if "classifier" not in n:
#         p.requires_grad_(False)

# # Loss, optimiser, scheduler
# criterion  = nn.CrossEntropyLoss(weight=torch.tensor(cls_weights, dtype=torch.float).to(DEVICE))
# optimizer  = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
#                               lr=BASE_LR)
# scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
#                                                         patience=3, factor=0.5)

# # ─────────────────────────── 6. Helper funcs ────────────────────────
# def validate(net, loader):
#     net.eval()
#     v_loss, correct, total = 0, 0, 0
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for x, y in loader:
#             x, y = x.to(DEVICE), y.to(DEVICE)
#             out  = net(x)
#             v_loss += criterion(out, y).item()
#             preds  = out.argmax(1)
#             correct += (preds == y).sum().item()
#             total   += y.size(0)
#             y_true.extend(y.cpu().tolist())
#             y_pred.extend(preds.cpu().tolist())
#     return v_loss/len(loader), 100*correct/total, y_true, y_pred

# # ─────────────────────────── 7. Training loop ───────────────────────
# history, best_acc, es_cnt = {"tl":[], "ta":[], "vl":[], "va":[]}, 0, 0

# for epoch in range(1, EPOCHS+1):
#     model.train()
#     tloss, tcorrect, ttotal = 0, 0, 0

#     for imgs, lbls in train_loader:
#         imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
#         optimizer.zero_grad()
#         out   = model(imgs)
#         loss  = criterion(out, lbls)
#         loss.backward()
#         optimizer.step()
#         tloss += loss.item()
#         preds  = out.argmax(1)
#         tcorrect += (preds == lbls).sum().item()
#         ttotal   += lbls.size(0)

#     vl, va, y_true, y_pred = validate(model, val_loader)
#     scheduler.step(vl)

#     tl, ta = tloss/len(train_loader), 100*tcorrect/ttotal
#     history["tl"].append(tl); history["ta"].append(ta)
#     history["vl"].append(vl); history["va"].append(va)

#     print(f"E{epoch:02d} | TL {tl:.3f} | TA {ta:.2f}% | VL {vl:.3f} | VA {va:.2f}% | LR {optimizer.param_groups[0]['lr']:.1e}")

#     # early‑stopping & checkpoint
#     if va > best_acc:
#         best_acc = va
#         torch.save(model.state_dict(), os.path.join(OUTPUT_DIR,"best_model.pth"))
#         es_cnt = 0
#     else:
#         es_cnt += 1
#         if es_cnt >= PATIENCE:
#             print("Early stopping triggered.")
#             break

#     # unfreeze backbone after warm‑up
#     if epoch == UNFREEZE_EPOC:
#         for p in model.parameters(): p.requires_grad_(True)
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

# # ─────────────────────────── 8. Evaluation ──────────────────────────
# model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR,"best_model.pth")))
# vl, va, y_true, y_pred = validate(model, val_loader)
# print(f"\nBest validation accuracy: {va:.2f}%")

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6,5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
#             xticklabels=classes, yticklabels=classes)
# plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
# plt.tight_layout(); plt.show()

# print(classification_report(y_true, y_pred, target_names=classes))

# # Curves
# epochs = range(1, len(history["tl"])+1)
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(epochs, history["tl"], label="Train")
# plt.plot(epochs, history["vl"], label="Val")
# plt.title("Loss"); plt.xlabel("Epoch"); plt.legend()
# plt.subplot(1,2,2)
# plt.plot(epochs, history["ta"], label="Train")
# plt.plot(epochs, history["va"], label="Val")
# plt.title("Accuracy"); plt.xlabel("Epoch"); plt.legend()
# plt.tight_layout(); plt.show()






## combined code of B4 , b3 and b0
# -----------------------------------------------------------
#  EfficientNet Transfer‑Learning Boilerplate (B0 / B3 / B4)
# -----------------------------------------------------------
# """
# A single, configurable training script that lets you switch between
# EfficientNet‑B0, B3, and B4 without copy‑pasting three separate files.

# Usage (Colab/CLI):
# >>> MODEL_VARIANT = "b3"   # choices: "b0", "b3", "b4"
# >>> !python efficientnet_variants.py   # or run all cells in Colab

# Key improvements vs. the original snippet:
# • Deterministic seeds & cuDNN flags for reproducibility.
# • Guard against zero‑division in class weights.
# • Unfreeze backbone at the **start** of epoch 6.
# • Size‑aware transforms (224, 300, 380).
# • Mixed‑precision friendly (wrap train step if desired).
# • Leaner optimizer reset logic (retain momentum for classifier).
# """

# import os, random, time, argparse, itertools
# import numpy as np
# import torch, torch.nn as nn
# from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torchvision.models import (
#     efficientnet_b0, EfficientNet_B0_Weights,
#     efficientnet_b3, EfficientNet_B3_Weights,
#     efficientnet_b4, EfficientNet_B4_Weights,
# )
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, classification_report
# import matplotlib.pyplot as plt
# import seaborn as sns

# # -------------------------------
# # 1. Configuration
# # -------------------------------
# MODEL_VARIANT = os.getenv("MODEL_VARIANT", "b0").lower()  # "b0", "b3", or "b4"
# DATA_DIR      = "/content/drive/My Drive/train"            # adjust as needed
# NUM_CLASSES   = 5
# BATCH_SIZE    = {"b0": 32, "b3": 16, "b4": 8}[MODEL_VARIANT]
# EPOCHS        = 20
# PATIENCE      = 5
# LR_BASE       = {"b0": 1e-4, "b3": 2e-4, "b4": 3e-4}[MODEL_VARIANT]
# IMG_SIZE      = {"b0": 224, "b3": 300, "b4": 380}[MODEL_VARIANT]
# DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# SEED          = 42

# # -------------------------------
# # 2. Reproducibility
# # -------------------------------
# random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

# # -------------------------------
# # 3. Transforms
# # -------------------------------
# train_tf = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
#     transforms.RandomPerspective(0.2, p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

# val_tf = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

# # -------------------------------
# # 4. Dataset & loaders
# # -------------------------------
# full_ds = ImageFolder(DATA_DIR, transform=train_tf)
# train_idx, val_idx = train_test_split(
#     range(len(full_ds)), test_size=0.2, stratify=full_ds.targets, random_state=SEED
# )

# # swap transforms for val subset only
# full_ds_val = ImageFolder(DATA_DIR, transform=val_tf)
# train_ds = Subset(full_ds, train_idx)
# val_ds   = Subset(full_ds_val, val_idx)

# # Weighted sampler to balance batches
# class_counts   = np.bincount([full_ds.targets[i] for i in train_idx])
# class_weights  = np.where(class_counts == 0, 0.0, 1.0 / class_counts)
# sample_weights = [class_weights[full_ds.targets[i]] for i in train_idx]
# sampler        = WeightedRandomSampler(sample_weights, num_samples=len(train_idx), replacement=True)

# train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, pin_memory=torch.cuda.is_available())
# val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,  num_workers=2, pin_memory=torch.cuda.is_available())
# classes = full_ds.classes

# # -------------------------------
# # 5. Model factory
# # -------------------------------
# model_dispatch = {
#     "b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
#     "b3": (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
#     "b4": (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
# }
# make_model, weights_enum = model_dispatch[MODEL_VARIANT]
# model = make_model(weights=weights_enum)
# model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
# model.to(DEVICE)

# # Freeze backbone initially
# for name, param in model.named_parameters():
#     if "classifier" not in name:
#         param.requires_grad_(False)

# criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE))
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_BASE)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

# # -------------------------------
# # 6. Helper: validation pass
# # -------------------------------
# @torch.no_grad()
# def validate():
#     model.eval()
#     v_loss, correct, total = 0, 0, 0
#     y_true, y_pred = [], []
#     for imgs, labels in val_loader:
#         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#         out  = model(imgs)
#         loss = criterion(out, labels)
#         v_loss += loss.item()
#         preds   = out.argmax(dim=1)
#         correct += (preds == labels).sum().item()
#         total   += labels.size(0)
#         y_true.extend(labels.cpu().tolist())
#         y_pred.extend(preds.cpu().tolist())
#     return v_loss / len(val_loader), 100 * correct / total, y_true, y_pred

# # -------------------------------
# # 7. Training loop
# # -------------------------------
# best_acc, es_counter = 0, 0
# history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc"]}

# for epoch in range(1, EPOCHS + 1):
#     # Un‑freeze backbone at the *start* of epoch 6
#     if epoch == 6:
#         for param in model.parameters():
#             param.requires_grad_(True)
#         # re‑build optimizer but retain momentum on existing params
#         optimizer = torch.optim.Adam(model.parameters(), lr=LR_BASE)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3)

#     # -------- Train --------
#     model.train()
#     t_loss, t_correct, t_total = 0, 0, 0
#     for imgs, labels in train_loader:
#         imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
#         optimizer.zero_grad()
#         out  = model(imgs)
#         loss = criterion(out, labels)
#         loss.backward()
#         optimizer.step()

#         t_loss    += loss.item()
#         preds      = out.argmax(dim=1)
#         t_correct += (preds == labels).sum().item()
#         t_total   += labels.size(0)

#     # -------- Validate --------
#     v_loss, v_acc, y_true, y_pred = validate()
#     scheduler.step(v_loss)

#     # -------- Metrics --------
#     t_loss /= len(train_loader); t_acc = 100 * t_correct / t_total
#     history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
#     history['train_acc'].append(t_acc);   history['val_acc'].append(v_acc)

#     print(f"E{epoch:02d} | TL {t_loss:.3f} | TA {t_acc:.2f}% | VL {v_loss:.3f} | VA {v_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.1e}")

#     # -------- Checkpoint / Early‑stop --------
#     if v_acc > best_acc:
#         best_acc = v_acc
#         torch.save(model.state_dict(), f"best_{MODEL_VARIANT}.pth")
#         es_counter = 0
#     else:
#         es_counter += 1
#         if es_counter >= PATIENCE:
#             print("\n[Early‑Stop] validation accuracy plateaued.")
#             break

# # -------------------------------
# # 8. Final evaluation + plots
# # -------------------------------
# model.load_state_dict(torch.load(f"best_{MODEL_VARIANT}.pth"))
# v_loss, v_acc, y_true, y_pred = validate()
# print(f"\nBest Val‑Acc ({MODEL_VARIANT.upper()}): {v_acc:.2f}%\n")

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
# plt.xlabel('Predicted'); plt.ylabel('True'); plt.title(f'Confusion Matrix – {MODEL_VARIANT.upper()}'); plt.show()

# print(classification_report(y_true, y_pred, target_names=classes))

# # Loss & accuracy curves
# epochs_ran = range(1, len(history['train_loss']) + 1)
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_ran, history['train_loss'], label='Train'); plt.plot(epochs_ran, history['val_loss'], label='Val')
# plt.title('Loss'); plt.legend()
# plt.subplot(1, 2, 2)
# plt.plot(epochs_ran, history['train_acc'], label='Train'); plt.plot(epochs_ran, history['val_acc'], label='Val')
# plt.title('Accuracy'); plt.legend(); plt.show()
