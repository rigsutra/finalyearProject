import os, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------ CONFIG --------------
DATA_DIR = r"E:\college final year project\the_app\data\train"  # <-- Change this to your dataset folder
NUM_CLASSES = 5
IMG_SIZE = 380  # native for B4
BATCH_SIZE = 8
LR = 3e-4
EPOCHS = 20
PATIENCE = 5
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- reproducibility ---------
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# -------- transforms --------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomPerspective(0.2, p=0.5),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------- dataset / loaders -------
full_ds = ImageFolder(DATA_DIR, transform=train_tf)
train_idx, val_idx = train_test_split(range(len(full_ds)), test_size=0.2,
                                      stratify=full_ds.targets, random_state=SEED)
full_ds_val = ImageFolder(DATA_DIR, transform=val_tf)
train_ds, val_ds = Subset(full_ds, train_idx), Subset(full_ds_val, val_idx)

class_counts = np.bincount([full_ds.targets[i] for i in train_idx])
class_weights = np.where(class_counts == 0, 0.0, 1.0 / class_counts)
sample_weights = [class_weights[full_ds.targets[i]] for i in train_idx]
sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=torch.cuda.is_available())
classes = full_ds.classes

# ------------- model --------------
model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.to(DEVICE)

# Freeze all layers except for the classifier
for n, p in model.named_parameters():
    if 'classifier' not in n:
        p.requires_grad_(False)

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE))

@torch.no_grad()
def validate():
    model.eval()
    v_loss = correct = total = 0
    yt, yp = [], []
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        v_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        yt.extend(y.cpu().tolist())
        yp.extend(preds.cpu().tolist())
    return v_loss / len(val_loader), 100 * correct / total, yt, yp

def main():
    # Initialize the optimizer inside main function
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    best_acc = es = 0
    for epoch in range(1, EPOCHS + 1):
        if epoch == 6:
            # Unfreeze all layers after 5 epochs
            for p in model.parameters(): 
                p.requires_grad_(True)
            # Re-initialize optimizer with all parameters unfrozen
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        model.train()
        tl = tc = tt = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tl += loss.item()
            tc += (out.argmax(1) == y).sum().item()
            tt += y.size(0)

        vl, va, _, _ = validate()
        scheduler.step(vl)
        ta = 100 * tc / tt
        print(f"E{epoch:02d}|TL {tl/len(train_loader):.3f}|TA {ta:.2f}%|VL {vl:.3f}|VA {va:.2f}%")
        
        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), 'best_b4.pth')
            es = 0
        else:
            es += 1
            if es >= PATIENCE:
                print('Early stop')
                break

    model.load_state_dict(torch.load('best_b4.pth'))
    vl, va, yt, yp = validate()
    print(f"Best Val‑Acc: {va:.2f}%")
    cm = confusion_matrix(yt, yp)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix B4')
    plt.show()
    print(classification_report(yt, yp, target_names=classes))

if __name__ == '__main__':
    main()



# The code is designed to train an EfficientNet-B4 model on a dataset of images organized in folders by class.
#for colab
# 

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# ----------- Install Dependencies -------------
!pip install torch torchvision matplotlib seaborn scikit-learn

# ----------- Imports --------------------------
import os, random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ------------ CONFIG --------------------------
DATA_DIR = '/content/drive/MyDrive/train_cleaned dataser'  # ✅ Change to your dataset path
MODEL_PATH = '/content/drive/MyDrive/efficientnet_b4/best_b4.pth'  # ✅ Where to save model
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

NUM_CLASSES = 5
IMG_SIZE = 380
BATCH_SIZE = 8
LR = 3e-4
EPOCHS = 20
PATIENCE = 5
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------- Reproducibility ----------------------
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# -------- Transforms ---------------------------
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.RandomPerspective(0.2, p=0.5),
    transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------- Dataset & Loaders --------------------
full_ds = ImageFolder(DATA_DIR, transform=train_tf)
train_idx, val_idx = train_test_split(range(len(full_ds)), test_size=0.2,
                                      stratify=full_ds.targets, random_state=SEED)
full_ds_val = ImageFolder(DATA_DIR, transform=val_tf)
train_ds, val_ds = Subset(full_ds, train_idx), Subset(full_ds_val, val_idx)

class_counts = np.bincount([full_ds.targets[i] for i in train_idx])
class_weights = np.where(class_counts == 0, 0.0, 1.0 / class_counts)
sample_weights = [class_weights[full_ds.targets[i]] for i in train_idx]
sampler = WeightedRandomSampler(sample_weights, len(train_idx), replacement=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=2, pin_memory=torch.cuda.is_available())
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=torch.cuda.is_available())
classes = full_ds.classes

# ----------- Model -----------------------------
model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.to(DEVICE)

# Freeze all layers except classifier
for n, p in model.named_parameters():
    if 'classifier' not in n:
        p.requires_grad_(False)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(DEVICE))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# ----------- Validation ------------------------
@torch.no_grad()
def validate():
    model.eval()
    v_loss = correct = total = 0
    yt, yp = [], []
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        v_loss += loss.item()
        preds = out.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        yt.extend(y.cpu().tolist())
        yp.extend(preds.cpu().tolist())
    return v_loss / len(val_loader), 100 * correct / total, yt, yp

# ----------- Training --------------------------
def main():
    global optimizer, scheduler
    best_acc = es = 0
    for epoch in range(1, EPOCHS + 1):
        if epoch == 6:
            for p in model.parameters():
                p.requires_grad_(True)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

        model.train()
        tl = tc = tt = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tl += loss.item()
            tc += (out.argmax(1) == y).sum().item()
            tt += y.size(0)

        vl, va, _, _ = validate()
        scheduler.step(vl)
        ta = 100 * tc / tt
        print(f"E{epoch:02d}|TL {tl/len(train_loader):.3f}|TA {ta:.2f}%|VL {vl:.3f}|VA {va:.2f}%")

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), MODEL_PATH)
            es = 0
        else:
            es += 1
            if es >= PATIENCE:
                print('Early stop')
                break

    model.load_state_dict(torch.load(MODEL_PATH))
    vl, va, yt, yp = validate()
    print(f"Best Val‑Acc: {va:.2f}%")
    cm = confusion_matrix(yt, yp)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix B4')
    plt.show()
    print(classification_report(yt, yp, target_names=classes))

if __name__ == '__main__':
    main()
