# 1. 라이브러리 불러오기
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import cv2

# 2. 데이터셋 준비
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dir = '../Monkey/training/training'
val_dir = '../Monkey/validation/validation'

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes

# 3. EfficientNet_b0 모델 준비
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_classes = len(train_dataset.classes)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 4. 학습 + Best 모델 저장
train_losses = []
val_accuracies = []
best_acc = 0.0
save_path = 'best_efficientnet.pth'

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f'Epoch [{epoch+1}/10], Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        print(f'📦 Best 모델 저장됨! (Accuracy: {best_acc:.2f}%)')

# 5. 학습 결과 시각화
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, marker='o')
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1,2,2)
plt.plot(val_accuracies, marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()

# -------------------------------------------------------
# [Best 모델 불러오기 후 평가]
# -------------------------------------------------------

# 6. Best 모델 불러오기
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.load_state_dict(torch.load('best_efficientnet.pth'))
model = model.to(device)
model.eval()

# 7. 전체 Validation 데이터 예측
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 8. Confusion Matrix 시각화
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax, cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# 9. F1 Score, Precision, Recall 출력
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

# -------------------------------------------------------
# [Softmax 확률 시각화 + Grad-CAM 분석]
# -------------------------------------------------------

# 10. 1배치 예측 및 Softmax 확률
dataiter = iter(val_loader)
images, labels = next(dataiter)

images = images.to(device)
labels = labels.to(device)

outputs = model(images)
probs = F.softmax(outputs, dim=1)
_, preds = torch.max(outputs, 1)

images = images.cpu()
labels = labels.cpu()
preds = preds.cpu()
probs = probs.cpu()

# 11. 4개 샘플 Softmax 예측 시각화
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

plt.figure(figsize=(12,10))
for idx in range(4):
    ax = plt.subplot(2, 2, idx+1)
    imshow(images[idx])
    pred_label = class_names[preds[idx]]
    true_label = class_names[labels[idx]]
    confidence = probs[idx][preds[idx]].item() * 100
    ax.set_title(f"True: {true_label}\nPred: {pred_label} ({confidence:.2f}%)")
    ax.axis('off')

plt.tight_layout()
plt.show()

# -------------------------------------------------------
# [Grad-CAM 분석 (1개 이미지)]
# -------------------------------------------------------

# 12. Grad-CAM 준비 (Hook 걸기)
feature_maps = None
gradients = None

def forward_hook(module, input, output):
    global feature_maps
    feature_maps = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# 모델 Hook 연결
target_layer = model.features[-1]
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# 13. 하나의 이미지에 대해 Grad-CAM
img = images[0].unsqueeze(0).to(device)

# Forward
output = model(img)
pred_class = output.argmax(dim=1)

# Backward
model.zero_grad()
output[0, pred_class].backward()

# Grad-CAM 만들기
weights = gradients.mean(dim=(2, 3), keepdim=True)
grad_cam = (weights * feature_maps).sum(dim=1, keepdim=True)
grad_cam = torch.relu(grad_cam)

# Resize
grad_cam = torch.nn.functional.interpolate(grad_cam, size=(224, 224), mode='bilinear', align_corners=False)
grad_cam = grad_cam.squeeze().cpu().detach().numpy()

# Normalize
grad_cam -= grad_cam.min()
grad_cam /= grad_cam.max()

# 14. Grad-CAM 시각화
original_img = images[0].numpy().transpose((1, 2, 0))
heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
combined = 0.4 * heatmap + 0.6 * original_img

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(original_img)
plt.title('Original')

plt.subplot(1,3,2)
plt.imshow(grad_cam, cmap='jet')
plt.title('Grad-CAM')

plt.subplot(1,3,3)
plt.imshow(combined)
plt.title('Overlay')

plt.tight_layout()
plt.show()
