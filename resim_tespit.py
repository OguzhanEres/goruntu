import sys
import os
import time
import torch
import cv2
import numpy as np
from pathlib import Path

# YOLOv5 klasörünü Python modül yoluna ekleyin
sys.path.append(os.path.abspath("/home/oguzhan/goruntu/yolov5"))

# YOLOv5 modüllerini içe aktarın
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors

# Model ve cihaz ayarları
model_path = "/home/oguzhan/goruntu/yolov5l.pt"  # Eğitilmiş YOLOv5 modelinizin yolu
image_path = "/home/oguzhan/goruntu/images/train/000000001072_jpg.rf.11ce73acf69cf1b59b432c42a7b4d8ea.jpg"     # İşlenecek görüntü dosyasının yolu
device = select_device("cuda" if torch.cuda.is_available() else "cpu")  # GPU varsa kullan

# Modeli yükleme
model = DetectMultiBackend(model_path, device=device, dnn=False)
model.model.float()
stride, names, pt = model.stride, model.names, model.pt
img_size = 640  # Modelin eğitimde kullanıldığı boyut

# Görüntüyü yükle
img = cv2.imread(image_path)
if img is None:
    print(f"Hata: Görüntü dosyası bulunamadı: {image_path}")
    exit()

# Görüntüyü modelin girdi boyutuna göre yeniden boyutlandır
img_resized = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_tensor = torch.from_numpy(img_rgb).to(device)
img_tensor = img_tensor.permute(2, 0, 1).float() / 255.0  # Normalize et
img_tensor = img_tensor.unsqueeze(0)  # Batch boyutu ekle

# Tahmin süresini ölçmek için zaman ölçümünü başlat
start_time = time.time()

# Model ile tahmin yap (inference)
with torch.no_grad():
    pred = model(img_tensor)

# Sonuçları filtreleme (NMS - Non-Maximum Suppression)
pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

# Görüntü üzerinde çizim yapmak için Annotator kullanımı
annotator = Annotator(img.copy(), line_width=2, example=names)

# Tespit edilen her nesne için kutu çizimi
for det in pred:
    if det is not None and len(det):
        # Tespit edilen kutuları orijinal görüntü boyutuna ölçeklendir
        det[:, :4] = scale_boxes(img_resized.shape[:2], det[:, :4], img.shape).round()
        for *xyxy, conf, cls in det:
            # cls değerini int'e çevir
            class_id = int(cls)
            # names içinde bu id varsa etiketi, yoksa "Unknown" şeklinde etiketle
            if class_id in names:
                label_text = f"{names[class_id]} {conf:.2f}"
            else:
                label_text = f"Unknown({class_id}) {conf:.2f}"
                print(f"Uyarı: Tanımlı olmayan sınıf id'si tespit edildi: {class_id}")
            annotator.box_label(xyxy, label_text, color=colors(class_id, True))
    else:
        print("Uyarı: Hiçbir nesne tespit edilemedi!")

# İşlem süresini hesapla ve FPS değerini görüntüye ekle
end_time = time.time()
fps = 1 / (end_time - start_time)
cv2.putText(annotator.im, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2)
print(f"FPS: {fps:.2f}")

# Pencereyi daha küçük hale getirerek görüntüyü göster
display_resized = cv2.resize(annotator.im, (600, 700))  # 800x600 boyutuna getir
cv2.imshow("YOLOv5 Video Detection", display_resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
