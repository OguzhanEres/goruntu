import sys
import os
import time
import torch
import cv2
import numpy as np
from pathlib import Path

# YOLOv5 klasörünü Python modül yoluna ekleyin
sys.path.append(os.path.abspath("/home/oguzhan/goruntu/yolov5"))

# YOLOv5 modüllerini içe aktar
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors

# Model ve cihaz ayarları
model_path = "/home/oguzhan/goruntu/yolov5.pt"  # Eğitilmiş YOLOv5 modelinizin yolu
video_path = "/home/oguzhan/goruntu/a.mp4"  # Video dosyası veya 0 (webcam)
device = select_device("cuda" if torch.cuda.is_available() else "cpu")  # GPU varsa kullan

# Modeli yükleme
model = DetectMultiBackend(model_path, device=device, dnn=False)
model.model.float()
stride, names, pt = model.stride, model.names, model.pt
img_size = 640  # Modelin eğitimde kullanıldığı boyut

# Video kaynağını aç
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Video kaynağına erişilemiyor {video_path}")
    exit()

prev_time = time.time()  # FPS hesaplamak için önceki zamanı sakla

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Video bittiğinde çık

    start_time = time.time()  # FPS hesaplamak için zaman ölçümünü başlat

    # Görseli ön işleme
    img_resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_rgb = torch.from_numpy(img_rgb).to(device)
    img_rgb = img_rgb.permute(2, 0, 1).float() / 255.0  # Normalize et
    img_rgb = img_rgb.unsqueeze(0)  # Batch boyutu ekle

    # Model ile tahmin yap
    with torch.no_grad():
        pred = model(img_rgb)

    # Sonuçları filtreleme (NMS - Non-Maximum Suppression)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Görsel üzerinde çizim yapmak için Annotator kullanımı
    annotator = Annotator(frame, line_width=2, example=names)

    for det in pred:  # Tüm tahminler için döngü
        if det is not None and len(det):  # Eğer tespit edilen nesne varsa
            det[:, :4] = scale_boxes(img_resized.shape[:2], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:  # Tensor içindeki her nesne için
                label = f"{names[int(cls)]} {conf:.2f}"
                annotator.box_label(xyxy, label, color=colors(int(cls), True))
        else:
            print("Warning: No objects detected!")

    # FPS hesaplama
    end_time = time.time()
    fps = 1 / (end_time - start_time)  # FPS hesapla

    # FPS değerini görüntüye yazdır
    cv2.putText(annotator.im, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # FPS değerini terminalde göster
    print(f"FPS: {fps:.2f}")

    # Pencereyi daha küçük hale getirerek görüntüyü göster
    display_resized = cv2.resize(annotator.im, (600, 700))  # 800x600 boyutuna getir
    cv2.imshow("YOLOv5 Video Detection", display_resized)


    # ESC tuşuna basılırsa çık
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
