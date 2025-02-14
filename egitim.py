import torch
from yolov5 import train  # YOLOv5'in train fonksiyonunu içe aktar

# Eğitim parametrelerini belirleyin
train.run(
    data='/home/oguzhan/goruntu/data.yml',  # data.yaml dosyanızın tam yolu
    weights='yolov5l.pt',  # Önceden eğitilmiş model (YOLOv5 Large)
    img_size=640,  # Görüntü boyutu
    batch_size=4,  # Batch size
    epochs=50,  # Epoch sayısı
    device='cuda' if torch.cuda.is_available() else 'cpu'  # GPU kullanılabiliyorsa kullan
)
