from ultralytics import YOLO
import cv2

# Caminho para o SEU modelo treinado
MODEL_PATH = 'runs/detect/train/weights/best.pt' # Verifique se este caminho está correto

# Escolha uma das suas 5 imagens originais para testar
IMAGE_TO_TEST = 'dataset/images/val/f88b9964-01._Saneamento_agua_-_Ponta_Grossa_-_2025.jpg' # MUDE PARA O NOME DA SUA IMAGEM

# Carrega o seu cérebro treinado
model = YOLO(MODEL_PATH)

# Faz a predição
results = model(IMAGE_TO_TEST)

# Desenha os resultados na imagem
img = cv2.imread(IMAGE_TO_TEST)
for box in results[0].boxes:
    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
    conf = float(box.conf[0])
    
    # Desenha o retângulo
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Coloca o texto com a confiança
    cv2.putText(img, f'Telhado {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Salva a imagem com os resultados
output_path = "resultado_teste.jpg"
cv2.imwrite(output_path, img)

