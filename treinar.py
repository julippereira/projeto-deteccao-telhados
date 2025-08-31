from ultralytics import YOLO

# Carrega um modelo YOLOv8 pré-treinado (yolov8n.pt é o menor e mais rápido)
model = YOLO('yolov8n.pt')

# Inicia o treinamento com os seus dados
if __name__ == '__main__':
    # O bloco 'if __name__ == '__main__':' é importante para evitar problemas no macOS e Windows
    results = model.train(
        data='data.yaml',    # O nosso "mapa" de dados
        epochs=100,          # Número de vezes que o modelo verá os dados. Para poucos dados, 100 é um bom começo.
        imgsz=640,           # Tamanho das imagens
        patience=10          # Para o treino mais cedo se não houver melhora após 10 épocas
    )
    print("Treinamento concluído com sucesso!")
