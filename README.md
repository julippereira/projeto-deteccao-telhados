# Projeto de Detecção de Telhados com YOLOv8

Projeto de visão computacional para detecção automática de telhados em imagens aéreas/de satélite, utilizando o modelo YOLOv8.

## Visão Geral

O modelo é treinado para identificar e delimitar telhados em imagens, gerando bounding boxes com o nível de confiança de cada detecção.

## Estrutura do Projeto

```
projeto-deteccao-telhados/
│
├── data.yaml           # Configuração do dataset (caminhos e classes)
├── treinar.py          # Script de treinamento do modelo
├── testar_modelo.py    # Script de inferência/teste
├── yolov8n.pt          # Pesos pré-treinados do YOLOv8 Nano (base)
├── requirements.txt    # Dependências do projeto
│
└── dataset/
    └── images/
        ├── train/      # Imagens de treinamento
        └── val/        # Imagens de validação
```

Após o treinamento, os pesos do modelo treinado ficam disponíveis em:

```
runs/detect/train/weights/best.pt
```

## Requisitos

- Python 3.8+
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV

### Instalação das dependências

```bash
pip install -r requirements.txt
```

> As principais dependências são:
> - `ultralytics==8.4.16` — YOLOv8
> - `opencv-python==4.13.0.92` — Processamento de imagens
> - `torch==2.10.0` / `torchvision==0.25.0` — Framework de deep learning

## Dataset

O dataset é configurado no arquivo [data.yaml](data.yaml):

| Parâmetro | Valor |
|-----------|-------|
| Classes   | 1 (`telhado`) |
| Imagens de treino | `dataset/images/train/` |
| Imagens de validação | `dataset/images/val/` |

As anotações devem estar no formato YOLO (`.txt`), com os bounding boxes normalizados.

## Como Usar

### 1. Treinamento

Execute o script de treinamento:

```bash
python treinar.py
```

Parâmetros configurados no treinamento:

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `epochs`  | 100   | Número de épocas |
| `imgsz`   | 640   | Tamanho das imagens |
| `patience`| 10    | Early stopping após 10 épocas sem melhora |

Os resultados e os pesos do modelo treinado serão salvos em `runs/detect/train/`.

### 2. Teste / Inferência

Edite o arquivo [testar_modelo.py](testar_modelo.py) para definir a imagem a ser testada:

```python
IMAGE_TO_TEST = 'dataset/images/val/sua_imagem.jpg'
```

Em seguida, execute:

```bash
python testar_modelo.py
```

O resultado será salvo como `resultado_teste.jpg` na raiz do projeto, com os telhados detectados marcados por bounding boxes verdes e o respectivo score de confiança.

## Modelo Base

O projeto utiliza o **YOLOv8 Nano** (`yolov8n.pt`) como modelo base para fine-tuning. É o modelo mais leve da família YOLOv8, ideal para datasets pequenos e treinamento rápido.

## Referências

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [OpenCV Docs](https://docs.opencv.org/)
