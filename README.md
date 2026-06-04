# Detecção de LIBRAS com YOLO

Projeto final da disciplina de **Ciência de Dados** — treinamento de um modelo YOLO para detecção de gestos do **alfabeto manual da Língua Brasileira de Sinais (LIBRAS)**.

> **Classe inédita (fora do COCO):** as 80 classes padrão do YOLO/COCO não incluem gestos manuais de LIBRAS, atendendo ao requisito do projeto.

---

## 👥 Integrantes

- _Nome 1_
- _Nome 2_
- _Nome 3_

## 🎯 Objetivo

Treinar um modelo YOLO para detectar e classificar as letras do alfabeto em LIBRAS em imagens reais, com foco em uma possível aplicação em ferramentas de acessibilidade e tradução automática.

## 📦 Dataset

- **Fonte:** [Projeto LIBRAS — Roboflow Universe](https://universe.roboflow.com/gomes-project/projeto-libras)
- **Formato:** YOLO Detection (imagens + labels com bounding boxes)
- **Classes:** alfabeto manual em LIBRAS (verificar lista exata no `data.yaml` baixado)
- **Divisão:** train / valid / test (já vem dividido pelo Roboflow)

### Estrutura após download

```
<DATASET_PATH>/
├── train/
│   ├── images/   (*.jpg)
│   └── labels/   (*.txt — formato YOLO)
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml      (gerado pelo Roboflow)
```

## 🏗️ Estrutura do repositório

```
YOLO_new_class/
├── README.md                  # Este arquivo
├── YOLO_LIBRAS.ipynb          # Notebook all-in-one (rodar no Colab)
├── real_world_test/
│   └── images/                # Fotos reais capturadas pela equipe
├── results/                   # Métricas, matriz de confusão, predições
├── report/
│   └── relatorio_template.md  # Esqueleto do relatório técnico
└── .gitignore
```

## ▶️ Como executar

1. **Crie uma conta no Roboflow** (https://roboflow.com) e pegue sua API key em https://app.roboflow.com/settings/api.
2. Abra `YOLO_LIBRAS.ipynb` no **Google Colab** com runtime de **GPU (T4 ou superior)**.
3. Cole sua API key na célula de download do dataset (variável `ROBOFLOW_API_KEY`).
4. Confirme a versão atual do dataset em https://universe.roboflow.com/gomes-project/projeto-libras/dataset e ajuste `VERSION_NUMBER` se necessário.
5. Execute as células sequencialmente:
   - 📦 Instalação de dependências
   - 📥 Download do dataset via Roboflow API
   - 🔍 Exploração (distribuição + visualização de bboxes)
   - 🏋️ Treinamento do YOLOv8m
   - 📊 Avaliação (mAP, precisão, recall, matriz de confusão)
   - 🤳 Inferência em imagens reais capturadas pela equipe
6. Faça download dos artefatos (`best.pt`, gráficos) e descompacte em `results/`.
7. Adicione fotos capturadas em `real_world_test/images/` antes da célula de inferência.

## 📊 Resultados

> Preencher após o treinamento.

| Métrica | Valor |
|---|---|
| mAP@0.5 | _preencher_ |
| mAP@0.5:0.95 | _preencher_ |
| Precisão | _preencher_ |
| Recall | _preencher_ |

### Predição em imagem real

> Substituir pelos prints das predições nas fotos capturadas pela equipe.

![Exemplo de predição](results/predicao_exemplo.png)

## 📄 Relatório técnico

O relatório completo está disponível em [`report/relatorio_tecnico.pdf`](report/relatorio_tecnico.pdf) e foi entregue via AVA.

## 🛠️ Stack

- Python 3.12
- [Ultralytics YOLO](https://docs.ultralytics.com/) (v8m detection)
- PyTorch + CUDA
- Roboflow (download do dataset)
- Google Colab (treinamento em nuvem com GPU T4)
