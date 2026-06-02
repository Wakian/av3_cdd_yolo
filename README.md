# Detecção de LIBRAS com YOLO

Projeto final da disciplina de **Ciência de Dados** — treinamento e avaliação de um modelo YOLO para detecção de gestos do **alfabeto manual da Língua Brasileira de Sinais (LIBRAS)**.

> **Classe inédita (fora do COCO):** as 80 classes padrão do YOLO/COCO não incluem gestos manuais de LIBRAS, atendendo ao requisito do projeto.

---

## 👥 Integrantes

- _Nome 1_
- _Nome 2_
- _Nome 3_

## 🎯 Objetivo

Treinar um modelo YOLO para detectar e classificar as letras do alfabeto em LIBRAS em imagens reais, com foco em uma possível aplicação em ferramentas de acessibilidade e tradução automática.

## 📦 Dataset

- **Fonte:** Roboflow Universe (link e versão definidos no notebook)
- **Formato:** YOLO (bounding boxes + classes)
- **Classes:** 26 letras do alfabeto manual em LIBRAS (A–Z, com adaptações dos sinais dinâmicos H, J, K, X, Y, Z conforme o dataset escolhido)
- **Divisão:** train / val / test

> ⚠️ Datasets de "sign language" frequentemente são ASL (americano). Confirme se o dataset escolhido é LIBRAS ou documente as diferenças no relatório (ex: H, J, K, X, Y, Z mudam entre ASL e LIBRAS).

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

1. Abra `YOLO_LIBRAS.ipynb` no **Google Colab** com runtime de **GPU (T4 ou superior)**.
2. Configure a chave de API do Roboflow na célula correspondente (siga as instruções no notebook).
3. Execute as células sequencialmente:
   - 📦 Instalação de dependências
   - 📥 Download e exploração do dataset
   - 🏋️ Treinamento do YOLO
   - 📊 Avaliação (mAP, precisão, recall, matriz de confusão)
   - 🔍 Inferência em imagens reais capturadas pela equipe
4. Faça download dos artefatos gerados (`best.pt`, gráficos, predições) e coloque na pasta `results/` deste repositório.
5. Adicione as fotos capturadas em `real_world_test/images/` antes de rodar a célula de inferência real.

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
- [Ultralytics YOLO](https://docs.ultralytics.com/) (v8 ou superior)
- PyTorch + CUDA
- Roboflow (download do dataset)
- Google Colab (treinamento em nuvem)
