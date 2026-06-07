# Resultados

Esta pasta contém os artefatos gerados pelo treino (`YOLO_LIBRAS.ipynb`, no Colab).
A evidência do teste em tempo real (snapshots da webcam detectando ao vivo) fica em
`real_world_test/images/`.

## Arquivos presentes

| Arquivo | Descrição |
|---|---|
| `results.png` | Curvas de loss, precisão, recall e mAP por época |
| `results.csv` | Métricas por época em formato tabular (50 épocas) |
| `confusion_matrix.png` | Matriz de confusão na validação (contagens absolutas) |
| `confusion_matrix_normalized.png` | Matriz de confusão normalizada (validação) |
| `test_confusion_matrix.png` | Matriz de confusão no conjunto de teste (absoluta) |
| `test_confusion_matrix_normalized.png` | Matriz de confusão no conjunto de teste (normalizada) |
| `train_batch0.jpg` | Batch de treino com data augmentation aplicada |
| `labels.jpg` | Distribuição de classes e densidade de bboxes |
| `metricas_por_classe.csv` | Tabela de métricas separadas por classe (35 classes) |
| `class_distribution.png` | Gráfico de barras do balanceamento de classes |
| `test_samples.png` | Amostras do conjunto de teste com bboxes ground-truth |
| `predicoes_reais.png` | Predições nas imagens reais e do conjunto de teste |

> Os snapshots do teste em tempo real (webcam) ficam em `real_world_test/images/`.

> O modelo `best.pt` **não** fica nesta pasta — ele vai para `models/best.pt`
> (caminho que o notebook da webcam procura).

> Observação: as curvas PR/F1/P/R por classe (`PR_curve.png`, `F1_curve.png`, etc.)
> não foram incluídas neste conjunto de artefatos. Se forem necessárias para o
> relatório, podem ser regeradas rodando a validação do modelo novamente
> (`model.val(..., plots=True)`).
