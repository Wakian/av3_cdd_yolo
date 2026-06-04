# Resultados

Esta pasta recebe os artefatos gerados pelo notebook `YOLO_LIBRAS.ipynb` rodado no Colab.

## O que deve estar aqui ao final

Depois de treinar no Colab, baixe o zip `artefatos_libras.zip` gerado pela última célula do notebook e descompacte-o aqui. Conteúdo esperado:

| Arquivo | Descrição |
|---|---|
| `best.pt` | Modelo YOLO com melhor desempenho na validação |
| `results.png` | Curvas de loss, precisão, recall e mAP por época |
| `results.csv` | Métricas por época em formato tabular |
| `confusion_matrix.png` | Matriz de confusão (contagens absolutas) |
| `confusion_matrix_normalized.png` | Matriz de confusão normalizada |
| `PR_curve.png` | Curva Precisão × Revocação |
| `F1_curve.png` | Curva F1 × Confiança |
| `P_curve.png` | Curva Precisão × Confiança |
| `R_curve.png` | Curva Revocação × Confiança |
| `test_*.png` | Mesmas curvas/matrizes calculadas no conjunto de teste |
| `train_batch0.jpg` | Batch de treino com data augmentation aplicada |
| `labels.jpg` | Distribuição de classes e densidade de bboxes |
| `metricas_por_classe.csv` | Tabela de métricas separadas por letra |
| `class_distribution.png` | Gráfico de barras do balanceamento de classes |
| `test_samples.png` | Amostras do conjunto de teste com bboxes ground-truth |
| `predicoes_reais.png` | Predições nas fotos capturadas pela equipe |

> ⚠️ `best.pt` pode ser grande (~50 MB) — está no `.gitignore` por padrão. Se quiser versionar, use Git LFS ou hospede no Drive/Releases.
