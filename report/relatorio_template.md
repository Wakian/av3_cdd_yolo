# Relatório Técnico — Detecção de LIBRAS com YOLO

> Este é um esqueleto. Use-o como base para gerar o PDF final via Word/Google Docs/LaTeX e fazer upload no AVA.

---

## 1. Identificação

- **Título:** Detecção do alfabeto manual de LIBRAS com YOLO
- **Disciplina:** Ciência de Dados — Projeto 3 (Deep Learning)
- **Integrantes:**
  - _Nome 1 — matrícula_
  - _Nome 2 — matrícula_
  - _Nome 3 — matrícula_
- **Repositório GitHub:** _<INSERIR LINK PÚBLICO DO REPOSITÓRIO — OBRIGATÓRIO no topo do relatório>_
- **Data:** _dd/mm/aaaa_

---

## 2. Introdução e Motivação

A **Língua Brasileira de Sinais (LIBRAS)** é a língua oficial da comunidade surda no Brasil, reconhecida pela Lei nº 10.436/2002. Apesar de seu reconhecimento legal, ainda há uma barreira significativa de comunicação entre surdos e ouvintes, e o desenvolvimento de ferramentas automáticas de tradução visual contribui diretamente para a acessibilidade.

Este projeto propõe utilizar o modelo **YOLO (You Only Look Once)** para a detecção do alfabeto manual em LIBRAS — um conjunto de 26 configurações de mão correspondentes às letras A–Z. Diferente do ASL (American Sign Language), o alfabeto LIBRAS possui particularidades em letras dinâmicas (H, J, K, X, Y, Z), o que justifica a necessidade de um modelo treinado especificamente para o contexto brasileiro.

### Justificativa da classe inédita

As 26 classes do alfabeto manual LIBRAS **não estão presentes** nas 80 categorias do COCO. A categoria mais próxima no COCO é "person", o que não cobre nem a região anatômica (mão) nem a semântica (configuração de letra). Portanto, a tarefa atende plenamente ao requisito de classe inédita do projeto.

---

## 3. Metodologia

### 3.1 Dataset

| Atributo | Valor |
|---|---|
| Fonte | _Roboflow Universe — link do dataset_ |
| Total de imagens | _preencher_ |
| Treino | _preencher_ |
| Validação | _preencher_ |
| Teste | _preencher_ |
| Número de classes | _preencher (provavelmente 26)_ |
| Formato | YOLO (bounding box + classe) |
| Pré-processamento | _Auto-orientação, redimensionamento, etc._ |
| Data augmentation | _Aplicada no Roboflow / aplicada pelo YOLO durante o treino_ |

#### Distribuição de classes

> Inserir gráfico de barras `class_distribution.png` aqui.

Comente eventuais desbalanceamentos: alguma letra com muito mais ou menos exemplos que as demais? Como isso pode afetar o resultado?

### 3.2 Ferramentas e ambiente

- **Linguagem:** Python 3.12
- **Biblioteca principal:** Ultralytics YOLO v8.x
- **Framework:** PyTorch com CUDA
- **Ambiente de treino:** Google Colab — GPU Tesla T4 (15 GB VRAM)
- **Versionamento:** GitHub
- **Gerenciamento de dataset:** Roboflow

### 3.3 Configuração de treinamento

| Hiperparâmetro | Valor | Justificativa |
|---|---|---|
| Arquitetura | YOLOv8m | Equilíbrio entre acurácia e tempo de treino na T4 |
| Pré-treinado em | COCO | Transfer learning padrão |
| Épocas | 50 | Suficiente para convergência com early stopping |
| Batch size | 16 | Limite da memória da T4 para imgsz=640 |
| Image size | 640 × 640 | Padrão YOLO, bom equilíbrio detalhe/velocidade |
| Otimizador | SGD (default) | Padrão Ultralytics |
| Patience (early stop) | 15 | Para evitar overfitting tardio |
| Data augmentation | HSV, flip, mosaic (Ultralytics defaults) | _comentar se ajustou_ |

---

## 4. Análise de Resultados

### 4.1 Métricas globais no conjunto de teste

| Métrica | Valor |
|---|---|
| Precisão média | _preencher_ |
| Revocação média | _preencher_ |
| mAP@0.5 | _preencher_ |
| mAP@0.5:0.95 | _preencher_ |

**Interpretação:** _Escreva 2–3 parágrafos analisando os números. Por exemplo: "O mAP@0.5 de X indica que o modelo identifica corretamente a maioria das letras quando a sobreposição é tolerante; já o mAP@0.5:0.95 de Y mostra que a precisão da bounding box ainda pode melhorar..."_

### 4.2 Curvas de treinamento

> Inserir `results.png` do treino aqui.

Comente:
- O loss diminuiu de forma estável ou houve oscilações?
- O mAP de validação acompanhou o de treino ou começou a divergir (sinal de overfitting)?
- Em que época o early stopping foi acionado (se foi)?

### 4.3 Matriz de confusão

> Inserir `confusion_matrix_normalized.png` aqui.

**Análise crítica:** quais letras são mais confundidas entre si? Em LIBRAS, espera-se confusão entre:
- M e N (semelhança de número de dedos dobrados)
- U e V (mesma configuração, abertura diferente)
- A e S (mão fechada similar)
- R e U (cruzamento de dedos)
- Letras dinâmicas (H, J, K, X, Y, Z) capturadas em frame estático

Comente o que de fato apareceu nos seus dados.

### 4.4 Métricas por classe

> Inserir tabela do `metricas_por_classe.csv` aqui.

Quais letras tiveram melhor desempenho? Quais foram as mais difíceis? Há correlação com:
- Número de exemplos no treino?
- Similaridade visual com outras letras?
- Movimento envolvido na letra (sinais dinâmicos)?

### 4.5 Curvas PR e F1

> Inserir `PR_curve.png` e `F1_curve.png`.

Qual é o threshold de confiança ótimo para maximizar F1? Comente.

---

## 5. Inferência Prática

### 5.1 Aplicação em imagens reais (capturadas pela equipe)

> Inserir as imagens originais e as imagens com as predições anotadas.

Para cada imagem, comente:
- O modelo detectou corretamente?
- A confiança foi alta?
- Houve falsos positivos? Detecção parcial?
- Condições de captura (iluminação, fundo, ângulo) impactaram o resultado?

### 5.2 Aplicação no conjunto de teste

> Inserir 2–3 exemplos do conjunto de teste com predições.

Comente se o desempenho em dados "controlados" foi consistente com a métrica geral.

### 5.3 Análise crítica de generalização

Discuta: o modelo generaliza bem das imagens do dataset (provavelmente capturadas em condições homogêneas) para as fotos reais da equipe (condições variadas)?

Possíveis causas de queda de desempenho fora do domínio:
- Iluminação diferente
- Fundo cluttered (vs fundo limpo do dataset)
- Tom de pele diferente do conjunto de treino
- Tamanho da mão na imagem (escala)
- Ângulo da câmera

---

## 6. Conclusão

### 6.1 Síntese dos resultados

Resuma em 1–2 parágrafos o que foi feito e o que foi obtido.

### 6.2 Desafios enfrentados

Exemplos do tipo de coisa que vale comentar:
- Encontrar um dataset LIBRAS realmente brasileiro (vs ASL)
- Tempo de treino no Colab gratuito (limite de GPU)
- Sinais dinâmicos vs frame estático
- Desbalanceamento de classes
- Generalização para fotos com iluminação real

### 6.3 Soluções adotadas

Para cada desafio, comente como foi resolvido (ou contornado).

### 6.4 Oportunidades de melhoria

- Modelo maior (YOLOv8l/x ou YOLOv11)
- Mais épocas + scheduler de learning rate customizado
- Data augmentation customizado (rotação, oclusão parcial da mão)
- Coletar dataset próprio focado em LIBRAS brasileiro real
- Migrar para detecção em vídeo + tracking temporal para sinais dinâmicos
- Combinar com landmarks da mão (MediaPipe Hands) para features adicionais

---

## 7. Referências

- Ultralytics YOLO Documentation. https://docs.ultralytics.com/
- Roboflow Universe. https://universe.roboflow.com/
- Lei nº 10.436/2002 — Reconhecimento da LIBRAS. http://www.planalto.gov.br/ccivil_03/leis/2002/l10436.htm
- COCO Dataset Classes. https://cocodataset.org/
- _Outras referências usadas (artigos, tutoriais)._
