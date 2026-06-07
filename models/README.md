# models/

Esta pasta guarda o **modelo treinado** (`best.pt`) que o notebook da webcam
(`YOLO_LIBRAS_webcam.ipynb`) carrega para rodar a detecção em tempo real.

## O que colocar aqui

| Arquivo | Origem | Obrigatório? |
|---|---|---|
| `best.pt` | Baixado do Colab após o treino (`runs/detect/libras_yolov8m/weights/best.pt`) | Sim, para rodar a webcam e a inferência local |

## Como obter o `best.pt`

1. No Colab, após o treino concluído, o arquivo fica em
   `runs/detect/libras_yolov8m/weights/best.pt`.
2. Faça o download dele (ou pegue de dentro do `artefatos_libras.zip` gerado
   pela última célula do notebook de treino).
3. Coloque-o aqui como `models/best.pt`.

> O caminho esperado pelo notebook da webcam é exatamente `models/best.pt`.
> Se você salvar com outro nome ou em outro lugar, ajuste a variável
> `MODEL_PATH` na célula de configuração do `YOLO_LIBRAS_webcam.ipynb`.

## Versionamento

O `best.pt` (~50 MB) está no `.gitignore` e **não vai para o GitHub** por padrão,
porque arquivos binários grandes não devem ser versionados em Git comum.

Para o professor ter acesso ao modelo, use uma destas opções:
- Hospedar no Google Drive e colocar o link no README / relatório.
- Publicar em **GitHub Releases** (aceita binários grandes).
- Usar **Git LFS** (`git lfs track "*.pt"`).
