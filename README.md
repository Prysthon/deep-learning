# Deep Learning

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

rede neural no tensorflow para a aula de machine learning

## Execução do classificador de imagens

Na raiz do projeto, com Python 3.12 e `uv` instalados:

```sh
uv sync --locked
uv run python -m deep_learning.main
```

O código em `deep_learning/` corresponde ao notebook
`notebooks/03-image-classifier.ipynb`: carrega a CIFAR-10, normaliza os dados,
mostra exemplos, cria e treina a CNN, exibe as curvas de treinamento, avalia o
modelo e mostra as predições e a matriz de confusão.

O treinamento usa os mesmos parâmetros do notebook: 8 épocas, batch de 64 e
20% dos dados de treinamento para validação. Os gráficos são exibidos em sequência;
feche cada janela para prosseguir quando a execução aguardar. O fluxo não salva
modelos nem figuras em arquivos. O notebook original foi preservado.

## Organização do projeto

```text
├── Makefile                     # Comandos de ambiente, limpeza e Ruff
├── README.md
├── pyproject.toml               # Metadados, dependências e configuração do Ruff
├── uv.lock                      # Versões das dependências
├── docs/                        # Documentação com MkDocs
├── models/                      # Reservada para modelos
├── notebooks/                   # Notebooks das atividades
│   ├── 01-atividade_tensorflow.ipynb
│   ├── 02-structured_data_classification_from_scratch.ipynb
│   └── 03-image-classifier.ipynb
├── references/                  # Reservada para materiais de referência
├── reports/
│   └── figures/                 # Figuras do projeto
└── deep_learning/
    ├── __init__.py              # Inicialização do truststore
    ├── config.py                # Nomes das classes da CIFAR-10
    ├── dataset.py               # Carregamento e normalização dos dados
    ├── main.py                  # Execução das etapas na ordem do notebook
    ├── modeling/
    │   ├── __init__.py
    │   ├── train.py             # Construção, compilação e treinamento da CNN
    │   └── predict.py           # Avaliação e predições
    └── plots.py                 # Exemplos, curvas, predições e matriz de confusão
```
