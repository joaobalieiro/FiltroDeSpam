# FiltroDeSpam — Classificador de e-mails ham/spam com SVM a partir de Bag-of-Words

Este repositório implementa um pipeline simples e reprodutível para **classificar e-mails como ham (não spam) ou spam**. A ideia central é transformar cada e-mail em um **vetor de frequências de palavras** (vocabulário fixo) e treinar um classificador linear eficiente para lidar com milhares de exemplos.

O projeto está organizado em três etapas claras: **construção do vocabulário**, **geração da matriz de features** e **treinamento/avaliação do modelo**.

---

## Origem do dataset

- Baixe o dataset em: `https://tinyurl.com/y93s2kcm`
- Copie todos os e-mails para uma pasta chamada **`emails`**, no mesmo nível dos scripts deste repositório.
- O dataset foi formado a partir de e-mails disponibilizados em: `http://www2.aueb.gr/users/ion/data/enron-spam/` (Enron Spam).
- O conjunto contém **aproximadamente 10 mil e-mails**.

---

## Estrutura de entrada esperada

O código assume uma única pasta:

- `emails/` contendo os arquivos `.txt`

### Rótulos (labels)

O rótulo é inferido pelo **sufixo do nome do arquivo**:

- `*.ham.txt`  → `output = 1`
- `*.spam.txt` → `output = -1`

---

## Arquivos e artefatos no repositório

- **`extrairPalavras.py`**  
  Gera um vocabulário a partir do corpus de e-mails. Faz tokenização por regex (palavras com 3+ letras), remove stopwords e aplica lematização com WordNet. O resultado é um CSV com contagens agregadas por palavra.

- **`listaPalavras.csv`** *(gerado)*  
  Vocabulário gerado por `extrairPalavras.py`, com as colunas `palavra` e `contador`.

- **`frequenciaPalavras.py`**  
  Transforma cada e-mail em um vetor de contagens com base no vocabulário (`listaPalavras.csv`) e gera a matriz de features. Ignora arquivos `._*` e processa apenas `.txt`.

- **`frequencia.csv`** *(gerado)*  
  Matriz final de entrada do modelo. Colunas = palavras do vocabulário; última coluna = `output` (1 para ham, -1 para spam).

- **`svm.py`**  
  Treinamento e avaliação usando scikit-learn com modelos otimizados:
  - `linearsvc` (baseline recomendado)
  - `sgd` com loss hinge (alternativa leve e rápida)

  Salva resultados e matriz de confusão em `resultados.txt`.

- **`resultados.txt`** *(gerado)*  
  Saída do treino contendo matriz de confusão, precision, recall e tempo de execução.

---
