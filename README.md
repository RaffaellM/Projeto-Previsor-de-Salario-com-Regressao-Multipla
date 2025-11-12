🚀 Projeto: Previsor de Salário com Regressão Múltipla

Este é um script em Python que constrói e treina um modelo de machine learning para prever o salário de um funcionário. Diferente de uma regressão simples, este modelo utiliza **Regressão Linear Múltipla**, o que significa que ele baseia sua previsão em **múltiplas variáveis** de entrada.

As variáveis de entrada (features) usadas são:
* `YearsExperience` (Anos de Experiência)
* `Score` (Score de Performance)

A variável de saída (target) que o modelo prevê é:
* `Salary` (Salário)

---

### O que o Código Faz

O script `train_model_multi.py` executa as seguintes etapas:

1.  **Carregamento dos Dados:** Utiliza a biblioteca `pandas` para ler os dados do arquivo `data_multi.csv` e carregá-los em um DataFrame.
2.  **Preparação dos Dados:**
    * Separa o DataFrame em `X` (as colunas de features: `YearsExperience` e `Score`) e `y` (a coluna alvo: `Salary`).
3.  **Divisão para Treino e Teste:**
    * Utiliza a função `train_test_split` do `scikit-learn` para dividir os dados em dois conjuntos: um conjunto de **treinamento** (80% dos dados) e um conjunto de **teste** (20% dos dados).
    * Isso é crucial para avaliar se o modelo consegue generalizar para dados que nunca viu antes.
4.  **Criação e Treinamento do Modelo:**
    * Cria uma instância do modelo `LinearRegression`.
    * Treina o modelo (encontra os melhores coeficientes $m_1$, $m_2$ e o intercepto $b$) usando o método `.fit()` nos dados de treinamento (`X_train` e `y_train`).
5.  **Avaliação do Modelo:**
    * Imprime os **coeficientes** (`model.coef_`) e o **intercepto** (`model.intercept_`) que o modelo aprendeu. Isso mostra o "peso" que o modelo deu para cada feature (Anos de ExperiLencia e Score).
    * Faz previsões nos dados de teste (`X_test`).
    * Calcula e imprime o **R² (R-squared)**, uma métrica que indica o quão bem o modelo se ajusta aos dados (um valor mais próximo de 1.0 é melhor).
6.  **Previsão Interativa:**
    * Inicia um loop `while True` que permite ao usuário fazer previsões em tempo real.
    * Pede ao usuário para digitar os **Anos de Experiência** e o **Score de Performance**.
    * Utiliza o modelo treinado (`model.predict()`) para calcular o salário previsto com base nessas duas entradas.
    * Imprime o resultado formatado e continua pedindo novas entradas até que o usuário digite 'sair'.

---

### 🛠️ Ferramentas e Bibliotecas Utilizadas

* **Python:** A linguagem de programação base.
* **Pandas:** Usada para carregar e manipular os dados do arquivo `.csv`.
* **Scikit-learn (sklearn):** A principal biblioteca de machine learning utilizada para:
    * `model_selection.train_test_split`: Para dividir os dados em conjuntos de treino e teste.
    * `linear_model.LinearRegression`: A classe que implementa o algoritmo de Regressão Linear.
    * `metrics.r2_score`: Para calcular o score de performance R² do modelo.
