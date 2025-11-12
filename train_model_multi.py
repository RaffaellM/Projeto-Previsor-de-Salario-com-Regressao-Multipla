import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- 1. Carregar e Preparar os Dados ---
data = pd.read_csv('data_multi.csv')

X = data[['YearsExperience', 'Score']] 
y = data['Salary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Criar e Treinar o Modelo ---

model = LinearRegression()
model.fit(X_train, y_train)

# --- 3. Avaliar o Modelo ---

print(f"Coeficientes (m1, m2): {model.coef_}")
print(f"Intercepto (b): {model.intercept_}")


y_pred = model.predict(X_test)

print(f"R² (Score de performance): {r2_score(y_test, y_pred)}")

# --- 4. Fazer uma Nova Previsão ---
print("\n--- Fazendo uma nova previsão ---")

anos_exp = 8
score = 95
dados_para_prever = [[anos_exp, score]]
salario_previsto = model.predict(dados_para_prever)
print(f"Para {anos_exp} anos e score {score}, o salário previsto é: R$ {salario_previsto[0]:.2f}")

# --- 5. Previsão Interativa (Bônus) ---
print("\n--- Previsor Interativo (Múltiplo) ---")

while True:
    try:
        anos_str = input("\nDigite os anos de experiência (ou 'sair'): ")
        if anos_str.lower() == 'sair':
            break
        
        score_str = input("Digite o Score de Performance (0-100): ")
        if score_str.lower() == 'sair':
            break

        anos_exp = float(anos_str)
        score = float(score_str)

        dados_para_prever = [[anos_exp, score]]
        salario_previsto = model.predict(dados_para_prever)
        print(f"-> Para {anos_exp} anos e score {score}, o salário previsto é: R$ {salario_previsto[0]:.2f}")

    except ValueError:
        print("Entrada inválida. Por favor, digite números.")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")

print("Até logo!")