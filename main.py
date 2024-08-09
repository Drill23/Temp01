from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Dados reais e sintéticos
dados = [
    # Dia, Velocidade, Peso médio, T_escaldagem, T_entrada, T_saída, T_depenadeira, T_mangueira, Qualidade
    (15, 86, 2.5, 56.6, 53.8, 59.0, 60.1, 60.7, 0.9),
    (16, 100, 2.6, 61.7, 57.0, 55.5, 53.4, 55.9, 0.6),
    (17, 108, 2.4, 59.9, 55.2, 55.0, 53.3, 55.2, 0.2),
    (18, 106, 2.7, 52.6, 52.8, 54.0, 52.6, 53.8, 0.9),
    (19, 82, 2.8, 54.5, 56.0, 54.6, 53.1, 53.5, 0.5),
    (22, 63, 2.3, 54.3, 54.5, 55.7, 59.0, 60.0, 0.0),
    (23, 86, 2.6, 56.3, 55.1, 54.0, 52.0, 51.8, 1.0),
    (24, 106, 2.5, 53.7, 54.7, 55.4, 57.9, 59.5, 0.7),
    (25, 105, 2.7, 52.1, 55.7, 55.5, 53.0, 54.6, 0.8),
    (26, 110, 2.4, 48.5, 54.2, 55.3, 54.3, 56.3, 0.6),
    (29, 110, 2.6, 56.6, 54.4, 56.0, 57.0, 57.2, 0.7),
    (30, 106, 2.5, 56.7, 55.1, 55.8, 55.7, 55.1, 0.6),
    (31, 106, 2.7, 57.2, 57.3, 56.5, 36.8, 57.9, 0.9),
    # Dados sintéticos
    (32, 95, 2.4, 55.0, 54.0, 55.0, 54.0, 55.0, 0.8),
    (33, 100, 2.6, 56.0, 55.0, 56.0, 55.0, 56.0, 0.7),
    (34, 105, 2.5, 57.0, 56.0, 57.0, 56.0, 57.0, 0.6),
    (35, 90, 2.7, 54.0, 53.0, 54.0, 53.0, 54.0, 0.9),
    (36, 98, 2.8, 55.5, 54.5, 55.5, 54.5, 55.5, 0.75),
]

# Preparação dos dados
X = np.array([[d[1], d[2]] for d in dados])
y = np.array([[d[3], d[4], d[5], d[6], d[7]] for d in dados])

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

# Treinamento do modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    velocidade = float(request.form['velocidade'])
    peso = float(request.form['peso'])

    input_data = np.array([[velocidade, peso]])
    input_scaled = scaler_X.transform(input_data)

    previsao_scaled = model.predict(input_scaled)
    previsao = scaler_y.inverse_transform(previsao_scaled)

    resultado = {
        'escaldagem': round(previsao[0][0], 1),
        'entrada': round(previsao[0][1], 1),
        'saida': round(previsao[0][2], 1),
        'depenadeira': round(previsao[0][3], 1),
        'mangueira': round(previsao[0][4], 1)
    }

    return jsonify(resultado)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)