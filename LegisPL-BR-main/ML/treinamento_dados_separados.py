import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import shap
import numpy as np

# Carregar o dataset de treino
df_treino = pd.read_excel("final.xlsx")
# Carregar o dataset de teste
df_teste = pd.read_excel("df_teste.xlsx")

def rf(name):
    # Codificar colunas categóricas
    colunas_categoricas = X_train.select_dtypes(include='object').columns
    encoders = {}
    for col in colunas_categoricas:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        encoders[col] = le

    for col in colunas_categoricas:
        le = encoders[col]
        X_test[col] = le.fit_transform(X_test[col].astype(str))
    # Treinar modelo com GridSearch
    modelo_rf = RandomForestClassifier(random_state=42, n_estimators=200,max_depth=None)
    modelo_rf.fit(X_train, y_train)
    # Previsões e avaliação
    y_pred = modelo_rf.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    # Matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=["Reprovado", "Aprovado"])
    disp.plot(cmap='Blues', text_kw={'fontsize': 22})
    plt.title("Random Forest", fontsize=22)
    plt.xlabel("Predito", fontsize=22)
    plt.ylabel("Real", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig("cm - rf "+name+".png")
    plt.close()

    # Criar o objeto explainer para o modelo XGBoost
    explainer = shap.Explainer(modelo_rf, X_train.astype(float))

    print(type(explainer))
    # Calcular os valores SHAP para o conjunto de teste
    shap_values = explainer(X_test.astype(float))
    shap_values_classe_1 = shap_values[:, :, 1]
    # Gerar o gráfico beeswarm
    shap.plots.beeswarm(shap_values_classe_1, max_display=12, show=False)

    # Exibir o gráfico
    plt.tight_layout()
    plt.savefig("beeswarm-"+name+".png")
    plt.show()

lista = [
    '012021',
    '022021',
    '032021',
    '042021',
    '052021',
    '062021',
    '072021',
    '082021',
    '092021',
    '102021',
    '112021',
    '122021',
    '012022',
    '022022',
    '032022'
]

lista = np.array(lista, dtype=np.int64)
# Filtre o DataFrame para obter os dados que estão na lista
df_in_list = df_treino[df_treino['data_status'].isin(lista)]
X_train = df_in_list.drop(columns=["aprovado", "data_status"])
y_train = df_in_list["aprovado"].astype(int)
print(len(df_in_list))
# Filtre o DataFrame para obter os dados que estão na lista
df_in_list = df_teste[df_teste['data_status'].isin(lista)]
X_test = df_in_list.drop(columns=["aprovado", "data_status"])
y_test = df_in_list["aprovado"].astype(int)
print(len(df_in_list))
rf("antes")

df_not_in_list = df_treino[~df_treino['data_status'].isin(lista)]
X_train = df_not_in_list.drop(columns=["aprovado", "data_status"])
y_train = df_not_in_list["aprovado"].astype(int)

df_not_in_list = df_teste[~df_teste['data_status'].isin(lista)]
X_test = df_not_in_list.drop(columns=["aprovado", "data_status"])
y_test = df_not_in_list["aprovado"].astype(int)

rf("depois")



