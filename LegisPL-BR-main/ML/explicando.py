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

# Carregar o dataset
df = pd.read_excel("final.xlsx")
#df = df.rename(columns={'tema_dominante2': 'tema_dominante'})
# Preparar dados
X = df.drop(columns=["aprovado", "data_status"])
y = df["aprovado"].astype(int)

# Carregar o dataset_test
df = pd.read_excel("df_teste.xlsx")
#df = df.rename(columns={'tema_dominante2': 'tema_dominante'})
# Preparar dados
X_test = df.drop(columns=["aprovado", "data_status"])
y_test = df["aprovado"].astype(int)

# Codificar colunas categóricas
colunas_categoricas = X.select_dtypes(include='object').columns
encoders = {}
for col in colunas_categoricas:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

for col in colunas_categoricas:
    le = encoders[col]
    X_test[col] = le.fit_transform(X_test[col].astype(str))

# Ver o mapeamento da coluna 'temas_dominantes'
encoder_tema = encoders['tema_dominante']

# Mostrar os pares (número → categoria)
for i, classe in enumerate(encoder_tema.classes_):
    print(f"{i}: {classe}")

# Ver o mapeamento da coluna 'temas_dominantes'
encoder_regiao = encoders['região']

# Mostrar os pares (número → categoria)
for i, classe in enumerate(encoder_regiao.classes_):
    print(f"{i}: {classe}")

# Ver o mapeamento da coluna 'temas_dominantes'
encoder_bloco = encoders['bloco_partidario']

# Mostrar os pares (número → categoria)
for i, classe in enumerate(encoder_bloco.classes_):
    print(f"{i}: {classe}")


def rf():
    # Treinar modelo com GridSearch
    modelo_rf = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None)
    modelo_rf.fit(X, y)
    y_pred = modelo_rf.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Criar o objeto explainer para o modelo XGBoost
    explainer = shap.Explainer(modelo_rf, X.astype(float))

    # Calcular os valores SHAP para o conjunto de teste
    shap_values = explainer(X.astype(float))
    # Gerar e exibir o gráfico waterfall para o elemento 668
    print(f"Gerando gráfico Waterfall para o elemento 668...")
    shap_values_classe_1 = shap_values[:, :, 1]

    for i in [202]:
        shap.plots.waterfall(shap_values_classe_1[i], show=False)
        plt.tight_layout()
        plt.savefig("wf4545.png")
        plt.show()


    # Gerar o gráfico beeswarm
    shap.plots.beeswarm(shap_values_classe_1, max_display=12, show=False)

    # Exibir o gráfico
    plt.tight_layout()
    plt.savefig("beeswarm.png")
    plt.show()



rf()

