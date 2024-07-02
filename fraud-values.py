import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import time

# Carregando os dados
data = pd.read_csv('creditcard.csv')

# Separando as features e o target
X = data.drop('Class', axis=1)
y = data['Class']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalizando os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # Calcula-se valores de média e desvio padrão
X_test = scaler.transform(X_test) # Aplica ao conjunto de teste os valores calculados

# Lidando com o desbalanceamento de classes
smote = SMOTE(random_state=100)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Inicializando os modelos
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss'), # Logloss penaliza previsões incorretas
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Treinando e avaliando os modelos
for model_name, model in models.items():
    print(f"Training {model_name}...")
    # Start timer
    start_time = time.time()
    model.fit(X_train_res, y_train_res)
    # End timer
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time to train model {model_name}: ", elapsed_time)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"Results for {model_name}:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_prob)}")
    print("\n" + "=================================" + "\n")
