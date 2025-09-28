import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from google.cloud import vision
from google.oauth2 import service_account
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Importa funções utilitárias
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR / "src"))
from utils import extract_text, compare_faces, extract_face_and_save

# Configuração de credenciais
SERVICE_ACCOUNT_FILE = BASE_DIR / "cred" / "dts-10-ds-32748754226a.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(SERVICE_ACCOUNT_FILE)
creds = service_account.Credentials.from_service_account_file(
    str(SERVICE_ACCOUNT_FILE)
)
client = vision.ImageAnnotatorClient(credentials=creds)
print("Credenciais carregadas com sucesso.")

# Caminhos
doc_path = BASE_DIR / "data/002.JPG"
comp_path = BASE_DIR / "data/003.jpg"
selfies = {
    "LUIZ": {"path": BASE_DIR / "data/LUIZ.png", "label": 1},   # esperado válido
    "MARIA": {"path": BASE_DIR / "data/Maria.png", "label": 0}, # esperado inválido
}
out_dir = BASE_DIR / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# OCR (CNH e Comprovante)
print("Extraindo OCR da CNH...")
doc_text = extract_text(client, str(doc_path))

print("Extraindo OCR do comprovante...")
comp_text = extract_text(client, str(comp_path))

print("Extraindo rosto da CNH...")
face_from_doc = extract_face_and_save(
    client, str(doc_path), str(out_dir / "face_doc.jpg")
)

# Loop de testes
y_true, y_pred = [], []
THRESHOLD = 0.90  # limite mínimo de similaridade

for nome, info in selfies.items():
    selfie_path = info["path"]
    label = info["label"]

    print(f"\n=== Rodando teste para {nome} ===")
    match, score = compare_faces(str(selfie_path), str(face_from_doc))

    # aplica threshold manual
    face_valid = score >= THRESHOLD

    if face_valid:
        print(f"{nome} → Face compatível. Similaridade: {score:.3f}")
    else:
        print(f"{nome} → Face não compatível. Similaridade: {score:.3f}")

    # salvar resultado
    resultado = {
        "nome": nome,
        "label_real": label,
        "predicao": int(face_valid),
        "similaridade": round(score, 3),
        "documento_extraido": doc_text,
        "comprovante_extraido": comp_text,
    }

    json_file = out_dir / f"results_{nome}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)

    y_true.append(label)
    y_pred.append(int(face_valid))

# Métricas
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nAvaliação do pipeline:")
print(f"Acurácia  : {acc:.2f}")
print(f"Precisão  : {prec:.2f}")
print(f"Recall    : {rec:.2f}")
print(f"F1-Score  : {f1:.2f}")

metrics_file = out_dir / "metrics_summary.csv"
df_metrics = pd.DataFrame([{
    "acuracia": acc,
    "precisao": prec,
    "recall": rec,
    "f1_score": f1,
}])
df_metrics.to_csv(metrics_file, index=False, encoding="utf-8-sig")
print(f"Métricas salvas em {metrics_file.resolve()}")

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
labels = ["Inválido", "Válido"]

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)
plt.xlabel("Predição")
plt.ylabel("Real")
plt.title("Matriz de Confusão")

cm_file = out_dir / "confusion_matrix.png"
plt.savefig(cm_file)
print(f"Matriz de confusão salva em {cm_file.resolve()}")
