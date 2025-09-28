import os
import sys
import json
import pandas as pd
from pathlib import Path
from google.cloud import vision
from google.oauth2 import service_account
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Importa utils.py
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

# Definição de caminhos
doc_path = BASE_DIR / "data/002.JPG"  # CNH
comp_path = BASE_DIR / "data/003.jpg"  # Comprovante
selfies = {
    "LUIZ": {"path": BASE_DIR / "data/LUIZ.png", "label": 1},   # esperado válido
    "MARIA": {"path": BASE_DIR / "data/Maria.png", "label": 0}, # esperado inválido
}
out_dir = BASE_DIR / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# OCR (CNH e comprovante)
print("Extraindo OCR da CNH...")
doc_text = extract_text(client, str(doc_path))

print("Extraindo OCR do comprovante...")
comp_text = extract_text(client, str(comp_path))

print("Extraindo rosto da CNH...")
face_from_doc = extract_face_and_save(
    client, str(doc_path), str(out_dir / "face_doc.jpg")
)

# Avaliação
y_true, y_pred = [], []

for nome, info in selfies.items():
    selfie_path = info["path"]
    label = info["label"]

    print(f"\n=== Rodando teste para {nome} ===")
    print(f"Selfie: {selfie_path}")

    match, score = compare_faces(str(selfie_path), str(face_from_doc))

    if match:
        print(f"{nome} → Face compatível. Similaridade: {score:.3f}")
    else:
        print(f"{nome} → Face não compatível. Similaridade: {score:.3f}")

    # Estrutura do resultado
    resultado = {
        "documento_nome": "LUIZ ANTONIO DE OLIVEIRA",
        "documento_cpf": "076.763.758-51",
        "comprovante_nome": "LUIZ ANTONIO DE OLIVEIRA",
        "comprovante_endereco": "R JOSE BASILIO GAMA 65, JACAREI/SP",
        "face_match": match,
        "similaridade": round(score, 3),
        "documento_extraido": doc_text,
        "comprovante_extraido": comp_text,
        "nome_valido": "LUIZ ANTONIO DE OLIVEIRA" in comp_text,
    }

    # Salvar JSON/CSV
    json_file = out_dir / f"results_{nome}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)
    print(f"JSON salvo em {json_file.resolve()}")

    csv_file = out_dir / f"results_{nome}.csv"
    pd.DataFrame([resultado]).to_csv(
        csv_file, index=False, encoding="utf-8-sig"
    )
    print(f"CSV salvo em {csv_file.resolve()}")

    # Atualiza métricas
    y_true.append(label)
    y_pred.append(1 if match else 0)

# Métricas de avaliação
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\nAvaliação do pipeline:")
print(f"Acurácia  : {acc:.2f}")
print(f"Precisão  : {prec:.2f}")
print(f"Recall    : {rec:.2f}")
print(f"F1-Score  : {f1:.2f}")

# Salvar resumo de métricas
metrics_file = out_dir / "metrics_summary.csv"
df_metrics = pd.DataFrame([{
    "acuracia": acc,
    "precisao": prec,
    "recall": rec,
    "f1_score": f1
}])
df_metrics.to_csv(metrics_file, index=False, encoding="utf-8-sig")
print(f"Métricas salvas em {metrics_file.resolve()}")
