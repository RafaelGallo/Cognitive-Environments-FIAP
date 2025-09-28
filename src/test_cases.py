import os
import json
import pandas as pd
from pathlib import Path
from google.cloud import vision
from google.oauth2 import service_account

# Importa funções do utils.py
from utils import extract_text, compare_faces, extract_face_and_save

# Configuração de credenciais
SERVICE_ACCOUNT_FILE = (
    r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Cognitive Environments"
    r"\trabalho_final\cred\dts-10-ds-32748754226a.json"
)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE
creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE
)
client = vision.ImageAnnotatorClient(credentials=creds)
print("Credenciais carregadas com sucesso.")

# Definição de caminhos
BASE_DIR = Path(__file__).resolve().parent.parent
doc_path = BASE_DIR / "data/002.JPG"   # CNH
comp_path = BASE_DIR / "data/003.jpg"  # Comprovante de Endereço
selfies = {
    "LUIZ": BASE_DIR / "data/LUIZ.png",
    "MARIA": BASE_DIR / "data/Maria.png",
}
out_dir = BASE_DIR / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# OCR (CNH e Comprovante - comum para ambos)
print("Extraindo OCR da CNH...")
doc_text = extract_text(client, str(doc_path))

print("Extraindo OCR do Comprovante...")
comp_text = extract_text(client, str(comp_path))

print("Extraindo rosto da CNH...")
face_from_doc = extract_face_and_save(
    client, str(doc_path), str(out_dir / "face_doc.jpg")
)

# Loop de testes (LUIZ e MARIA)
for nome, selfie_path in selfies.items():
    print(f"\n=== Rodando teste para {nome} ===")
    print(f"Selfie: {selfie_path}")

    match, score = compare_faces(str(selfie_path), str(face_from_doc))

    if match:
        print(f"{nome} → Face compatível! Similaridade: {score:.3f}")
    else:
        print(f"{nome} → Face não compatível! Similaridade: {score:.3f}")

    # Resultado estruturado
    resultado = {
        "documento_nome": "LUIZ ANTONIO DE OLIVEIRA",
        "documento_cpf": "076.763.758-51",
        "comprovante_nome": "LUIZ ANTONIO DE OLIVEIRA",
        "comprovante_endereco": "R JOSE BASILIO GAMA 65, JACAREI/SP",
        "face_match": bool(match),
        "similaridade": float(round(score, 3)),
        "documento_extraido": str(doc_text),
        "comprovante_extraido": str(comp_text),
        "nome_valido": bool("LUIZ ANTONIO DE OLIVEIRA" in comp_text),
    }

    # Salvar JSON
    json_file = out_dir / f"results_{nome}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)
    print(f"JSON salvo em {json_file.resolve()}")

    # Salvar CSV
    csv_file = out_dir / f"results_{nome}.csv"
    pd.DataFrame([resultado]).to_csv(
        csv_file, index=False, encoding="utf-8-sig"
    )
    print(f"CSV salvo em {csv_file.resolve()}")

print("\nTestes concluídos com sucesso.")
