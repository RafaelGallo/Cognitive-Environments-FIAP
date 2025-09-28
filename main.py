import os
import json
import pandas as pd
from pathlib import Path
from google.cloud import vision
from google.oauth2 import service_account

# importa funções utilitárias
from utils import extract_text, compare_faces, extract_face_and_save

# configuração de credenciais
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

# configuração global
THRESHOLD = 0.7  # limite mínimo de similaridade aceito

# definição de caminhos
doc_path = Path(
    r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Cognitive Environments"
    r"\trabalho_final\data\006.jpeg"
)
comp_path = Path(
    r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Cognitive Environments"
    r"\trabalho_final\data\003.jpg"
)
selfie_path = Path(
    r"G:\Meu Drive\AI_data_lab\Cursos_ml_AI\Fiap\Cognitive Environments"
    r"\trabalho_final\data\007.png"
)
out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# OCR
print("Extraindo OCR da CNH...")
doc_text = extract_text(client, str(doc_path))

print("Extraindo OCR do Comprovante...")
comp_text = extract_text(client, str(comp_path))

# extração de rosto
print("Extraindo rosto da CNH...")
face_from_doc = extract_face_and_save(
    client, str(doc_path), str(out_dir / "face_doc.jpg")
)

# comparação facial
print("Comparando selfie com CNH...")
match, similarity = compare_faces(
    str(selfie_path), str(face_from_doc), threshold=THRESHOLD
)

if match:
    print(
        f"Face compatível! Similaridade: {similarity:.3f} "
        f"(mínimo aceito = {THRESHOLD})"
    )
else:
    print(
        f"Face não compatível! Similaridade: {similarity:.3f} "
        f"(mínimo aceito = {THRESHOLD})"
    )

# consolidação de resultados
resultado = {
    "face_match": bool(match),
    "similaridade": float(round(similarity, 3)),
    "threshold_utilizado": THRESHOLD,
    "documento_extraido": str(doc_text),
    "comprovante_extraido": str(comp_text),
    "nome_valido": bool("LUIZ ANTONIO DE OLIVEIRA" in comp_text),
}

# salvar JSON
json_file = out_dir / "results.json"
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(resultado, f, ensure_ascii=False, indent=4)
print(f"JSON salvo em {json_file.resolve()}")

# salvar CSV
df = pd.DataFrame([resultado])
csv_file = out_dir / "results.csv"
df.to_csv(csv_file, index=False, encoding="utf-8-sig")
print(f"CSV salvo em {csv_file.resolve()}")

print("Pipeline concluído com sucesso!")
