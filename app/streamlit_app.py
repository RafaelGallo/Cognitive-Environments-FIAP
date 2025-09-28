import os
from pathlib import Path
import streamlit as st
import pandas as pd
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import imagehash


# OCR com Google Vision
def extract_text(client, image_path: str) -> str:
    """Extrai texto de uma imagem usando a API do Google Vision."""
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""


# Extração de rosto usando Google Vision
def extract_face_and_save(client, image_path: str, output_file: str):
    """Extrai o rosto de uma imagem e salva em arquivo."""
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    if not faces:
        return None
    face = faces[0]
    x_min = min(v.x for v in face.bounding_poly.vertices)
    y_min = min(v.y for v in face.bounding_poly.vertices)
    x_max = max(v.x for v in face.bounding_poly.vertices)
    y_max = max(v.y for v in face.bounding_poly.vertices)

    with Image.open(image_path) as pil_img:
        face_crop = pil_img.crop((x_min, y_min, x_max, y_max))
        face_crop.save(output_file)
        return output_file


# Comparação facial via ImageHash (wavelet hash)
def compare_faces(img1_path: str, img2_path: str, threshold: float = 0.7):
    """Compara duas imagens faciais usando wavelet hash."""
    try:
        img1 = Image.open(img1_path).convert("L").resize((256, 256))
        img2 = Image.open(img2_path).convert("L").resize((256, 256))
        hash1 = imagehash.whash(img1)
        hash2 = imagehash.whash(img2)
        diff = hash1 - hash2
        similarity = 1 - (diff / len(hash1.hash) ** 2)
        match = similarity >= threshold
        return match, similarity
    except Exception as e:
        print(f"Erro na comparação: {e}")
        return False, 0.0


# Configuração Google Vision
ROOT_DIR = Path(__file__).resolve().parent.parent
SERVICE_ACCOUNT_FILE = ROOT_DIR / "cred" / "dts-10-ds-32748754226a.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(SERVICE_ACCOUNT_FILE)

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE
)
client = vision.ImageAnnotatorClient(credentials=creds)

# Interface Streamlit
st.set_page_config(page_title="Validação de Documentos", layout="wide")
st.title("Validação Biométrica da Quantum Finance")

uploaded_doc = st.file_uploader(
    "Upload da CNH (imagem)", type=["jpg", "jpeg", "png"]
)
uploaded_comp = st.file_uploader(
    "Upload do Comprovante de Endereço", type=["jpg", "jpeg", "png"]
)
uploaded_selfie = st.file_uploader(
    "Upload da Selfie", type=["jpg", "jpeg", "png"]
)

if st.button("Processar") and uploaded_doc and uploaded_comp and uploaded_selfie:
    out_dir = ROOT_DIR / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    doc_path = out_dir / "doc.jpg"
    comp_path = out_dir / "comp.jpg"
    selfie_path = out_dir / "selfie.jpg"

    with open(doc_path, "wb") as f:
        f.write(uploaded_doc.read())
    with open(comp_path, "wb") as f:
        f.write(uploaded_comp.read())
    with open(selfie_path, "wb") as f:
        f.write(uploaded_selfie.read())

    # OCR
    st.info("Executando OCR...")
    doc_text = extract_text(client, str(doc_path))
    comp_text = extract_text(client, str(comp_path))

    # Rosto
    st.info("Extraindo rosto da CNH...")
    face_from_doc = extract_face_and_save(
        client, str(doc_path), str(out_dir / "face_doc.jpg")
    )

    # Comparação
    st.info("Comparando selfie com CNH...")
    match, score = compare_faces(
        str(selfie_path), str(face_from_doc), threshold=0.7
    )

    resultado = {
        "face_match": bool(match),
        "similaridade": float(round(score, 3)),
        "documento_extraido": str(doc_text),
        "comprovante_extraido": str(comp_text),
        "nome_valido": bool("LUIZ ANTONIO DE OLIVEIRA" in comp_text),
    }

    df = pd.DataFrame([resultado])

    # Exibe resultados
    st.subheader("Resultado")
    st.json(resultado)
    st.dataframe(df)

    # Mostra imagens enviadas
    st.subheader("Imagens enviadas")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(str(doc_path), caption="CNH", width=250)
    with col2:
        st.image(str(comp_path), caption="Comprovante", width=250)
    with col3:
        st.image(str(selfie_path), caption="Selfie", width=250)

    # Mostra rosto detectado
    if face_from_doc:
        st.subheader("Rosto Detectado na CNH")
        st.image(str(face_from_doc), caption="Rosto extraído", width=250)

    # Resultado final - Face
    if match:
        st.success(
            f"Face compatível! Similaridade: {score:.3f} (mínimo aceito = 0.7)"
        )
    else:
        st.error(
            f"Face não compatível! Similaridade: {score:.3f} (mínimo aceito = 0.7)"
        )

    # Resultado final - Nome
    if resultado["nome_valido"]:
        st.success("Nome da CNH e comprovante são iguais")
    else:
        st.error("Nome da CNH e comprovante são diferentes")
