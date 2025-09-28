import io
from google.cloud import vision
from PIL import Image
import imagehash


def extract_text(client, image_path: str) -> str:
    """
    Extrai texto de uma imagem usando Google Cloud Vision OCR.

    Args:
        client: Cliente autenticado do Google Vision.
        image_path (str): Caminho para a imagem de entrada.

    Returns:
        str: Texto extraído da imagem (ou string vazia se não houver texto).
    """
    with io.open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(response.error.message)

    return texts[0].description if texts else ""


def extract_face_and_save(client, image_path: str, output_file: str):
    """
    Extrai o rosto principal de um documento e salva em arquivo.

    Args:
        client: Cliente autenticado do Google Vision.
        image_path (str): Caminho para a imagem de entrada.
        output_file (str): Caminho para salvar o rosto recortado.

    Returns:
        str | None: Caminho do arquivo salvo ou None se não detectar rosto.
    """
    with io.open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    if not faces:
        print("Nenhum rosto detectado na imagem.")
        return None

    with Image.open(image_path) as pil_img:
        face = faces[0]
        x_min = min(v.x for v in face.bounding_poly.vertices)
        y_min = min(v.y for v in face.bounding_poly.vertices)
        x_max = max(v.x for v in face.bounding_poly.vertices)
        y_max = max(v.y for v in face.bounding_poly.vertices)

        face_crop = pil_img.crop((x_min, y_min, x_max, y_max))
        face_crop.save(output_file)

        print(f"Rosto salvo em: {output_file}")
        return output_file


def compare_faces(img1_path: str, img2_path: str, threshold: float = 0.75):
    """
    Compara duas imagens faciais usando perceptual hash (pHash).

    Args:
        img1_path (str): Caminho da primeira imagem (selfie).
        img2_path (str): Caminho da segunda imagem (rosto extraído da CNH).
        threshold (float): Valor mínimo de similaridade aceitável (0 a 1).

    Returns:
        tuple: (match, similaridade) onde:
            match (bool): True se similaridade >= threshold.
            similaridade (float): Valor da similaridade (0 a 1).
    """
    try:
        hash1 = imagehash.phash(Image.open(img1_path))
        hash2 = imagehash.phash(Image.open(img2_path))
        diff = hash1 - hash2
        similarity = 1 - (diff / 64.0)  # hash de 64 bits

        match = similarity >= threshold
        return match, float(similarity)

    except Exception as e:
        print(f"Erro na comparação: {e}")
        return False, 0.0
