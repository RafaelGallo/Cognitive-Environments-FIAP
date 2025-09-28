import face_recognition
import cv2

def compare_faces_embeddings(img1_path: str, img2_path: str, tolerance: float = 0.6):
    """
    Compara duas imagens usando embeddings faciais (face_recognition).
    Retorna (match, distance).
    - Quanto menor a distância, mais parecidas são as faces.
    - tolerance padrão = 0.6 (valor recomendado pela lib).
    """
    # Carregar imagens
    img1 = face_recognition.load_image_file(img1_path)
    img2 = face_recognition.load_image_file(img2_path)

    # Extrair embeddings
    enc1 = face_recognition.face_encodings(img1)
    enc2 = face_recognition.face_encodings(img2)

    if not enc1 or not enc2:
        return False, 1.0  # Nenhum rosto detectado

    # Pegar o primeiro rosto encontrado em cada imagem
    dist = face_recognition.face_distance([enc1[0]], enc2[0])[0]

    return dist <= tolerance, float(dist)
