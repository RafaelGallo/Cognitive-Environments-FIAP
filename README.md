# 📄 QuantumFinance – Validação Biométrica de Documentos (Google Cloud Streamlit)

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Vision%20API-yellow.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Status](https://img.shields.io/badge/Status-Ativo-success.svg)

## 📌 Contexto

Este projeto implementa um **pipeline de validação biométrica** para documentos (CNH, RG, comprovantes) utilizando:

* **Google Cloud Vision API** → Extração de texto (OCR) e detecção facial.
* **ImageHash (pHash/wHash)** → Comparação de similaridade entre selfie e foto extraída do documento.
* **Streamlit** → Aplicação web interativa para upload de arquivos e exibição dos resultados.
* **Pandas / JSON** → Consolidação e exportação dos resultados.

O sistema é capaz de:
✅ Extrair texto de documentos com OCR.
✅ Detectar e recortar o rosto da CNH.
✅ Comparar a selfie enviada com o rosto extraído.
✅ Validar se o nome no comprovante corresponde ao da CNH.
✅ Gerar relatórios em **JSON, CSV** e visualização em **Streamlit**.

## 📂 Estrutura do Projeto

```
trabalho_final/
│── app/
│   └── streamlit_app.py        # Aplicação web interativa
│── src/
│   └── utils.py                # Funções de OCR, face extraction e comparação
│── data/
│   ├── 002.JPG                 # CNH de teste
│   ├── 003.jpg                 # Conta/Comprovante
│   ├── LUIZ.png                # Selfie válida
│   ├── Maria.png               # Selfie inválida
│── outputs/                    # Resultados gerados (JSON, CSV, métricas, imagens)
│── cred/
│   └── dts-10-ds-xxxx.json     # Chave de API do Google Vision
│── main.py                     # Pipeline em batch (OCR + Face Match + Export)
│── requirements.txt            # Dependências do projeto
│── README.md                   # Documentação
```
## ⚙️ Instalação

### 1️⃣ Clonar o repositório

```bash
git clone https://github.com/seu-usuario/validador-biometrico.git
cd validador-biometrico
```

### 2️⃣ Criar ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
```

### 3️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 4️⃣ Configurar Google Cloud Vision

* Criar um projeto no [Google Cloud Console](https://console.cloud.google.com/).
* Ativar a **Vision API**.
* Gerar uma chave de serviço JSON e salvar em `cred/dts-10-ds-xxxx.json`.
* Definir a variável de ambiente:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="cred/dts-10-ds-xxxx.json"
# ou no Windows PowerShell:
setx GOOGLE_APPLICATION_CREDENTIALS "cred\dts-10-ds-xxxx.json"
```

## 🚀 Como Usar

### 🔹 Rodar o pipeline batch (main.py)

```bash
python main.py
```

➡️ Isso gera **JSON/CSV** em `outputs/` com os resultados da comparação facial e OCR.

### 🔹 Rodar aplicação web (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

➡️ Interface interativa para upload de CNH, comprovante e selfie.

---

## 📊 Resultados

Exemplo de saída em JSON (`outputs/results_LUIZ.json`):

```json
{
  "documento_nome": "LUIZ ANTONIO DE OLIVEIRA",
  "documento_cpf": "076.763.758-51",
  "comprovante_nome": "LUIZ ANTONIO DE OLIVEIRA",
  "comprovante_endereco": "R JOSE BASILIO GAMA 65, JACAREI/SP",
  "face_match": true,
  "similaridade": 0.92,
  "nome_valido": true
}
```

## 📈 Métricas

O sistema gera automaticamente métricas de desempenho para avaliação do modelo:

* **Acurácia** → Percentual de acertos.
* **Precisão** → Quantidade de verdadeiros positivos sobre todos os positivos previstos.
* **Recall** → Cobertura dos verdadeiros positivos.
* **F1-Score** → Média harmônica entre precisão e recall.
* **Matriz de Confusão** → Exportada em PNG em `outputs/confusion_matrix.png`.

## 🛠️ Tecnologias

* **Python 3.10+**
* **Google Cloud Vision API**
* **Pandas, Numpy**
* **ImageHash (pHash/wHash)**
* **Streamlit**
* **Seaborn / Matplotlib**

## ✅ Próximos Passos

* [ ] Melhorar comparação facial com modelos de embeddings (ex.: `face_recognition`, `DeepFace`).
* [ ] Criar banco de dados para armazenar históricos de validações.
* [ ] Adicionar testes unitários (pytest).
* [ ] Deploy da aplicação Streamlit em **Streamlit Cloud** ou **GCP App Engine**.

---

## 📜 Licença

Este projeto é de uso acadêmico/profissional e segue a licença MIT.