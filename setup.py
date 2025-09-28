from setuptools import setup, find_packages

setup(
    name="trabalho_final_cloud_cognitive",
    version="0.1.0",
    description="Projeto Integrado MBA - Cloud & Cognitive (OCR + Face Match com Google Cloud Vision API)",
    author="Rafael Gallo",
    author_email="seu_email@example.com",  # substitua pelo seu
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "pandas",
        "matplotlib",
        "google-cloud-vision",
        "google-auth",
        "google-auth-oauthlib",
        "google-auth-httplib2"
    ],
    python_requires=">=3.8",
)
