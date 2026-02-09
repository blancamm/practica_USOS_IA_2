# Proyecto Final - Asistente de Estudio con Mapa Mental (sin API key)

## Qué hace
- Lee PDFs en /data
- Crea embeddings y agrupa por temas
- Genera un mapa mental en Mermaid: outputs/mindmap.mmd
- (Opcional) Permite hacer preguntas usando RAG local

## Requisitos
Python 3.10+

## Instalación
Creación de un entorno para instalar los requerimientos, escribiendo en la terminal:
pip install -r requirements.txt

## Uso
1) Mete PDFs en /data
2) Ejecuta:
python main.py
3) Abre outputs/mindmap.mmd en un visor Mermaid (o pégalo en un Markdown compatible)

## Casos de uso demostrados (3)
1) Generación de mapa mental
2) Identificación de temas principales (outputs/mindmap.json)
3) Modo QA opcional (RAG local)
