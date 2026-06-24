# RAG sobre Normativa de Telecomunicaciones

Sistema de Retrieval-Augmented Generation (RAG) para consultar documentos normativos del sector de telecomunicaciones en Argentina. Permite cargar PDFs de regulaciones y hacer preguntas en lenguaje natural sobre su contenido.

**Demo en vivo:** [HuggingFace Spaces](https://huggingface.co/spaces/Barra-79/TP_PLN_1C26)

---

## Arquitectura

```
PDF → PyPDFLoader → chunks → embeddings → ChromaDB
                                              ↓
pregunta → embedding → búsqueda semántica → contexto → LLM → respuesta
```

| Componente | Tecnología |
|---|---|
| Carga de PDFs | LangChain + PyPDFLoader |
| Fragmentación | RecursiveCharacterTextSplitter (chunk=600, overlap=80) |
| Embeddings | `intfloat/multilingual-e5-large` (sentence-transformers) |
| Base vectorial | ChromaDB |
| LLM local | Ollama + llama3.2:3b |
| LLM cloud | Groq API + llama-3.1-8b-instant |
| Interfaz | Gradio |
| Deploy | HuggingFace Spaces |

---

## Corpus

Resoluciones y normas técnicas de ENACOM (Ente Nacional de Comunicaciones):

- Norma Técnica ENACOM-Q2-60.14 — Dispositivos de banda ancha
- Normas técnicas de homologaciones
- Nota técnica sobre dispositivos de baja potencia
- Resoluciones 57, 793, 729/80, 2097

---

## Instalación local

**Requisitos:** Python 3.10+, [Ollama](https://ollama.com)

```bash
git clone https://github.com/hbarrabasqui/PLN_1C2026_RAG
cd PLN_1C2026_RAG

pip install -r requirements.txt

# Descargar modelo LLM
ollama pull llama3.2:3b

# Iniciar servidor Ollama
ollama serve
```

Abrir `07_proyecto_final_rag_gradio.ipynb` y ejecutar las celdas.

---

## Deploy en HuggingFace Spaces

La versión cloud reemplaza Ollama por la API de Groq (gratuita):

1. Crear cuenta en [console.groq.com](https://console.groq.com) y obtener API key
2. Crear un Space en HuggingFace (SDK: Gradio)
3. Subir `app.py`, `requirements.txt` y los PDFs al repo del Space
4. En Settings → Secrets, agregar `GROQ_API_KEY`

---

## Limitaciones

**Calidad de respuestas:** El modelo `llama-3.1-8b-instant` está optimizado para velocidad. Para documentos técnicos con terminología específica, modelos más grandes (70B+) producen respuestas más precisas.

**Tablas en PDFs:** `PyPDFLoader` extrae texto corrido y pierde la estructura de tablas. Datos como límites de potencia o bandas de frecuencia quedan descontextualizados. Una solución más robusta es `pdfplumber` con serialización de tablas antes de generar los embeddings.

**Contexto limitado:** El pipeline recupera los 3 fragmentos más similares (k=3). Preguntas que requieren información distribuida en múltiples secciones pueden no responderse correctamente.

**Re-indexación al iniciar:** Los PDFs incluidos en el repositorio se indexan automáticamente al arrancar el Space. El proceso tarda unos minutos en la primera carga.

---

## Stack

```
langchain · langchain-groq · chromadb · sentence-transformers
gradio · pypdf · ollama (local) · groq (cloud)
```
