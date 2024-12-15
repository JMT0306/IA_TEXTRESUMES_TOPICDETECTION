import streamlit as st
from preprocessing import preprocess_text
from models import create_bow, apply_kmeans, apply_lda


def generate_summary(document, topic_keywords, num_sentences=3):
    """
    Gera um resumo baseado nas palavras-chave e frases mais relevantes.
    """
    sentences = document.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]

    summary = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in topic_keywords):
            summary.append(sentence)
        if len(summary) >= num_sentences:
            break

    return '. '.join(summary) + '.' if summary else "Não foi possível gerar um resumo."


# Configuração da Interface
st.title("Sistema de Deteção de Tópicos, Resumos e Análise de Textos Académicos")
st.write("Faça upload de textos académicos ou insira o seu texto para análise.")

# Input de texto manual
user_input = st.text_area("Insira o seu texto (opcional):", height=200)

# Upload de ficheiros
uploaded_files = st.file_uploader("Faça upload de ficheiros (.txt):", accept_multiple_files=True, type=['txt'])

# Processamento de textos
documents = []

# Ler textos inseridos manualmente
if user_input:
    documents.append(user_input)

# Ler ficheiros carregados
if uploaded_files:
    for file in uploaded_files:
        try:
            content = file.read().decode("utf-8")
            if content.strip():
                documents.append(content)
        except UnicodeDecodeError:
            try:
                content = file.read().decode("latin-1")
                if content.strip():
                    documents.append(content)
            except Exception as e:
                st.warning(f"Erro ao ler o ficheiro '{file.name}': {e}")

if not documents:
    st.error("Nenhum texto válido foi carregado. Insira texto ou faça upload de ficheiros corretos.")
    st.stop()

st.write(f"**{len(documents)} textos carregados com sucesso!**")

# Pré-processamento
preprocessed_docs = [preprocess_text(doc) for doc in documents]
preprocessed_docs = [doc for doc in preprocessed_docs if len(doc) > 0]

if not preprocessed_docs:
    st.error("Todos os documentos estão vazios após o pré-processamento. Verifique os textos.")
    st.stop()

# Bag-of-Words
bow_matrix, vectorizer = create_bow(preprocessed_docs)

# Clusterização com k-means
num_clusters = st.slider("Número de clusters:", min_value=2, max_value=10, value=2)
if len(preprocessed_docs) < num_clusters:
    num_clusters = len(preprocessed_docs)
kmeans = apply_kmeans(bow_matrix, num_clusters)

# Exibir clusters
st.write("**Clusters atribuídos:**")
for i, cluster in enumerate(kmeans.labels_):
    st.write(f"Documento {i + 1}: Cluster {cluster}")

# Modelagem de tópicos com LDA
lda_model, corpus, dictionary = apply_lda(preprocessed_docs, num_topics=num_clusters)

st.write("**Tópicos gerados pelo LDA:**")
for idx, topic in lda_model.print_topics(num_words=10):
    st.write(f"Tópico {idx + 1}: {topic}")

# Extrair palavras-chave do primeiro tópico
topic_keywords = [word for word, _ in lda_model.show_topic(0, topn=5)]

# Resumo baseado no primeiro tópico
st.write("**Resumo gerado com base no primeiro tópico:**")
summary = generate_summary(documents[0], topic_keywords)
st.write(summary)
