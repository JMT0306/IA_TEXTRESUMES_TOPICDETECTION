import streamlit as st
from preprocessing import preprocess_text
from models import create_bow, apply_kmeans, apply_lda


def generate_summary(document, topic_keywords):
    """
    Gera um resumo simples selecionando frases relevantes do texto original.
    """
    sentences = document.split('.')
    summary = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in topic_keywords):
            summary.append(sentence.strip())
    return '. '.join(summary) + '.'


# Configuração da Interface
st.title("Sistema de Deteção de Tópicos e Recomendações")
st.write("Faça upload de textos académicos ou insira o seu texto para análise.")

# Input de texto manual
user_input = st.text_area("Insira o seu texto (opcional):", height=200)

# Upload de ficheiros
uploaded_files = st.file_uploader("Faça upload de ficheiros (.txt):", accept_multiple_files=True, type=['txt'])

# Processamento de textos
documents = []

if user_input:
    documents.append(user_input)

if uploaded_files:
    for file in uploaded_files:
        try:
            documents.append(file.read().decode("utf-8"))
        except UnicodeDecodeError:
            documents.append(file.read().decode("latin-1"))  # Fallback

if documents:
    st.write(f"**{len(documents)} textos carregados com sucesso!**")

    # Pré-processamento
    preprocessed_docs = [preprocess_text(doc) for doc in documents]

    # Bag-of-Words
    bow_matrix, vectorizer = create_bow(preprocessed_docs)

    # Clusterização com k-means
    num_clusters = st.slider("Número de clusters:", min_value=2, max_value=10, value=2)
    kmeans = apply_kmeans(bow_matrix, num_clusters)

    # Exibir clusters
    st.write("**Clusters atribuídos:**")
    for i, cluster in enumerate(kmeans.labels_):
        st.write(f"Documento {i + 1}: Cluster {cluster}")

    # Modelagem de tópicos com LDA
    lda_model, corpus, dictionary = apply_lda(preprocessed_docs, num_topics=num_clusters)

    st.write("**Tópicos gerados pelo LDA:**")
    for idx, topic in lda_model.print_topics(num_words=10):  # Mostrar 10 palavras por tópico
        st.write(f"Tópico {idx + 1}: {topic}")

    # Extrair palavras-chave do primeiro tópico
    topic_keywords = [word for word, _ in lda_model.show_topic(0, topn=5)]

    # Recomendações
    if st.button("Gerar Recomendações"):
        recommendations = []
        for i, doc in enumerate(preprocessed_docs):
            if any(keyword in doc for keyword in topic_keywords):
                recommendations.append(documents[i])

        st.write("**Documentos recomendados com base no primeiro tópico:**")
        if recommendations:
            for rec in recommendations:
                st.write(f"- {rec}")
        else:
            st.write("Nenhuma recomendação encontrada.")

    # Resumo baseado no primeiro tópico
    st.write("**Resumo gerado com base no primeiro tópico:**")
    summary = generate_summary(documents[0], topic_keywords)
    st.write(summary)
else:
    st.write("**Nenhum texto carregado. Insira ou faça upload de textos.**")
