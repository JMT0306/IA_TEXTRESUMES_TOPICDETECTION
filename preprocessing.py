import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Configuração do NLTK
nltk_data_dir = 'nltk_data'
nltk.data.path.append(nltk_data_dir)

# Baixar os recursos necessários, se não estiverem disponíveis
for resource in ['punkt', 'stopwords', 'wordnet']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

def preprocess_text(text):
    """
    Pré-processa o texto: tokenização, remoção de stopwords e lematização.
    """
    stop_words = set(stopwords.words('portuguese'))
    lemmatizer = WordNetLemmatizer()

    # Tokenizar texto
    tokens = word_tokenize(text.lower(), language='portuguese')

    # Remover stopwords, palavras curtas e lematizar
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word.isalnum() and len(word) > 2 and word not in stop_words
    ]
    return tokens
