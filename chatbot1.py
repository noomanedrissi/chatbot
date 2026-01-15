import streamlit as st
import nltk
import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# 1. Load Data
try:
    with open('Tunisia1.txt', 'r', encoding='utf-8') as f:
        data = f.read().replace('\n', ' ')
except FileNotFoundError:
    st.error("Error: 'Tunisia.txt' not found. Please ensure the file exists.")
    st.stop()

# 2. Tokenize into original sentences for the final output
sentences = sent_tokenize(data)

# 3. Preprocessing function
def preprocess(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    words = word_tokenize(sentence.lower())
    # Filter stopwords and punctuation, then lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(w) for w in words 
        if w not in stop_words and w not in string.punctuation
    ]
    # IMPORTANT: Join with a space so the Vectorizer sees separate words
    return " ".join(cleaned_words)

# 4. Prepare Corpus and Vectorizer
# We create a processed version for calculations, but keep 'sentences' for the UI
corpus = [preprocess(s) for s in sentences]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def chatbot(question):
    # Process the user input the same way as the corpus
    processed_question = preprocess(question)
    question_tfidf = vectorizer.transform([processed_question])
    
    # Calculate similarity
    similarity = cosine_similarity(question_tfidf, tfidf_matrix)
    index = similarity.argmax()
    
    # Check if there is any similarity at all
    if similarity[0][index] < 0.1:
        return "I'm sorry, I couldn't find specific information about that in my database."
    
    # Return the ORIGINAL sentence (not the processed one) for readability
    return sentences[index]

# 5. Streamlit UI
st.title("N. DRISSI Chatbot about Tunisia")

user_question = st.text_input("Ask a question about Tunisia (type 'quit' to exit):")

if user_question:
    if user_question.lower() == 'quit':
        st.warning("Session ended.")
        st.stop() 
    
    if "hello" in user_question.lower() or "hi" in user_question.lower():
        st.info("Bot: Hi there! How can I help you today?")
    else:
        with st.spinner('Searching for the best answer...'):
            response = chatbot(user_question)
            st.markdown(f"**Bot Response:**")

            st.write(response)
