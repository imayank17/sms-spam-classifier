import nltk
nltk.data.path.append("nltk_data")

nltk.download('punkt')# Download punkt tokenizer for word_tokenize
nltk.download('stopwords')


import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



# Setup
ps = PorterStemmer()


# --- TEXT TRANSFORMATION FUNCTION ---
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = [i for i in text if i.isalnum()]
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    y = [ps.stem(i) for i in y]

    return " ".join(y)


# --- LOAD MODEL & VECTORIZER ---
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- PAGE TITLE + SUBTITLE ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📩 SMS/Email Spam Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Detect whether a message is Spam or Ham!</p>",
            unsafe_allow_html=True)
st.markdown("---")

# --- INPUT SECTION ---
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    input_sms = st.text_area(
        "📩 Enter your message here:",
        height=100, # Increase height (default is ~100)
        width=500,
        placeholder="Type your message here..."
    )

# --- PREDICTION ---
# Slightly shift the button to the right using columns
btn_space1, btn_space2, btn_space3 = st.columns([2.2, 2, 1])
with btn_space2:
    predict = st.button('🔍 Predict')
if predict:
    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display Result
    st.markdown("---")
    if result == 1:
        st.error("🚫 ** This message is SPAM!**")
    else:
        st.success("✅ **This message is NOT a Spam.**")


# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 13px;'>Made with ❤️ using Streamlit & Scikit-learn</p>",
            unsafe_allow_html=True)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
