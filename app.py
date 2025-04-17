import streamlit as st
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
import pandas as pd
from textblob import TextBlob
from collections import Counter

# run this in terminal; python -m nltk.downloader all
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stopwords=set(stopwords.words('english'))

def clean(text):
    text=text.lower()

    text=re.sub(r'[^a-zA-z0-9\s]','',text)

    text=word_tokenize(text)

    text=[w for w in text if w not in stopwords]
    return " ".join(text)

def get_keywords(text):
    blob = TextBlob(text)

    # Extract only the desired POS tags
    keywords = [word for word, tag in blob.tags if tag in (
        'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
        'JJ', 'JJR', 'JJS',          # Adjectives
        'RB', 'RBR', 'RBS'           # Adverbs
    )]

    keyword_counts = Counter(keywords)
    most_common_keywords = keyword_counts.most_common(5)
    
    return most_common_keywords


# App UI Design
st.title("Auto Keyword Extraction From Articles Text Using TextBlob")
Upload_file=st.sidebar.file_uploader('Upload a csv file',type='csv')


if Upload_file:
    df=pd.read_csv(Upload_file)
    st.write("Uploaded File")
    st.dataframe(df.head())

    if 'abstract' in df.columns and not df['abstract'].empty():
        df['cleaned_abstract']=df['abstract'].apply(clean)
        df['keywords']=df['cleaned_abstract'].apply(get_keywords)

        st.write('With keywords file')
        st.dataframe(df)

        if 'abstract' in df.columns and not df['abstract'].empty():
            df['cleaned_abstract']=df['abstract'].apply(clean)
            df['keywords']=df['cleaned_abstract'].apply(get_keywords)
            

            st.write("With keywords file")
            st.dataframe(df)

        else:
            st.write("Uploaded data must have (abstract column)")