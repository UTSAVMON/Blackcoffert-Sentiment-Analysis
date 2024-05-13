import requests
import pandas as pd
from bs4 import BeautifulSoup
import regex as re
import nltk
from nltk.corpus import cmudict
nltk.download('cmudict')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import re
import math

data = pd.read_excel("Input.xlsx")
data

a = data.URL
b = data.URL_ID
ID = []
text=[]
cleaned_text = []
URL_Link = []

for i in range(100):
    url = a[i]
    response =  requests.get(url)
    if response.status_code == 200 :
        soup = BeautifulSoup(response.content, "html.parser")
        codes = soup.findAll("div",{"class": "td-post-content tagdiv-type"})
        text=[]
        for t in codes:
            f = t.get_text()
            text.append(f)
        for j in text:
            item = re.sub('\n', '', j)
            cleaned_text.append(item)
            ID.append(b[i])
            URL_Link.append(a[i])

for i in range(len(ID)):
    f = open(ID[i] + ".txt", "w" , encoding="utf-8")
    f.write(cleaned_text[i])
f.close()

zip_data = list(zip(ID, URL_Link, cleaned_text))

df = pd.DataFrame(zip_data, columns=['ID', 'URL_Link' , 'Text'])
Text = df.Text
df

lemma = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def text_prep(x):
     corp = str(x).lower()
     corp = re.sub('[^a-zA-Z]+',' ', corp).strip()
     tokens = word_tokenize(corp)
     words = [t for t in tokens if t not in stop_words]
     lemmatize = [lemma.lemmatize(w) for w in words]
     return lemmatize

preprocess_tag = [text_prep(i) for i in df["Text"]]
df["preprocess_txt"] = preprocess_tag

df['total_len'] = df['preprocess_txt'].map(lambda x: len(x))

file = open('Negative_Texts.txt', 'r', encoding="utf8")
neg_words = file.read().split()
file = open('Positive_Texts.txt', 'r', encoding="utf8")
pos_words = file.read().split()

num_pos = df['preprocess_txt'].map(lambda x: len([i for i in x if i in pos_words]))
df['pos_count'] = num_pos
num_neg = df['preprocess_txt'].map(lambda x: len([i for i in x if i in neg_words]))
df['neg_count'] = num_neg

df['sentiment'] = round((df['pos_count'] - df['neg_count']) / df['total_len'], 2)

df["polar_score"] = round((df['pos_count'] - df['neg_count']) / ((df['pos_count'] + df['neg_count']) + 0.000001), 2)

df.head()

def Avg_sen_len(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    num_words = len(words)
    num_sentences = len(sentences)
    if num_sentences == 0:
        return 0
    else:
        return (num_words / num_sentences)

def Per_Complex_words(text):
    words = word_tokenize(text)
    cmu = cmudict.dict()
    num_complex_words = sum(1 for word in words if word.lower() in cmu)
    num_words = len(words)
    if num_words == 0:
        return 0
    else:
        return (num_complex_words / num_words * 100)

def fog_index(text):
    avg_sentence_length = Avg_sen_len(text)
    percentage_complex_words = Per_Complex_words(text)
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    return (fog_index)

def avg_words_per_sentence(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
    return (total_words / len(sentences))

def complex_word_count(text):
    words = word_tokenize(text)
    stopwords_set = set(stopwords.words("english"))
    complex_words = [word for word in words if len(wn.synsets(word)) > 2 and word not in stopwords_set]
    return (len(complex_words))

def word_count(text):
    words = word_tokenize(text)
    return (len(words))

def syllables_count(word):
    return sum(1 for vowel in word if vowel.lower() in 'aeiou')

def syllables_per_word(text):
    words = word_tokenize(text)
    syllables = sum(syllables_count(word) for word in words)
    return (syllables / len(words))

def personal_pronouns_count(text):
    words = word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    personal_pronouns = [word for word, tag in tagged_words if tag == 'PRP']
    return (len(personal_pronouns))

def avg_word_length(text):
    words = word_tokenize(text)
    total_length = sum(len(word) for word in words)
    return (total_length / len(words))

text_list = df["Text"]

Average_Sentence_Length = []
Percentage_Of_Complex_Words =[]
Fog_Index = []
Average_Words_Per_Sentence = []
Complex_Words_Count = []
Word_Count = []
Syllables_Count = []
Syllables_Per_Word = []
Personal_Pronouns_Count = []
Average_Word_Length = []

for i in range(len(text_list)):
    text = text_list[i]
    Average_Sentence_Length.append(Avg_sen_len(text))
    Percentage_Of_Complex_Words.append(Per_Complex_words(text))
    Fog_Index.append(fog_index(text))
    Average_Words_Per_Sentence.append(avg_words_per_sentence(text))
    Complex_Words_Count.append(complex_word_count(text))
    Word_Count.append(word_count(text))
    Syllables_Count.append(syllables_count(text))
    Syllables_Per_Word.append(syllables_per_word(text))
    Personal_Pronouns_Count.append(personal_pronouns_count(text))
    Average_Word_Length.append(avg_word_length(text))

df_new = pd.DataFrame(list(zip(Average_Sentence_Length, Percentage_Of_Complex_Words, Fog_Index,
                           Average_Words_Per_Sentence, Complex_Words_Count, Word_Count,
                           Syllables_Count, Syllables_Per_Word, Personal_Pronouns_Count, Average_Word_Length)),
               columns =['Average_Sentence_Length', 'Percentage_Of_Complex_Words', 'Fog_Index',
                         'Average_Words_Per_Sentence', 'Complex_Words_Count', 'Word_Count',
                         'Syllables_Count', 'Syllables_Per_Word', 'Personal_Pronouns_Count', 'Average_Word_Length'])

dataframe = pd.concat([df, df_new], axis=1, join='inner')

output = dataframe.copy()
output

final_output = output.drop(['Text', 'preprocess_txt'], axis=1)
final_output

final_output.to_excel("Output Data Structure.xlsx", index = False)

