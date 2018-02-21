# _____TF-IDF libraries_____
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy
import nltk
from nltk.corpus import stopwords
import nltk.data
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import StanfordNERTagger

# _____helper Libraries____
import csv
import timeit
import random
import sys
import collections
import rhetoricalquestion
import json
import os
import requests, zipfile
from io import BytesIO

reload(sys)  
sys.setdefaultencoding('utf8')
'''
get_stanford_ner = requests.get("https://nlp.stanford.edu/software/stanford-ner-2017-06-09.zip")
zip_file = zipfile.ZipFile(BytesIO(get_stanford_ner.content))
zip_file.extractall()
os.system("cp stanford-ner-2017-06-09/stanford-ner.jar stanford-ner.jar")
os.system("cp stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz english.all.3class.distsim.crf.ser.gz")
'''
st = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz',
					   'stanford-ner.jar',
					   encoding='utf-8')
target_count = 0
list_sentence = []

# Setup
snow_stem = nltk.stem.SnowballStemmer("english")

sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")

# Hardcoded word lists
yesnowords = ["can", "could", "would", "is", "does", "has", "was", "were", "had", "have", "did", "are", "will"]
commonwords = ["the", "a", "an", "is", "are", "were", "."]
questionwords = ["who", "what", "where", "when", "why", "how", "whose", "which", "whom"]

# Take in a tokenized question and return the question type and body
def user_query_process(query):
    
    # Find "question word" (what, who, where, etc.)
    questionword = ""
    quest_index = -1

    for (idx, word) in enumerate(query):
        if word.lower() in questionwords:
            questionword = word.lower()
            quest_index = idx
            break
        elif word.lower() in yesnowords:
            return ("YESNO", query)

    if quest_index < 0:
        return ("MISC", query)

    if quest_index > len(query) - 3:
        target = query[:quest_index]
    else:
        target = query[quest_index+1:]
    entity_type = "MISC"

    # Determine question type
    if questionword in ["who", "whose", "whom"]:
        entity_type = "PERSON"
    elif questionword == "where":
        entity_type = "PLACE"
    elif questionword == "when":
        entity_type = "TIME"
    elif questionword == "how":
        if target[0] in ["few", "little", "much", "many"]:
            entity_type = "QUANTITY"
            target = target[1:]
        elif target[0] in ["young", "old", "long"]:
            entity_type = "TIME"
            target = target[1:]

    # Trim possible extra helper verb
    if questionword == "which":
        target = target[1:]
    if target[0] in yesnowords:
        target = target[1:]
    
    # Return question data
    return (entity_type, target)

# Get command line arguments
articlefilename = sys.argv[1]

# Process article file
article = open(articlefilename, 'r').read()
article = sent_detector.tokenize(article)

def initial_greet_talk(greet_sent_sentence):
    synonyms = {"temperature": ["weather"],
                "who":["what","how","when","which batsman","which player"],
                "how": ["what", "whwn", "when"],
                "when": ["what", "how", "who"],
                "what": ["who", "how", "when"]}
    stopWords = ("didn't", 'been', 'wouldn', 'because', 'isn', "you'd", 'just', 'his', "weren't", 'not', 'having', 'about', 'down', 'him', 'shan', 'ours', "you've", 'yours', "hadn't", 'and', 'after', 'can', 'each', 'mightn', 'the', 'over', 'd', 'once', 'such', 'this', 'with', "hasn't", 'ma', 'mustn', 'or', 'only', 'too', 'did', 'was', 'll', 'hadn', 'weren', 'y', 'be', 'doing', 'it', 'further', 'there', 'any', 'herself', 'he', 'being', 'when', 'm', "aren't", 'up', "haven't", 'myself', 'shouldn', "don't", 'more', 'if', 'her', 'needn', 'your', 'ourselves', 'between', 'against', 'itself', 'does', 'she', 'while', "isn't", "doesn't", 'own', 'through', 'didn', 'its', 'by', "wouldn't", 'where', 'had', "won't", 'you', 'some', 'ain', 'am', "that'll", 'so', 'during', 'won', 'few', 'o', 'a', 'couldn', 'above', 're', 'on', 's', "needn't", 'out', "couldn't", 'but', 'in', 'himself', "shouldn't", 'we', "mightn't", 'aren', 'before', 'until', 'most', 'very', 'they', 'both', 'all', 'our', 'of', 'to', 'nor', 'than', "should've", 'an', 'here', 'at', 'their', 'again', "it's", 'are', 'now', 'theirs', 'don', 'haven', 'i', "shan't", 'other', 'is', 'has', 'doesn', 'hasn', 'have', "she's", 'hers', "mustn't", 'below', "wasn't", 't', 'under', 'do', 'then', 'were', 've', 'should', 'themselves', 'my', 'off', 'no', 'from', 'as', 'same', 'for', 'into', 'me', 'wasn', 'will', "you're", 'yourself', 'them', 'yourselves', "you'll", 'these', 'those', 'that')
    liss = greet_sent_sentence.split()
    fin_array=[]
    for iter in liss:
        if iter not in stopWords:
            fin_array.append(iter)
    greet_sent_sentence=" ".join(fin_array)
    csv_file_path = "Chatbot.csv"
    i = 0
    sentences = []
    sentences.append(" BJB ")
    sentences.append(" BJB4 ")
    
    # processing response
    greet_sent = [greet_sent_sentence]
    greet_sent_sentence=greet_sent_sentence.split()
    for each in range(len(greet_sent_sentence)):
        if greet_sent_sentence[each] in synonyms:
            for items in synonyms[greet_sent_sentence[each]]:
                greet_sent.append(" ".join(greet_sent_sentence[0:each]+[items]+greet_sent_sentence[each+1:len(greet_sent_sentence)]))

    with open(csv_file_path, "r") as sentences_file:
        reader = csv.reader(sentences_file, delimiter=',')
        for row in reader:
            wordsFiltered=[]
            data=word_tokenize(row[0])
            for w in data:
                if w not in stopWords:
                    wordsFiltered.append(w)

            sentences.append(" ".join(wordsFiltered))
            i += 1
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(sentences)  # finds the tfidf score with normalization

    tfidf_matrix_test = tfidf_vectorizer.transform(greet_sent)

    cosine = cosine_similarity(tfidf_matrix_test, tfidf_matrix_train)

    cosine = numpy.delete(cosine, 0)
    max = cosine.max()
    response_index = 0
    if (max > 0.1):
        new_max = max - 0.01
        list = numpy.where(cosine == max)
        response_index = random.choice(list[0])

    else:
        return None

    j = 0
    with open(csv_file_path, "r") as sentences_file:
        reader = csv.reader(sentences_file, delimiter=',')
        for row in reader:
            j += 1  # we begin with 1 not 0 &    j is initialized by 0
            if j == response_index:

                return row[1], response_index,
                break
print "Agribot says : Hello"
while 1:
    greet = raw_input('Your Response : ')
    if greet.lower()=="yes":
        break
    if greet.lower()=="no":
        sys.exit()
    reply=initial_greet_talk(greet)
    if reply:
        response_primary, line_id_primary = initial_greet_talk(greet)
        print "Agribot says : ",response_primary
    else:
        print "Agribot says : Sorry I dont Understand. Can you please repeat, Do you want to start talking about agriculture?(yes - to start / no - to exit)"

# Iterate through all questions
print "Agribot says : What is your question to me?"
while True:
    question = raw_input('Your Response: ').strip() 

    if question.lower()=="bye":
        break
    # Answer is not yet processed
    done = False

    # Tokenize question
    #print (question)
    query = nltk.word_tokenize(question.replace('?', ''))
    questionPOS = nltk.pos_tag(query)

    # Process question
    (entity_type, target) = user_query_process(query)

    # Handling yes/no questions
    if entity_type == "YESNO":
        yesno.answeryesno(article, query)
        continue

    # Get sentence keywords
    searchwords = set(target).difference(commonwords)
    dict = collections.Counter()
        
    # Find most relevant sentences
    for (i, sent) in enumerate(article):
        sentwords = nltk.word_tokenize(sent)
        wordmatches = set(filter(set(searchwords).__contains__, sentwords))
        dict[sent] = len(wordmatches)
    
    max_match = max(dict.values())

    for i in dict:
        if max_match == dict[i]:
            list_sentence.append(i)
    if len(list_sentence) == 1:
        tokens = nltk.word_tokenize(list_sentence[0])
        target_stem = snow_stem.stem(target[-1])
        for word in tokens:
            stemmed = snow_stem.stem(word)
            if stemmed == target_stem:
                endidx = list_sentence[0].index(word)
            else:
                if list_sentence[0].index(target[-1]) is None:
                    proc_answer = "Sorry I don't understand the code"
                else:
                    endidx = list_sentence[0].index(target[-1])               

        #proc_answer = list_sentence[0][:endidx + len(target[-1])]
        proc_answer = list_sentence[0]
        done = True
    else:
        # Focus on 10 most relevant sentences
        for (most_common_sent, matches) in dict.most_common(10):
            tokens = nltk.word_tokenize(most_common_sent)
            parse = st.tag(tokens)
            sentencePOS = nltk.pos_tag(nltk.word_tokenize(most_common_sent))


            #stemmed values are being checked
            for each_target in target:
                if each_target in most_common_sent:
                    target_count+=1
            if target_count == len(target):
                tokens = nltk.word_tokenize(list_sentence[0])
                target_stem = snow_stem.stem(target[-1])
                for word in tokens:
                    stemmed = snow_stem.stem(word)
                if stemmed == target_stem:
                    endidx = list_sentence[0].index(word)
                else:
                    if list_sentence[0].index(target[-1]) is None:
                        proc_answer = "Sorry I don't understand the code"
                    else:
                        endidx = list_sentence[0].index(target[-1])               

                #proc_answer = list_sentence[0][:endidx + len(target[-1])]
                proc_answer = list_sentence[0]
                done  = True
            # Check if solution is found
            if done:
                continue

            # Check by question type
            proc_answer = ""
            for worddata in parse:
                # Mentioned in the question
                if worddata[0] in searchwords:
                    continue
            
                if entity_type == "PERSON":
                    if worddata[1] == "PERSON":
                        proc_answer = proc_answer + " " + worddata[0]
                        done = True
                    elif done:
                        break

                if entity_type == "PLACE":
                    if worddata[1] == "LOCATION":
                        proc_answer = proc_answer + " " + worddata[0]
                        done = True
                    elif done:
                        break

                if entity_type == "QUANTITY":
                    if worddata[1] == "NUMBER":
                        proc_answer = proc_answer + " " + worddata[0]
                        done = True
                    elif done:
                        break

                if entity_type == "TIME":
                    if worddata[1] == "NUMBER":
                        proc_answer = proc_answer + " " + worddata[0]
                        done = True
                    elif done:
                        proc_answer = proc_answer + " " + worddata[0]
                        break
            
    if done:
        print "Agribot says : ",proc_answer
    if not done:
        (proc_answer, matches) = dict.most_common(1)[0]
        print "Agribot says : ", proc_answer
