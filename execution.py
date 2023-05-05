import numpy as np
import pandas as pd
import tensorflow
# !pip install transformers
from transformers import BertForQuestionAnswering
from transformers import AutoTokenizer
from transformers import pipeline
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tag import pos_tag
nltk.download('averaged_perceptron_tagger')
from keras.models import load_model
import pickle

def location_recommendation(input_text):
    from nltk.corpus import stopwords
    stopwords=stopwords.words('english')
    ",".join(stopwords[:116])
    selected_stopwords=stopwords[:116]
    def preprocess(input_text):
        # Convert to lowercase
        input_text = input_text.lower()
        # Remove stopwords
        stop_words = selected_stopwords
        tokens = input_text.split()
        tokens = [word for word in tokens if not word in stop_words and word.isalpha()]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        tagged_tokens = pos_tag(tokens)
        tokens = [word for word, tag in tagged_tokens if tag != 'NNP' and tag != 'NNPS']
        input_text = ' '.join(tokens)
        return input_text
    # input_text="horrible customer service hotel stay february 3rd 4th 2007my friend picked hotel monaco appealing website online package included champagne late checkout 3 free valet gift spa weekend, friend checked room hours earlier came later, pulled valet young man just stood, asked valet open said, pull bags didn__Ç_é_ offer help, got garment bag suitcase came car key room number says not valet, car park car street pull, left key working asked valet park car gets, went room fine bottle champagne oil lotion gift spa, dressed went came got bed noticed blood drops pillows sheets pillows, disgusted just unbelievable, called desk sent somebody 20 minutes later, swapped sheets left apologizing, sunday morning called desk speak management sheets aggravated rude, apparently no manager kind supervisor weekend wait monday morning, young man spoke said cover food adding person changed sheets said fresh blood rude tone, checkout 3pm package booked, 12 1:30 staff maids tried walk room opening door apologizing closing, people called saying check 12 remind package, finally packed things went downstairs check, quickly signed paper took, way took closer look room, unfortunately covered food offered charged valet, called desk ask charges lady answered snapped saying aware problem experienced monday like told earlier, life treated like hotel, not sure hotel constantly problems lucky ones stay recommend anybody know,  "
    input_preprocessed=preprocess(input_text)
    input_preprocessed=pd.Series(input_preprocessed)
    with open('X_train_preprocessed.pkl', 'rb') as f:
        X_train_preprocessed = pickle.load(f)
    tokenizer = Tokenizer(num_words=2500, lower= True)
    tokenizer.fit_on_texts(X_train_preprocessed)
    input_tokenized=tokenizer.texts_to_sequences(input_preprocessed)
    max_len = 150
    input_padded=pad_sequences(input_tokenized, padding='post', maxlen=max_len)
    input_reshaped= np.array(input_padded).reshape((input_padded.shape[0],input_padded.shape[1],1))
    loaded_model = load_model('lstm_model.h5')
    predictions = loaded_model.predict(input_reshaped)
    mapping = {'Disappointed': 0, 'Sad': 1, 'Neutral': 2, 'Joy': 3, 'Excited': 4}
    # Assuming you have a numerical rating stored in a variable called 'numerical_rating'
    emotion = list(mapping.keys())[list(mapping.values()).index(np.argmax(predictions))] # finding the key corresponding to the numerical rating
    # # print(emotion)

    df=pd.read_csv("output_cleaned.csv")

    df=df[['Location','Wikipedia','Tags','City']]
    df=df.dropna()

    mapped_tags={'excited_dict' :['Amusement Parks',  'Bowling Alleys', 'Escape Rooms', 'Film Studios', 'Hot Spring Resorts', 
        'Jungle Leaps', 'Science & Technology Museums', 'Theme Parks', 'Water Parks', 'Water Sports','Wineries/Distilleries'],
    'joy_dict' : ['Beaches', 'Castles', 'Caves', 'Famous Residences', 'Fountains', 'Lakes', 'Lighthouses', 'Markets', 'Monuments', 'Museums', 
        'Observation Decks', 'Palaces', 'Rivers', 'Seashores', 'Stadiums', 'Statues/Sculptures', 'Theaters','Hill Station',
        'Waterfalls', 'Zoos', 'Recreation Centers','Dams'],
    'neutral_dict' :['Ancient Towns', 'Bridges', 'Canyons', 'Cemeteries', 'Exhibition Halls',  'Geological Sites', 'Libraries', 'Memorial Halls', 
        'Modern Architecture', 'Recreation Centers', 'Tall Buildings', 'Golf Courses','Water Activities', 'Water Conservancy Projects',
        'Other Sightseeing Tours', 'Spas'  ,'Restaurants & Bars'],
    'sad_dict' :['City Parks',  'Temples', 'Historical Architectures', 'Historical Sites', 'UNESCO World Heritage - Cultural Sites', 
        'Botanical Gardens', 'National Parks', ],
    'disappointed_dict' : ['Other Places of Worship', 'Churches and Cathedrals', 'Mountains', 'Gardens']}

    if emotion == "Excited":
        tags_to_filter=mapped_tags['excited_dict']
    elif emotion == "Joy":
        tags_to_filter=mapped_tags['joy_dict']
    elif emotion == "Neutral":
        tags_to_filter=mapped_tags['neutral_dict']
    elif emotion == "Sad":
        tags_to_filter=mapped_tags['sad_dict']
    elif emotion == "Disappointed":
        tags_to_filter=mapped_tags['disappointed_dict']

    # Use .isin() to select only the rows with the desired tags
    filtered_df = df[df['Tags'].isin(tags_to_filter)]
    # # print the filtered DataFrame
    # # print(filtered_df)

    random_rows = filtered_df.sample(n=5)
    # print(random_rows)



    questions = [
        "Describe {place} in a sentence?",
        "where is {place} located give answer in a sentence?",
        "Give brief information about {place} in a sentence?",
        "Describe the vibe of the {place} in a sentence?",
        "Describe the point of attraction of {place} in a sentence",
    ]

    locations= [location for location in random_rows['Location']]
    description=[desc for desc in random_rows['Wikipedia']]
    cities = [city for city in random_rows['City']]

    def create_question(locations):
        for location in locations:
            place=location
            questions = [
            "Describe {place} in a sentence?",
            "Give brief information about {place} in a sentence?",
            "Describe the vibe of the {place} in a sentence?",
            "Describe the point of attraction of {place} in a sentence",
            "What is near by {place}",
            "What is the speciality of the {place}",
            "What is the unique factor of {place}?"]
            for i in range(len(questions)):
                questions[i] = questions[i].format(place=place)
            return questions


    list_of_questions=[]
    for i in range(len(locations)):
        place = locations[i]
        questions = [
        "Describe {place} in a sentence?",
        "Give brief information about {place} in a sentence?",
        "Describe the vibe of the {place} in a sentence?",
        "Describe the point of attraction of {place} in a sentence",
        "What is near by {place}",
        "What is the speciality of the {place}",
        "What is the unique factor of {place}?"]
        for ques in range(len(questions)):
            questions[ques] = questions[ques].format(place=place)
        list_of_questions.append(questions)

    # print(list_of_questions)

    model = BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

    context=description[0]
    questions=list_of_questions[0]
    # print(context)
    # print(questions)
    # for j in range (len(questions)):
    #     # print(tokenizer.encode(questions[j], truncation = True, padding=True))


    nlp = pipeline('question-answering', model, tokenizer = tokenizer)
    def answer_func(i,k):
        a = nlp({
            'question': list_of_questions[i][k],
            'context':description[i]
        })
        return a['answer']

    list_of_answers = []
    for i in range(len(locations)):
        # Create an empty list for answers for this location
        location_answers = []
        
        for k in range(len(questions)):
            # Generate the answer for this location and question
            answer = answer_func(i, k)
            
            # Append the answer to the list of answers for this location
            location_answers.append(answer)
        
        # Append the list of answers for this location to the overall list of places
        list_of_answers.append(location_answers)

    # for response in range(len(locations)):
    #     # print(f"LOCATION: {locations[response]}")
    #     description_of_loc = '. '.join(list(set(list_of_answers[response])))
    #     description_of_loc = description_of_loc.capitalize()
    #     for d in range(len(description_of_loc)):
    #         if description_of_loc[d] == '.' and d+2 < len(description_of_loc):
    #             description_of_loc = description_of_loc[:d+2] + description_of_loc[d+2].upper() + description_of_loc[d+3:]
    #     # print(f"{description_of_loc}")
    #     # print(f"CITY: {cities[response]}")

    output_list = []
    for response in range(len(locations)):
        inner_list = []
        inner_list.append(f"LOCATION: {locations[response]}")
        description_of_loc = '. '.join(list(set(list_of_answers[response])))
        description_of_loc = description_of_loc.capitalize()
        for d in range(len(description_of_loc)):
            if description_of_loc[d] == '.' and d+2 < len(description_of_loc):
                description_of_loc = description_of_loc[:d+2] + description_of_loc[d+2].upper() + description_of_loc[d+3:]
        inner_list.append(description_of_loc)
        inner_list.append(f"CITY: {cities[response]}")
        output_list.append(inner_list)

    # # print(output_list)
    return output_list, emotion