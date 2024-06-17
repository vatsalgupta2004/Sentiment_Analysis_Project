# download all corpuses
# import nltk
# nltk.download("all")

# importing all required modules and functions 
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords, gutenberg, movie_reviews, reuters, opinion_lexicon
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re, string, random
from prettytable import PrettyTable
from project_code_logos import logo2
import timeit, datetime
import matplotlib.pyplot as plt 

# main source-code
def try_again():
    global content_try_store
    # initialising positive and negative cleansed token lists
    positive_cleansed_tokens_list = []
    negative_cleansed_tokens_list = []
    # removing noise from data
    def remove_noise(data_tokens, stop_words = ()):
        cleansed_tokens = []

        for token, tag in pos_tag(data_tokens):
            token = re.sub("(@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|https?://\S+)","", token)
            # @[A-Za-z0-9_] removes @
            # #[A-Za-z0-9_] removes #
            # https?://\S+ removes links
            # dictionary sorter using if-else for quick efficiency
            if tag.startswith("NN"): # nouns
                pos = 'n'
            elif tag.startswith('VB'): # verbs
                pos = 'v'
            elif tag.startswith('JJ'): # adjectives
                pos = 'a'
            elif tag.startswith('RB'): # adverbs
                pos = 'r'
            else: # if no match then adjectives default
                pos = 'a'

            # used to lematize or we could say to reduce the word to its base form (eg. running to run)
            lemmatizer = WordNetLemmatizer()
            token = lemmatizer.lemmatize(token, pos)

            # if token not empty, not have punctuation, is lowercase and should not have stop words
            if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
                cleansed_tokens.append(token.lower())
        return cleansed_tokens

    # extract words using generator function from the parameterised list for frequency distribution in this model list type [[x,y,z],[a,b,c],[d,e,f],...]
    def get_all_words(cleansed_tokens_list):
        for tokens in cleansed_tokens_list:
            for token in tokens:
                yield token

    # defining a generator function which instead of producing all dictionary at a single time it produces one dictionary at a time and saves data and resumes it from there only, it is more efficient in large datasets
    def get_data_for_model(cleansed_tokens_list):
        for data_tokens in cleansed_tokens_list:
            yield dict([token, True] for token in data_tokens)

    #showing the bar plot of size of each content seperated or frequency
    def matplotsize(positive_dataset,negative_dataset):
        barsentisize={"Positive":len(positive_dataset),"Neagtive":len(negative_dataset)}
        barsenti=list(barsentisize.keys())
        barsize=list(barsentisize.values())
        plt.figure(figsize=(5,5))
        for index, value in enumerate(barsize):
            plt.text(index,(value+125),str(value),ha="center")
        plt.bar(barsenti,barsize,color="aqua",width=.9,align="center",edgecolor="navy")
        plt.xlabel("Sentiments")
        plt.ylabel("Size of Dataset of each Sentiment")
        plt.title("Sentiment Distribution Within the Dataset")
        plt.show()

    # defining Twitter_samples corpus and using appropiate rules and functions to extract desired data from it
    def preprocess_twitter_X():
        stop_words = stopwords.words('english')
        positive_cleansed_tokens_list = []
        negative_cleansed_tokens_list = []

        positive_data_tokens = twitter_samples.tokenized('positive_tweets.json')
        negative_data_tokens = twitter_samples.tokenized('negative_tweets.json')

        for tokens in positive_data_tokens:
            positive_cleansed_tokens_list.append(remove_noise(tokens, stop_words))

        for tokens in negative_data_tokens:
            negative_cleansed_tokens_list.append(remove_noise(tokens, stop_words))

        return positive_cleansed_tokens_list, negative_cleansed_tokens_list

    # defining gutenberg corpus and using appropiate rules and functions to extract desired data from it
    def preprocess_gutenberg():
        stop_words = stopwords.words('english')
        positive_gutenberg_list = []
        negative_gutenberg_list = []
        
        positive_gutenberg_text = gutenberg.raw('austen-emma.txt')
        negative_gutenberg_text = gutenberg.raw('melville-moby_dick.txt')
        
        positive_sentences = sent_tokenize(positive_gutenberg_text)
        negative_sentences = sent_tokenize(negative_gutenberg_text)

        for sentence_st in positive_sentences:
            tokens = word_tokenize(sentence_st)
            positive_gutenberg_list.append(remove_noise(tokens, stop_words))

        for sentence_st in negative_sentences:
            tokens = word_tokenize(sentence_st)
            negative_gutenberg_list.append(remove_noise(tokens, stop_words))

        return positive_gutenberg_list, negative_gutenberg_list
    
    # defining movie_reviews corpus and using appropiate rules and functions to extract desired data from it
    def preprocess_movie_reviews():
        stop_words = stopwords.words('english')
        positive_reviews_data = []
        negative_reviews_data = []

        for fileid in movie_reviews.fileids('pos'):
            tokens = word_tokenize(movie_reviews.raw(fileid))
            positive_reviews_data.append(remove_noise(tokens, stop_words))

        for fileid in movie_reviews.fileids('neg'):
            tokens = word_tokenize(movie_reviews.raw(fileid))
            negative_reviews_data.append(remove_noise(tokens, stop_words))

        return positive_reviews_data, negative_reviews_data
    
    # defining reuters corpus and using appropiate rules and functions to extract desired data from it
    def preprocess_reuters_reviews():
        stop_words = stopwords.words('english')
        positive_reuters_data = []
        negative_reuters_data = []

        positive_reuters_category =["earn","acq","crude"]
        negative_reuters_category =["crude","trade","interest"]

        for category in positive_reuters_category:
            for fileid in reuters.fileids(category):
                tokens = word_tokenize(reuters.raw(fileid))
                positive_reuters_data.append(remove_noise(tokens, stop_words))

        for category in negative_reuters_category:
            for fileid in reuters.fileids(category):
                tokens = word_tokenize(reuters.raw(fileid))
                negative_reuters_data.append(remove_noise(tokens, stop_words))

        return positive_reuters_data, negative_reuters_data
    
    # defining opinion_lexicon corpus and using appropiate rules and functions to extract desired data from it
    def preprocess_opinion_lexicon():
        stop_words = stopwords.words('english')
        positive_opinion_lexicon_data = []
        negative_opinion_lexicon_data = []

        positive_words = opinion_lexicon.positive()
        negative_words = opinion_lexicon.negative()

        positive_text = " ".join(positive_words)
        negative_text = " ".join(negative_words)

        positive_sentences = sent_tokenize(positive_text)
        negative_sentences = sent_tokenize(negative_text)

        for sentence in positive_sentences:
            tokens = word_tokenize(sentence)
            positive_opinion_lexicon_data.append(remove_noise(tokens, stop_words))

        for sentence in negative_sentences:
            tokens = word_tokenize(sentence)
            negative_opinion_lexicon_data.append(remove_noise(tokens, stop_words))

        return positive_opinion_lexicon_data, negative_opinion_lexicon_data

    # defining and extracting lists from each corpus function and appending it to a common list
    if (True): # __name__ == "__main__"
        
        # return extraction from each corpus function
        
        # twitter_samples
        twitter_positive_cleansed_tokens, twitter_negative_cleansed_tokens = preprocess_twitter_X()
        # gutenberg
        gutenberg_positive_cleansed_tokens, gutenberg_negative_cleansed_tokens = preprocess_gutenberg()
        # movie_reviews
        movie_positive_cleansed_tokens, movie_negative_cleansed_tokens = preprocess_movie_reviews()
        # reuters
        reuters_positive_cleansed_tokens, reuters_negative_cleansed_tokens = preprocess_reuters_reviews()
        # opinion_lexicon
        opinion_lexicon_positive_cleansed_tokens, opinion_lexicon_negative_cleansed_tokens = preprocess_opinion_lexicon()

        # extending individual corpus lists to an common list for dataset creation 
        
        # twitter_samples
        positive_cleansed_tokens_list.extend(twitter_positive_cleansed_tokens)
        negative_cleansed_tokens_list.extend(twitter_negative_cleansed_tokens)
        # gutenberg
        positive_cleansed_tokens_list.extend(gutenberg_positive_cleansed_tokens)
        negative_cleansed_tokens_list.extend(gutenberg_negative_cleansed_tokens)
        # movie_reviews
        positive_cleansed_tokens_list.extend(movie_positive_cleansed_tokens)
        negative_cleansed_tokens_list.extend(movie_negative_cleansed_tokens)
        # reuters
        positive_cleansed_tokens_list.extend(reuters_positive_cleansed_tokens)
        negative_cleansed_tokens_list.extend(reuters_negative_cleansed_tokens)
        # opinion_lexicon
        positive_cleansed_tokens_list.extend(opinion_lexicon_positive_cleansed_tokens)
        negative_cleansed_tokens_list.extend(opinion_lexicon_negative_cleansed_tokens)

        # to create a frequency distribution of most frequently occurring tokens or text or words 
        all_pos_words = get_all_words(positive_cleansed_tokens_list)
        all_neg_words = get_all_words(negative_cleansed_tokens_list)

        # positive token's frequecy
        # freq_dist_pos = FreqDist(all_pos_words)
        ptext = nltk.Text(all_pos_words)
        freq_dist_pos = ptext.vocab()
        print(f"\n{freq_dist_pos.most_common(30)}")

        print(" ")

        # negative token's frequecy
        # freq_dist_neg = FreqDist(all_neg_words)
        ntext = nltk.Text(all_neg_words)
        freq_dist_neg = ntext.vocab()
        print(f"{freq_dist_neg.most_common(30)}")

        # positive_tokens_for_model is assumed to be an iterable containing dictionaries where each dictionary represents a tokenized piece of text data, with each token as a key and True as its value
        positive_tokens_for_model = get_data_for_model(positive_cleansed_tokens_list)
        negative_tokens_for_model = get_data_for_model(negative_cleansed_tokens_list)

        # each dictionary data_dict, a tuple (data_dict, "Positive") is created. This tuple contains the dictionary itself (data_dict) and the label either postive and negative
        positive_dataset = [(data_dict, "Positive") for data_dict in positive_tokens_for_model]
        negative_dataset = [(data_dict, "Negative") for data_dict in negative_tokens_for_model]

        # combining both the sentiment_dataset
        dataset =(positive_dataset + negative_dataset)

        # plotting size of positive and negative dataset
        matplotsize(positive_dataset,negative_dataset)

        # used random modules shuffle function to jumble the dataset values with same size as initially it was
        random.shuffle(dataset)

        # slice the dataset for faster comparisons as dataset is now smaller
        # 80:20 ratio is  for optimal accuracy
        # 93:7 ratio is  for best accuracy
        #manipulating dataset and its size
        desired_size_data = len(dataset)
        dataset = dataset[0:desired_size_data]
        size_data = len(dataset)
        ratio_data = 0.93
        index_data = int(size_data*(ratio_data))
        train_data = dataset[0:index_data]
        test_data = dataset[index_data:size_data]

        # It is based on Bayes' theorem with the "naive" assumption of independence between features
        # classifier learns the underlying patterns in the train_data
        # classifier object returned which encapsulates the learned model 
        classifier = NaiveBayesClassifier.train(train_data)

        # size of dataset, accuracy, and some most informative features n number of words that occur most number of times
        print(" ")
        print(f"Size of dataset taken is {size_data}")
        print("Accuracy is:", classify.accuracy(classifier, test_data))
        (classifier.show_most_informative_features(30))

        # taking input from end_user how about how many texts to be entered and what those texts are 
        user_name=input("\nEnter your name -->")
        user_sentences=[]
        user_sentences_number = int(input("\nEnter number of desired sentences you want to check whether they are positive or negative: "))
        print(" ")
        for num in range (0,user_sentences_number,1):
            print("Enter sentence -->")
            sentence_input=str(input(f"{num+1}."))
            user_sentences.append(sentence_input)

        # current date time in text file
        with open(file="data_collection.txt",mode="a") as data:
                data.write(f"\nUserName -->{user_name}\n{datetime.datetime.now()}")

        # remove noise classify data and various other comparissions and final output with results
        num_file=0
        for sentence in user_sentences:
            temp_store=[]
            custom_tokens = remove_noise(word_tokenize(sentence))
            #probability distribution
            probab_dist=classifier.prob_classify(dict([token, True] for token in custom_tokens))
            senti_obj_prob=probab_dist.max()#whether positive or negative
            probability=round(probab_dist.prob(senti_obj_prob),2)
            # print to user sentence, sentiment, score of probablity
            print(f"\nSentence:--> {sentence}")
            print(f"Sentiment Type:--> {classifier.classify(dict([token, True] for token in custom_tokens))}")
            print(f"Sentiment Score or Probability:--> {probability}")
            with open(file="data_collection.txt",mode="a") as data:
                data.write(f"\n{num_file+1}) {sentence} --> {classifier.classify(dict([token, True] for token in custom_tokens))}")
                data.write(f"\nScore --> {probability} ")
            num_file+=1
            temp_store.append([sentence,classifier.classify(dict([token, True] for token in custom_tokens)),probability])
            content_try_store.extend(temp_store)

# printing the beginning of project logos etc
logo2='''

   _____            _   _                      _                           _                    
  / ____|          | | (_)                    | |        /\               | |                   
 | (___   ___ _ __ | |_ _ _ __ ___   ___ _ __ | |_      /  \   _ __   __ _| |_   _ _______ _ __ 
  \___ \ / _ \ '_ \| __| | '_ ` _ \ / _ \ '_ \| __|    / /\ \ | '_ \ / _` | | | | |_  / _ \ '__|
  ____) |  __/ | | | |_| | | | | | |  __/ | | | |_    / ____ \| | | | (_| | | |_| |/ /  __/ |   
 |_____/ \___|_| |_|\__|_|_| |_| |_|\___|_| |_|\__|  /_/    \_\_| |_|\__,_|_|\__, /___\___|_|   
                                                                              __/ |             
                                                                             |___/              
'''
print(logo2)
# data written into file 
# with open(file="sentiment_analysis_data.txt",mode="a") as data:
#                 data.write(f"Collection of data by end-users\n")
# initialissing values of various variables to store data
table = PrettyTable(["Sno","Sentence","Sentiment","Score"])
table_time = PrettyTable(["Execution_Number","Time_Taken (Minutes)"])
content_try_store=[]
run_timer=[]
# using while loop to increase user interactions and faster and easy accessibility
condition= True
while(condition):
    try_check=(input("\nDo you want to continue with the Sentiment Analyser program (yes/no): ").lower())
    if(try_check=="yes"):
        condition=True
        # using timer function from timeit to measure time for each run and placing function in between start and end
        start=timeit.default_timer()
        try_again()
        end=timeit.default_timer()
        Total_time_found=round(((end-start)/60),2)
        min_total_time=int(Total_time_found)
        sec_total_time=(round(float(Total_time_found % 1)*(0.6),2))
        Total_time=(min_total_time + sec_total_time)
        run_timer.append(Total_time)
    else:
        condition= False
        # table for details of sentences and their sentiments respectively
        get_sentiment=(input("Enter sentiments you want to print table of (positive,negative,all): ").lower())
        print("\nAll your searches and their results from today are displayed below -->")
        num=1

        for content in content_try_store:
            if((get_sentiment=="positive") and ((content[1].lower())=="positive")):
                table.add_row([num,content[0],content[1],content[2]])
            elif((get_sentiment=="negative") and ((content[1].lower())=="negative")):
                table.add_row([num,content[0],content[1],content[2]])
            elif(get_sentiment == "all"):
                table.add_row([num,content[0],content[1],content[2]])
            num+=1
        table.align='l'
        print(table)
        # initialsing for pie chart
        pie_val=[]
        pie_color=["green","red"]
        pie_sent=["positive","negative"]
        sumpos=0
        sumneg=0
        for content in content_try_store:
            if(content[1].lower()=="positive"):
                sumpos+=content[2]
            elif(content[1].lower()=="negative"):
                sumneg+=content[2]
        pie_val.append(sumpos)
        pie_val.append(sumneg)

        # table for time of execution of each run
        print("\nAll your runtimes for each of the executions from today are displayed below -->")
        num_t=1
        for time_t in run_timer:
            table_time.add_row([num_t,time_t])
            num_t+=1
        table_time.align='l'
        print(table_time)
        # plotting pie chart
        plt.figure(figsize=(5,5))
        plt.pie(pie_val,labels=pie_sent,autopct="%0.2f%%",colors=pie_color,radius=1,labeldistance=1.1,textprops={"fontsize": 10},startangle=-45)
        plt.title("Sentiments Distribution from all today's searches")
        plt.show()
        # resetting values of content_try_store, run_timer, pie_val, pie_sent
        pie_val=[]
        content_try_store=[]
        run_timer=[]
        pie_sent=[]
        sumpos=0
        sumneg=0
        # ending message and thankyou
        print("\nThank you for visiting hope you are satisfied with our Sentiment Analyser project")
        print("\t\t\t😎 ^_^ ^_^ ^_^ 😎\n")