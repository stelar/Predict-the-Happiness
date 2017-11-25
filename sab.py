import numpy
from sklearn import metrics
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
train = pd.read_csv("train.csv")
print train.head(2)
print train.shape
from nltk.corpus import stopwords
sw=stopwords.words('english')
print sw[0]
from nltk.stem.snowball import SnowballStemmer
stemmer=SnowballStemmer("english")
print stemmer.stem("VERY")
stemer=''
sample="The room was kind of clean but had a VERY strong smell of dogs. Generally below average but ok for a overnight stay if you're not too fussy. Would consider staying again if the price was right. Breakfast was free and just about better than nothing."

a=[]
samplearr=sample.split(" ")
print samplearr[1]
print '*****'
for fields in samplearr:
    if fields not in sw:
        a.append(fields)
    else:
        print stemmer.stem(fields)

print a
example1 = BeautifulSoup(sample)

# Print the raw review and then the output of get_text(), for
# comparison
print sample
print example1.get_text()
import re
# Use regular expressions to do a find-and-replace
letters_only = re.sub("[^a-zA-Z]",           # The pattern to search for
                      " ",                   # The pattern to replace it with
                      example1.get_text() )  # The text to search
print letters_only
lower_case = letters_only.lower()        # Convert to lower case
words = lower_case.split()
print words

words = [w for w in words if not w in stopwords.words("english")]
print words

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))
b="\"I stayed at the Crown Plaza April -- - April --, ----. The staff was friendly and attentive. The elevators are tiny (about -' by -'). The food in the restaurant was delicious but priced a little on the high side. Of course this is Washington DC. There is no pool and little for children to do. My room on the fifth floor had two comfortable beds and plenty of space for one person. The TV is a little small by todays standards with a limited number of channels. There was a small bit of mold in the bathtub area that could have been removed with a little bleach. It appeared the carpets were not vacummed every day. I reported a light bulb was burned out. It was never replaced. Ice machines are on the odd numbered floors, but the one on my floor did not work. I encountered some staff in the elevator one evening and I mentioned the ice machine to them. Severel hours later a maid appeared at my door with ice and two mints. I'm not sure how they knew what room I was in. That was a little unnerving! I would stay here again for business, but would not come here on vacation.\"";
j=review_to_words(b)
print j

num_reviews = train["Description"].size
clean_train_reviews=[]
for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )
    clean_train_reviews.append( review_to_words( train["Description"][i] ))

print "Creating the bag of words...\n"
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 5000)

print len(clean_train_reviews)
train_data_features = vectorizer.fit_transform(clean_train_reviews)
#print train_data_features.size()
# Numpy arrays are easy to work with, so convert the result to an
# array
#train_data_features = train_data_features.toarray()

from sklearn.preprocessing import LabelEncoder
number = LabelEncoder()
train['Is_Response']=number.fit_transform(train['Is_Response'].astype('str'))
train['Device_Used']=number.fit_transform(train['Device_Used'].astype('str'))
train['Browser_Used']=number.fit_transform(train['Browser_Used'].astype('str'))
y=train['Is_Response']
print train.head(3)
#del train['Is_Response']
del train['User_ID']
Colpo = pd.DataFrame( data={"Device_Used":train["Device_Used"], "Browser_Used":train['Browser_Used'],"Desc":train_data_features} )

print Colpo.head(10)
rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=5, min_weight_fraction_leaf=0.0,
            n_estimators=500, n_jobs=1, oob_score=True, random_state=42,
            verbose=0, warm_start=False)
rf.fit( Colpo, train['Is_Response'])
print  "saddsad"
from sklearn.ensemble import RandomForestClassifier
test = pd.read_csv("test.csv")

# Verify that there are 25,000 rows and 2 columns
print test.shape


# Create an empty list and append the clean reviews one by one
num_reviews = test["Description"].size
clean_test_reviews=[]
clean_train_reviews=[]
for i in xrange( 0, num_reviews ):
    # If the index is evenly divisible by 1000, print a message
    if( (i+1)%1000 == 0 ):
        print "Review %d of %d\n" % ( i+1, num_reviews )

    clean_test_reviews.append( review_to_words( train["Description"][i] ))
print len(clean_test_reviews)

print "asda"
# Get a bag of words for the test set, and convert to a numpy array
test_data_features = vectorizer.transform(clean_test_reviews)
Polpo = pd.DataFrame( data={"Device_Used":test["Device_Used"], "Browser_Used":test['Browser_Used'],"Desc":test_data_features} )

print "sadsad"
#test_data_features = test_data_features.toarray()
print "sad"
# Use the random forest to make sentiment label predictions

print "casd"
result = rf.predict(Polpo)
print "dfdfs"
# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"User_ID":test["User_ID"], "Is_Response":result} )

# Use pandas to write the comma-separated output file
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )
