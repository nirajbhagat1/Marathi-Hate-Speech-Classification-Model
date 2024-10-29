# For text preprocessing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load Marathi stopwords from NLTK or define your own list of stopwords
marathi_stopwords = set(stopwords.words('data/marathi_stopwords.txt'))
