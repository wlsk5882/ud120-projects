#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string

def parseOutText(f):
    
	""" given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
	"""

	f.seek(0)  ### go back to beginning of file (annoying)
	all_text = f.read()

    ### split off metadata
	content = all_text.split("X-FileName:")
	words = ""
	if len(content) > 1:
		### remove punctuation
		text_string = content[1].translate(string.maketrans("", ""), string.punctuation)
		### project part 2: comment out the line below
		words = text_string
		
		"""
		Augment parseOutText() so that the string it returns has all the words stemmed
			using a SnowballStemmer (use the nltk package, some examples that 
			I found helpful can be found here: http://www.nltk.org/howto/stem.html ).
			Rerun parse_out_email_text.py, which will use your updated parseOutText() function
			--what's your output now?

			Hint: you'll need to break the string down into individual words, stem each word,
			then recombine all the words into one string.
		"""

			
		stemmer = SnowballStemmer("english")
		splited_words = words.split()

		stemmed_words = []
		for word in splited_words:
			stemmed_words.append(stemmer.stem(word))
		
		words = ' '.join(stemmed_words)

		### split the text string into individual words, stem each word,
		### and append the stemmed word to words (make sure there's a single
		### space between each stemmed word)





	return words

    

def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

