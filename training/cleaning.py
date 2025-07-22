# Cleaning the dataset 
import re
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.utils.dediac import dediac_ar
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.utils.charmap import CharMapper

arclean = CharMapper.builtin_mapper("arclean")

def clean_broken_arabic_words(text):
    """ Clean broken Arabic words and remove tatweel characters."""
    
    def clean_line(line, arclean):
        return simple_word_tokenize(arclean(dediac_ar(normalize_unicode(line.strip()))))
    
    # Step 1: Remove broken words like الشــــــمس <- الشــــ ــمس
    text = text.replace(' ـ', '')
    text = text.replace('ـ ', '')

    # use camel tools to clean the text, it removes tatweel characters and diacritics
    text = " ".join(clean_line(text, arclean))
    return text

def clean_sentence(sentence):
    sentence = clean_broken_arabic_words(sentence)

    # If sentence ends with arabic comma '،', remove it because it mess up with dependency parsing
    if sentence.endswith('،'):
        sentence = sentence[:-1].strip()

    # Remove extra spaces
    sentence = re.sub(r'\s+', ' ', sentence).strip()
        
    # Join with space to reconstruct a clean sentence
    return sentence

def clean_example(example):
    example["Sentence"] = clean_sentence(example["Sentence"])
    return example


# Adding space after punctuation if not exists in the sentence, because we did that while dependency parsing
# the sentences, in order to make sure the dependency parsing works correctly
# Because in arabic, after comma, semicolon, question mark, etc. there should be a space
def add_space_after_punctuation_if_not(example):
    sentence = example["Sentence"]
    map_english_punctuation = {
        ";": ["؛"],  # Arabic semicolon
        ",": ["،", "٫"],  # Arabic comma
        "?": ["؟"],  # Arabic question mark
        "%": ["٪"],  # Arabic percentage sign
        "*": ["۝"],  # Arabic symbol for verse end
    }
    
    # count all the punctuations found in sentence
    punctuations_occurences = []
    for i, letter in enumerate(sentence):
        if letter in map_english_punctuation.keys():
            punctuations_occurences.append(i)

    # Add space after the arabic punctuation marks if its directly attached to the next word
    offset = 0
    for index in punctuations_occurences:
        adjusted_index = index + offset
        if adjusted_index < len(sentence) - 1 and sentence[adjusted_index + 1] != ' ':
            sentence = sentence[:adjusted_index + 1] + ' ' + sentence[adjusted_index + 1:]
            offset += 1  # Adjust offset due to inserted space

    example["Sentence_space_after_punct"] = sentence
    return example