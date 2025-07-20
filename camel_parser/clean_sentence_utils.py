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
    
    # Step 1: Remove broken words
    text = text.replace(' ـ', '')
    text = text.replace('ـ ', '')

    # use camel tools to clean the text, it removes tatweel characters and diacritics
    text = " ".join(clean_line(text, arclean))
    return text

def clean_sentence(sentence):

    # # Arabic characters range: main + extended + some punctuation
    # arabic_pattern = re.compile(r'[\u0600-\u06FF]+')

    # # Extract all Arabic tokens
    # arabic_tokens = arabic_pattern.findall(sentence)

    # sentence = " ".join(arabic_tokens)
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