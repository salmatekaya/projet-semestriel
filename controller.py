from ast import keyword
from keywords_display import display_keyword
import re
import nltk.corpus
from myuse import myTextSegmenter
from bi_2class_wiki_use import myTextSegmenter_bi2w
from keybert import KeyBERT
import yake


def load_segmenter():
    segmenter = myTextSegmenter(r'C:\Users\dhia\OneDrive\Bureau\work\TextSegmentation\model\saved_weights\cnn_segmenter_tfhub_bidirectional_1.h5')
    return segmenter
def load_segmenterbi2w():
    segmenter = myTextSegmenter_bi2w(r'C:\Users\dhia\Downloads\segmenter_epoch_0004_loss_0.128.h5')
    return segmenter
def segmentbi2w(text):
    segmenter = load_segmenterbi2w()
    passages = segmenter.segmentbi2w(text)
    return passages
def segment(text):
    segmenter = load_segmenter()
    passages = segmenter.segment(text)
    return passages

def load_keyBERT():
    kw_model = KeyBERT('sentence-transformers/bert-base-nli-mean-tokens')
    return kw_model
def extract_with_keyBERT(kw_model,text,min_Ngrams, max_Ngrams,mmr,StopWords,top_N,Diversity):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(min_Ngrams, max_Ngrams),
        use_mmr=mmr,
        stop_words=StopWords,
        top_n=top_N,
        diversity=Diversity)
    display_keyword(text,keywords)

def extract_with_yake(text,max_Ngrams,deduplication_thresold,window_size,top_N):
    kw_model = yake.KeywordExtractor("en",max_Ngrams,deduplication_thresold,'seqm',window_size,top_N)
    keywords = kw_model.extract_keywords(text)
    display_keyword(text,keywords)

