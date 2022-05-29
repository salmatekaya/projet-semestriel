import streamlit as st
from controller import extract_with_keyBERT, extract_with_yake, segment,segmentbi2w,load_keyBERT


st.set_page_config(
    page_title="text labelisation",
    page_icon="📝",
    layout="wide",
)

def _max_width_():
    max_width_str = f"max-width: 400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()

c30, c31, c32 = st.columns([1.5,2,3])

with c30:
    st.image("https://miro.medium.com/max/720/1*4bKUfitlpPGQ42P7PV7Plg.jpeg", width=300)

with c31:
    st.header("")    
    st.header("")  
    st.header("")  
    st.header("")    
    st.title("SLICE & LABEL")   

    
with st.form(key="my_form"):


    ce, c1, ce, c2, c3 = st.columns([0.07, 1, 0.07, 5, 0.07])
    with c1:
        ModelType = st.radio(
            "Choose your model",
            ["keyBERT (Default)", "yake"],
            0,
            help="you can choose between 2 models (yake or DistilBERT) to embed your text",
        )
        top_N = st.slider(
            "# of results",
            min_value=1,
            max_value=30,
            value=10,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        min_Ngrams = st.number_input(
            "Minimum Ngram",
            min_value=1,
            max_value=4,
            help="""The minimum value for the ngram range.
*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
            # help="Minimum value for the keyphrase_ngram_range. keyphrase_ngram_range sets the length of the resulting keywords/keyphrases. To extract keyphrases, simply set keyphrase_ngram_range to (1, # 2) or higher depending on the number of words you would like in the resulting keyphrases.",
        )

        max_Ngrams = st.number_input(
            "Maximum Ngram",
            value=2,
            min_value=1,
            max_value=4,
            help="""The maximum value for the keyphrase_ngram_range.
*Keyphrase_ngram_range* sets the length of the resulting keywords/keyphrases.
To extract keyphrases, simply set *keyphrase_ngram_range* to (1, 2) or higher depending on the number of words you would like in the resulting keyphrases.""",
        )

        StopWordsCheckbox = st.checkbox(
            "Remove stop words",
            help="Tick this box to remove stop words from the document (currently English only)",
        )

        use_MMR = st.checkbox(
            "Use MMR",
            value=True,
            help="You can use Maximal Margin Relevance (MMR) to diversify the results. It creates keywords/keyphrases based on cosine similarity. Try high/low 'Diversity' settings below for interesting variations.",
        )

        Diversity = st.slider(
            "Keyword diversity (MMR only)",
            value=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            help="""The higher the setting, the more diverse the keywords.
            
Note that the *Keyword diversity* slider only works if the *MMR* checkbox is ticked.
""",
        )
        deduplication_thresold = st.slider(
            "duplication threshhold (only for yake)",
            value=0.9,
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            help="You can choose the number of keywords/keyphrases to display. Between 1 and 30, default number is 10.",
        )
        window_size = st.number_input(
            "window size (only for yake)",
            value=1,
            min_value=1,
            max_value=3,
            step=1,
        )

    with c2:
        doc = st.text_area(
            "Paste your text below",
            height=750,
        )
        submit_button1 = st.form_submit_button(label="✨ semantic sagmentation & labelisation")
        submit_button2 = st.form_submit_button(label="✨ labelisation of the whole document")

    if use_MMR:
        mmr = True
    else:
        mmr = False

    if StopWordsCheckbox:
        StopWords = "english"
    else:
        StopWords = None

if not submit_button1 and not submit_button2:
    st.stop()
if submit_button1:
    #passages = segment(doc)
    passages = segmentbi2w(doc)
else:
    passages = [doc]
if ModelType == "keyBERT (Default)":
    kw_model = load_keyBERT()
    for passage in passages:
        extract_with_keyBERT(kw_model,passage,min_Ngrams, max_Ngrams,mmr,StopWords,top_N,Diversity)
else:
    for passage in passages:
        extract_with_yake(passage,max_Ngrams,deduplication_thresold,window_size,top_N)