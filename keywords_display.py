import streamlit as st
import seaborn as sns
from pandas import DataFrame

def display_keyword(passage,keywords):
    st.header("")
    st.header(passage)

    df = (
        DataFrame(keywords, columns=["Keyword/Keyphrase", "Relevancy"])
        .sort_values(by="Relevancy", ascending=False)
        .reset_index(drop=True)
    )

    df.index += 1
    #Add styling
    cmGreen = sns.light_palette("blue", as_cmap=True)
    cmRed = sns.light_palette("red", as_cmap=True)
    df = df.style.background_gradient(
        cmap=cmGreen,
        subset=[
            "Relevancy",
        ],
    )

    c1, c2, c3 = st.columns([1, 3, 1])

    format_dictionary = {
        "Relevancy": "{:.1%}",
    }

    df = df.format(format_dictionary)

    with c2:
        st.table(df)