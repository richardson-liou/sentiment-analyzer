import streamlit as st
import pandas as pd
from model import analyze

# Page title
st.title("Sentiment Analyzer")
st.write("Analyze the sentiment of text — positive, negative, or neutral.")

#Single text mode
st.header("Analyze a sentence")

user_input = st.text_area("Type something here:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        label, score = analyze(user_input)

        # color coding label
        if label == "POSITIVE":
            st.success(f"Sentiment: {label}")
        elif label == "NEGATIVE":
            st.error(f"Sentiment: {label}")
        else:
            st.info(f"Sentiment: {label}")

        # confidence bar
        st.write("Confidence:")
        st.progress(score)
        st.write(f"{score:.2%}")

st.divider()

#csv file mode
st.header("Analyze a CSV file")
st.write("Upload a CSV with a column named 'text' to analyze multiple sentences at once.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "text" not in df.columns:
        st.error("Your CSV needs a column named 'text'.")
    else:
        # analyze on every row
        df = df.dropna(subset=["text"])
        df["text"] = df["text"].astype(str)
        results = df["text"].apply(lambda x: analyze(x))
        df["label"] = results.apply(lambda x: x[0])
        df["score"] = results.apply(lambda x: x[1])

        # results table
        st.write("Results:")
        st.dataframe(df)

        # bar chart distribution
        st.write("Sentiment breakdown:")
        chart_data = df["label"].value_counts().reset_index()
        chart_data.columns = ["Sentiment", "Count"]
        st.bar_chart(chart_data.set_index("Sentiment"))