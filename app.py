import streamlit as st
import speech_recognition as sr
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk
from tempfile import NamedTemporaryFile

# Make sure NLTK sentence tokenizer is available
nltk.download("punkt", quiet=True)


# --- Helper functions ---

def transcribe_audio(file_path: str) -> str:
    """
    Transcribe a short .wav audio file to text using Google's Web Speech API.
    Returns the transcribed text (string) or an empty string on failure.
    """
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)

        # Use Google's free web API (requires internet)
        text = recognizer.recognize_google(audio_data, language="en-IN")  # or "en-US"
        return text

    except sr.UnknownValueError:
        return ""  # could not understand audio
    except sr.RequestError:
        return ""  # API error / no internet


def summarize_text(text: str, num_sentences: int = 2) -> str:
    """
    Summarize the input text into a shorter version with num_sentences sentences.
    Uses an LSA-based extractive summarizer from sumy.
    """
    if not text or not text.strip():
        return ""

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()

    summary_sentences = summarizer(parser.document, num_sentences)
    summary = " ".join(str(sentence) for sentence in summary_sentences)
    return summary


# --- Streamlit app ---

st.title("AI Voice Note Summarizer")

st.write(
    "Upload a short `.wav` voice note in English. "
    "The app will transcribe it using speech recognition and then summarize the transcript."
)

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

num_sentences = st.slider(
    "Number of sentences in the summary:",
    min_value=1,
    max_value=5,
    value=2,
    step=1,
)

if st.button("Transcribe & Summarize"):
    if uploaded_file is None:
        st.warning("Please upload a .wav file first.")
    else:
        # Save uploaded file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Transcribing audio..."):
            transcript = transcribe_audio(tmp_path)

        if not transcript:
            st.error(
                "Could not transcribe the audio. "
                "Please check the audio quality, language, and internet connection."
            )
        else:
            st.subheader("Full Transcript")
            st.write(transcript)

            with st.spinner("Generating summary..."):
                summary = summarize_text(transcript, num_sentences=num_sentences)

            st.subheader(f"Summary ({num_sentences} sentences)")
            if summary:
                st.write(summary)
            else:
                st.write(
                    "Summary could not be generated (possibly due to very short or unclear transcript)."
                )