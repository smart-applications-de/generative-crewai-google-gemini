
    
import streamlit as st
from bible_study_crew import create_bible_study_crew
import markdown_pdf
from docx import Document
import base64

# Helper functions for exporting remain the same
def markdown_to_pdf(md_content):
    pdf = markdown_pdf.MarkdownPdf(body_font_size=12)
    pdf.add_section(md_content, toc=False)
    return pdf.get_buffer()

def markdown_to_docx(md_content):
    doc = Document()
    for line in md_content.split('\n'):
        if line.startswith('### '): doc.add_heading(line.replace('### ', ''), level=3)
        elif line.startswith('## '): doc.add_heading(line.replace('## ', ''), level=2)
        elif line.startswith('# '): doc.add_heading(line.replace('# ', ''), level=1)
        elif line.strip() == '---': doc.add_page_break()
        else: doc.add_paragraph(line)
    from io import BytesIO
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()

# Bible book translations remain the same
ENGLISH_BOOKS = ["Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy", "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel", "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra", "Nehemiah", "Esther", "Job", "Psalms", "Proverbs", "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations", "Ezekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk", "Zephaniah", "Haggai", "Zechariah", "Malachi", "Matthew", "Mark", "Luke", "John", "Acts", "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians", "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians", "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews", "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude", "Revelation"]
BIBLE_BOOKS_TRANSLATIONS = {
    "English": ENGLISH_BOOKS,
    "German": ["Genesis", "Exodus", "Levitikus", "Numeri", "Deuteronomium", "Josua", "Richter", "Ruth", "1. Samuel", "2. Samuel", "1. Könige", "2. Könige", "1. Chronik", "2. Chronik", "Esra", "Nehemia", "Esther", "Hiob", "Psalmen", "Sprüche", "Prediger", "Hohelied", "Jesaja", "Jeremia", "Klagelieder", "Hesekiel", "Daniel", "Hosea", "Joel", "Amos", "Obadja", "Jona", "Micha", "Nahum", "Habakuk", "Zefanja", "Haggai", "Sacharja", "Maleachi", "Matthäus", "Markus", "Lukas", "Johannes", "Apostelgeschichte", "Römer", "1. Korinther", "2. Korinther", "Galater", "Epheser", "Philipper", "Kolosser", "1. Thessalonicher", "2. Thessalonicher", "1. Timotheus", "2. Timotheus", "Titus", "Philemon", "Hebräer", "Jakobus", "1. Petrus", "2. Petrus", "1. Johannes", "2. Johannes", "3. Johannes", "Judas", "Offenbarung"],
    "French": ["Genèse", "Exode", "Lévitique", "Nombres", "Deutéronome", "Josué", "Juges", "Ruth", "1 Samuel", "2 Samuel", "1 Rois", "2 Rois", "1 Chroniques", "2 Chroniques", "Esdras", "Néhémie", "Esther", "Job", "Psaumes", "Proverbes", "Ecclésiaste", "Cantique des Cantiques", "Ésaïe", "Jérémie", "Lamentations", "Ézéchiel", "Daniel", "Osée", "Joël", "Amos", "Abdias", "Jonas", "Michée", "Nahum", "Habacuc", "Sophonie", "Aggée", "Zacharie", "Malachie", "Matthieu", "Marc", "Luc", "Jean", "Actes", "Romains", "1 Corinthiens", "2 Corinthiens", "Galates", "Éphésiens", "Philippiens", "Colossiens", "1 Thessaloniciens", "2 Thessaloniciens", "1 Timothée", "2 Timothée", "Tite", "Philémon", "Hébreux", "Jacques", "1 Pierre", "2 Pierre", "1 Jean", "2 Jean", "3 Jean", "Jude", "Apocalypse"],
    "Swahili": ["Mwanzo", "Kutoka", "Walawi", "Hesabu", "Kumbukumbu la Torati", "Yoshua", "Waamuzi", "Ruthu", "1 Samweli", "2 Samweli", "1 Wafalme", "2 Wafalme", "1 Mambo ya Nyakati", "2 Mambo ya Nyakati", "Ezra", "Nehemia", "Esta", "Ayubu", "Zaburi", "Methali", "Mhubiri", "Wimbo Ulio Bora", "Isaya", "Yeremia", "Maombolezo", "Ezekieli", "Danieli", "Hosea", "Yoeli", "Amosi", "Obadia", "Yona", "Mika", "Nahumu", "Habakuki", "Sefania", "Hagai", "Zekaria", "Malaki", "Mathayo", "Marko", "Luka", "Yohana", "Matendo", "Warumi", "1 Wakorintho", "2 Wakorintho", "Wagalatia", "Waefeso", "Wafilipi", "Wakolosai", "1 Wathesalonike", "2 Wathesalonike", "1 Timotheo", "2 Timotheo", "Tito", "Filemoni", "Waebrania", "Yakobo", "1 Petro", "2 Petro", "1 Yohana", "2 Yohana", "3 Yohana", "Yuda", "Ufunuo"]
}

# --- CORRECTED: List of specific Gemini Models ---
GEMINI_MODEL_LIST = [
    "gemini/gemini-2.5-pro-preview-03-25", "gemini/gemini-2.5-flash-preview-05-20",
    "gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash-lite-preview-06-17",
    "gemini/gemini-2.5-pro-preview-05-06", "gemini/gemini-2.5-pro-preview-06-05",
    "gemini/gemini-2.5-pro", "gemini/gemini-2.0-flash-exp", "gemini/gemini-2.0-flash",
    "gemini/gemini-2.0-flash-001", "gemini/gemini-2.0-flash-exp-image-generation",
    "gemini/gemini-2.0-flash-lite-001", "gemini/gemini-2.0-flash-lite",
    "gemini/gemini-2.0-flash-preview-image-generation", "gemini/gemini-2.0-flash-lite-preview-02-05",
    "gemini/gemini-2.0-flash-lite-preview", "gemini/gemini-2.0-pro-exp",
    "gemini/gemini-2.0-pro-exp-02-05", "gemini/gemini-exp-1206",
    "gemini/gemini-2.0-flash-thinking-exp-01-21", "gemini/gemini-2.0-flash-thinking-exp",
    "gemini/gemini-2.0-flash-thinking-exp-1219", "gemini/gemini-2.5-flash-preview-tts",
    "gemini/gemini-2.5-pro-preview-tts"
]

# Page Config
st.set_page_config(page_title="Multilingual AI Bible Study Generator", page_icon="🌍", layout="wide")

# Sidebar for API Keys and Model Selection
with st.sidebar:
    st.header("⚙️ API Configuration")
    st.markdown("Enter your API keys below to enable the AI crew.")
    st.session_state.gemini_key = st.text_input("Enter your Google Gemini API Key", type="password")
    st.session_state.serper_key = st.text_input("Enter your Serper API Key", type="password")
    
    st.header("🤖 Model Selection")
    # CORRECTED: Use the new specific list of models
    st.session_state.gemini_model = st.selectbox("Select Gemini Model", GEMINI_MODEL_LIST)

# App Header
st.title("📖 Multilingual AI Bible Study Generator")
st.markdown("Powered by Google Gemini and a team of AI Theologians, Pastors, and Historians.")

# User Input
st.header("1. Select Your Language and Book")
selected_language = st.selectbox("Choose your language:", list(BIBLE_BOOKS_TRANSLATIONS.keys()))
selected_book_translated = st.selectbox("Choose a book to study:", BIBLE_BOOKS_TRANSLATIONS[selected_language])

# Crew Execution
st.header("2. Generate Your Study Guide")
if st.button(f"Create Study Guide for {selected_book_translated}"):
    if not st.session_state.gemini_key or not st.session_state.serper_key:
        st.error("🚨 Please enter your Gemini and Serper API keys in the sidebar to continue.")
    else:
        if "study_guide_content" in st.session_state:
            del st.session_state["study_guide_content"]
        
        book_index = BIBLE_BOOKS_TRANSLATIONS[selected_language].index(selected_book_translated)
        english_book_name = ENGLISH_BOOKS[book_index]

        with st.spinner(f"Your AI Bible Study Team is preparing your guide for '{selected_book_translated}' in {selected_language}..."):
            try:
                bible_study_crew = create_bible_study_crew(
                    bible_book=english_book_name,
                    language=selected_language,
                    selected_model=st.session_state.gemini_model,
                    gemini_api_key=st.session_state.gemini_key,
                    serper_api_key=st.session_state.serper_key
                )
                result = bible_study_crew.kickoff()
                
                output_filename = f'final_study_guide_{selected_language.lower()}.md'
                with open(output_filename, 'r', encoding='utf-8') as file:
                    st.session_state["study_guide_content"] = file.read()
                
                st.success("Your study guide is ready!")
                st.balloons()
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Display and Export Section
if "study_guide_content" in st.session_state:
    st.header("3. Your Custom Study Guide")
    guide_content = st.session_state["study_guide_content"]
    st.markdown(guide_content)
    
    st.header("4. Export Your Guide")
    col1, col2, col3, col4, col5 = st.columns(5)
    filename_base = f"{selected_book_translated.replace(' ', '_')}_study_guide"
    
    with col1:
        st.download_button("⬇️ Download as PDF", markdown_to_pdf(guide_content), f"{filename_base}.pdf", "application/pdf")
    with col2:
        st.download_button("⬇️ Download as DOCX", markdown_to_docx(guide_content), f"{filename_base}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
    with col3:
        st.download_button("⬇️ Download as MD", guide_content.encode('utf-8'), f"{filename_base}.md", "text/markdown")
    with col4:
        st.download_button("⬇️ Download as HTML", guide_content.encode('utf-8'), f"{filename_base}.html", "text/html")
    with col5:
        st.download_button("⬇️ Download as TXT", guide_content.encode('utf-8'), f"{filename_base}.txt", "text/plain")