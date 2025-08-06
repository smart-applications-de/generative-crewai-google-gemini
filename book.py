import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- AGENT DEFINITIONS (No changes here) ---

class BookWritingAgents:
    """
    A class to encapsulate the definitions of all agents involved in the book writing process.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)
        self.search_tool = SerperDevTool()
        self.scrape_tool = ScrapeWebsiteTool()
        self.file_tool = FileReadTool()

    def chief_outline_architect(self):
        return Agent(
            role='Chief Outline Architect',
            goal='Create a comprehensive, chapter-by-chapter outline for a ~300-page book based on the user\'s topic and chosen language. The outline must be detailed, logical, and compelling.',
            backstory='A seasoned developmental editor and bestselling author, you have a knack for structuring complex ideas into engaging book formats across multiple languages. You know what sells and how to create a narrative arc that captivates readers.',
            llm=self.llm,
            tools=[self.search_tool, self.scrape_tool],
            allow_delegation=False,
            verbose=True
        )

    def research_specialist(self):
        return Agent(
            role='Research Specialist',
            goal='Gather, verify, and compile detailed information, facts, anecdotes, and data for each point in the book outline. The research must be thorough and well-documented.',
            backstory='You are a meticulous multilingual researcher with a Ph.D. in library and information science. You can find a needle in a digital haystack and have access to vast databases, academic journals, and web resources in many languages.',
            llm=self.llm,
            tools=[self.search_tool, self.scrape_tool, self.file_tool],
            allow_delegation=False,
            verbose=True
        )

    def narrative_crafter(self):
        return Agent(
            role='Narrative Crafter',
            goal='Write engaging, well-structured chapters in the specified language, based on the provided outline and research. The tone should match the book\'s theme, and the prose must be vivid and clear.',
            backstory='A master storyteller and ghostwriter, you are fluent in several languages and have penned numerous books across different genres and markets. You can adapt your writing style to any topic, bringing ideas to life with native-level fluency.',
            llm=self.llm,
            tools=[self.search_tool],
            allow_delegation=True,
            verbose=True
        )

    def senior_editor(self):
        return Agent(
            role='Senior Editor',
            goal='Review, edit, and polish the drafted chapters to ensure stylistic consistency, grammatical correctness, and overall narrative coherence in the specified language. Your final output should be a publish-ready manuscript.',
            backstory='With a red pen sharpened by years at top publishing houses in New York, London, and Berlin, you are the final gatekeeper of quality. You are a polyglot with an eagle eye for typos and a deep understanding of prose rhythm in multiple languages.',
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )


# --- TASK DEFINITIONS (Updated to accept 'language') ---

class BookWritingTasks:
    """
    A class to define the tasks for the book writing crew, now with language support.
    """
    def create_outline_task(self, agent, topic, user_prompt, language):
        return Task(
            description=f"""
                Analyze the user's book idea based on the topic: '{topic}' and the specific prompt: '{user_prompt}'.
                Develop a comprehensive, chapter-by-chapter outline for a book of approximately 300 pages.
                
                **CRITICAL REQUIREMENT: The entire output for this task (titles, synopsis, chapter descriptions) MUST be written in {language}.**

                The outline should include:
                1. A compelling book title in {language}.
                2. A brief synopsis of the book in {language}.
                3. A breakdown of Parts or Sections.
                4. A detailed list of chapters within each part, with a short paragraph in {language} describing the content of each chapter.

                Your final output is this detailed outline, written entirely in {language}.
            """,
            expected_output=f"A detailed, multi-level book outline with all content written strictly in {language}.",
            agent=agent,
        )
        
    def research_task(self, agent, context, language):
        return Task(
            description=f"""
                Take the detailed book outline and conduct thorough research for each chapter.
                While you can research from sources in any language, your compiled notes must be organized and presented in a way that is easily understandable for a writer whose target language is {language}.
                
                For each chapter, gather relevant:
                - Factual data and statistics.
                - Historical context.
                - Supporting anecdotes or case studies.
                - Key quotes (if quoting from another language, provide a faithful translation into {language}).
                
                Compile this research into a structured document, clearly organized by chapter.
            """,
            expected_output=f"A well-organized research document, tailored for a writer working in {language}.",
            agent=agent,
            context=context
        )

    def writing_task(self, agent, context, language):
        return Task(
            description=f"""
                Using the book outline and the comprehensive research document, write the full content for the book.
                
                **CRITICAL REQUIREMENT: You must write the entire chapter content strictly in {language}. Do not use any other language for the final prose.**
                
                Your writing should be:
                - Engaging and appropriate for an audience that reads {language}.
                - Consistent with the tone defined by the topic.
                - Weaving the research material seamlessly into the narrative.
                
                To manage this task, output the content for the *first three chapters* as a high-quality sample. This will serve as the style guide for the rest of the book.
            """,
            expected_output=f"The complete, well-written text for the first three chapters of the book, written entirely in {language}.",
            agent=agent,
            context=context
        )
        
    def editing_task(self, agent, context, language):
        return Task(
            description=f"""
                Take the drafted chapters and perform a comprehensive edit.
                
                **CRITICAL REQUIREMENT: Your review and all final edits must be performed in {language}. Your goal is to make the text sound like it was originally written by a native speaker of {language}.**
                
                Your editing process should cover:
                1.  Clarity, idiomatic expressions, and cultural nuances for a {language}-speaking audience.
                2.  Tone and Style consistency in {language}.
                3.  Grammar and Spelling specific to {language}.
                4.  Enhancing the prose to be powerful and engaging for the target reader.

                Provide the final, polished version of the chapters.
            """,
            expected_output=f"The final, edited, and polished text for the written chapters, ready for publication in {language}.",
            agent=agent,
            context=context,
            output_file=f'book_final_output_{language.lower()}.md' # Dynamic filename
        )

# --- CREW SETUP (Updated to accept and pass 'language') ---

def create_book_crew(topic, user_prompt, language):
    """
    Factory function to create and configure the book writing crew with language selection.
    """
    agents = BookWritingAgents()
    tasks = BookWritingTasks()

    # Instantiate Agents
    architect_agent = agents.chief_outline_architect()
    researcher_agent = agents.research_specialist()
    writer_agent = agents.narrative_crafter()
    editor_agent = agents.senior_editor()

    # Instantiate Tasks with the selected language
    outline_task = tasks.create_outline_task(architect_agent, topic, user_prompt, language)
    research_task = tasks.research_task(researcher_agent, [outline_task], language)
    writing_task = tasks.writing_task(writer_agent, [research_task], language)
    editing_task = tasks.editing_task(editor_agent, [writing_task], language)
    
    # Assemble the Crew
    book_crew = Crew(
        agents=[architect_agent, researcher_agent, writer_agent, editor_agent],
        tasks=[outline_task, research_task, writing_task, editing_task],
        process=Process.sequential,
        verbose=2,
        memory=True
    )

    return book_crew
import streamlit as st
from book_crew import create_book_crew

# --- Page Configuration ---
st.set_page_config(
    page_title="Multilingual AI Book Writing Crew",
    page_icon="📚",
    layout="wide",
)

# --- Header and Introduction ---
st.title("📚 Multilingual AI Book Writing Crew")
st.markdown("""
Welcome, author! This is your personal AI writer's room. 
Provide a topic, a detailed prompt, and **select your desired language**. Your AI crew will then generate a detailed outline and the first few chapters in that language.
""")

st.info("""
**How it works:**
1.  **You provide the vision:** A core topic, a detailed description, and a target language.
2.  **The Architect designs the blueprint:** An AI agent creates a detailed chapter-by-chapter outline in your chosen language.
3.  **The Researcher gathers the facts:** Another agent enriches the outline with data and details.
4.  **The Writer crafts the narrative:** A third agent writes the initial chapters with native-level fluency.
5.  **The Editor polishes the final product:** The final agent reviews and refines the text to publishing standards in the target language.
""", icon="🤖")


# --- User Input Fields ---
st.header("Step 1: Define Your Book's Vision")

# NEW: Language selection
language = st.selectbox(
    "**Select the language for your book:**",
    ("English", "German", "French", "Swahili")
)

topic = st.text_input(
    "**Enter the core topic or theme of your book:**",
    placeholder="e.g., My personal biography, The concept of Divine Grace, A history of quantum computing"
)

user_prompt = st.text_area(
    "**Provide a detailed description of your book idea:** (You can write this in English; the AI will understand and produce the book in your selected language)",
    height=250,
    placeholder="""
    Example for a biography:
    I want to write my life story. I was born in a small, poor village in Kenya. My family had very little, but we were happy. At 13, I became a devout Christian, which changed my life's direction. At 19, I got the opportunity to move to Germany as an au-pair. It was a huge culture shock, but I was determined. I managed to learn German, finish my studies, and get into university to study physics. After graduating, I transitioned into the tech world and became a successful Data Engineer. The book should focus on themes of faith, perseverance, and overcoming adversity.

    Example for a thematic book:
    I want to write a book about God's grace and forgiveness. It should be aimed at a modern Christian audience who may be struggling with these concepts. The book should explore what grace truly means, using biblical stories (like the Prodigal Son), theological insights (from writers like C.S. Lewis or Timothy Keller), and relatable modern-day anecdotes. It should be structured to help the reader move from understanding the concept to applying it in their own lives.
    """
)

# --- Crew Execution ---
st.header("Step 2: Assemble Your Crew and Start Writing")

if st.button(f"Start Writing My Book in {language}"):
    if not topic or not user_prompt:
        st.error("Please provide both a topic and a detailed description to proceed.")
    else:
        with st.spinner(f"Your AI crew is assembling to write in {language}... This may take several minutes."):
            try:
                # Create and run the crew with language parameter
                book_writing_crew = create_book_crew(topic, user_prompt, language)
                result = book_writing_crew.kickoff()

                st.success("Your AI crew has completed its task!")
                st.balloons()
                
                st.subheader("Final Book Output")
                
                # UPDATED: Read from the dynamic output file
                output_filename = f'book_final_output_{language.lower()}.md'
                try:
                    with open(output_filename, 'r', encoding='utf-8') as file:
                        final_output = file.read()
                    st.markdown(final_output)
                except FileNotFoundError:
                    st.error(f"The final output file ('{output_filename}') was not found. Displaying raw result instead.")
                    st.write(result)

            except Exception as e:
                st.error(f"An error occurred while running the AI crew: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by an AI Author & Python Expert.")