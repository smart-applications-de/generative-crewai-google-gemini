import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tools
search_tool = SerperDevTool()

# --- AGENT DEFINITIONS ---

class MusicCreationAgents:
    """
    A class that encapsulates the definitions of all agents in our AI worship team.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    def theological_lyricist(self):
        return Agent(
            role='Theological Lyricist & Bible Scholar',
            goal='Analyze user-provided Bible verses and topics to extract core theological truths, emotions, and imagery to serve as the foundation for a worship song.',
            backstory=(
                "With a Master's in Divinity and a heart for worship, you bridge the gap between deep biblical study and heartfelt lyrical expression. "
                "You unpack scripture to find the raw, emotional, and poetic elements that can inspire a powerful song."
            ),
            llm=self.llm,
            tools=[search_tool],
            allow_delegation=False,
            verbose=True
        )

    def worship_songwriter(self):
        return Agent(
            role='Worship Songwriter & Composer',
            goal='Craft compelling, structured song lyrics (verse, chorus, bridge) based on the theological concepts provided. The lyrics should be singable, relatable, and emotionally resonant.',
            backstory=(
                "You are a seasoned songwriter who has co-written with major worship movements. You understand the power of a simple, profound chorus and how to build a song's narrative arc. "
                "Your craft is in creating lyrics that are both poetic and accessible for congregational singing or personal devotion."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def music_arranger(self):
        return Agent(
            role='Music Arranger & Producer',
            goal='Define the musical arrangement and atmosphere for a song based on the chosen genre. Specify instrumentation, tempo, mood, and dynamics to create a production-ready blueprint.',
            backstory=(
                "As a producer who has worked in studios from Nashville to Sydney, you know how to create a sonic landscape. Whether it\'s the driving rhythm of African Gospel, the atmospheric pads of Bethel, or the anthemic rock of Elevation, "
                "you can define the precise musical elements needed to bring a genre to life."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def lyria_prompt_technician(self):
        return Agent(
            role='Lyria Prompt Technician',
            goal='Synthesize the lyrics and musical arrangement into a single, comprehensive, and technically sound prompt for Google\'s Lyria music generation model.',
            backstory=(
                "You are a specialist in generative AI for music. You understand the specific syntax and descriptive keywords that Lyria needs to generate high-quality music. "
                "You translate the creative vision of the team into a precise set of instructions that the AI can understand and execute."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

# --- TASK DEFINITIONS ---

class MusicCreationTasks:
    """
    Defines the tasks for the music creation crew.
    """
    def lyrical_concept_task(self, agent, verses, topic):
        return Task(
            description=f"""
                Analyze the following Bible verses and topics to create a Lyrical Concept Brief.
                - Verses/Quotes: "{verses}"
                - Core Topic: "{topic}"

                Your brief must identify:
                1.  **Core Message:** The central truth or declaration.
                2.  **Key Emotions:** The primary feelings to convey (e.g., awe, gratitude, hope, repentance).
                3.  **Visual Imagery:** Powerful metaphors or scenes from the text (e.g., "roaring lion," "calm waters," "a table in the wilderness").
            """,
            expected_output="A concise Lyrical Concept Brief with sections for Core Message, Key Emotions, and Visual Imagery.",
            agent=agent
        )

    def song_writing_task(self, agent, context):
        return Task(
            description="""
                Using the Lyrical Concept Brief, write a complete song.
                The song must have a clear structure, including at least:
                - Verse 1
                - Chorus
                - Verse 2
                - Chorus
                - Bridge
                - Chorus
                
                The lyrics should be heartfelt, creative, and easy to sing.
            """,
            expected_output="A complete song with clearly labeled sections (Verse 1, Chorus, etc.).",
            agent=agent,
            context=context
        )
    
    def arrangement_task(self, agent, genre, topic):
        return Task(
            description=f"""
                Create a detailed Musical Arrangement Guide for a new song.
                - Genre: "{genre}"
                - Topic: "{topic}"

                Your guide must specify:
                1.  **Tempo:** Describe it (e.g., "Slow and contemplative, around 68 BPM," "Uptempo and driving, around 125 BPM").
                2.  **Mood & Dynamics:** Describe the emotional arc (e.g., "Starts sparse and intimate, builds to an anthemic, powerful chorus, drops to a reflective bridge").
                3.  **Instrumentation:** List the key instruments based on the genre. 
                    - For 'Worship': Think atmospheric pads, delayed electric guitars, grand piano, solid bass, powerful drums.
                    - For 'Praise': Think rhythmic acoustic guitar, punchy synths, clean electric guitars, driving bass and drums.
                    - For 'African Gospel': Think prominent basslines, complex polyrhythmic percussion (djembe, congas), choir vocals, bright keys/organ, and clean electric guitar lines.
            """,
            expected_output="A detailed Musical Arrangement Guide with sections for Tempo, Mood/Dynamics, and Instrumentation.",
            agent=agent
        )

    def prompt_generation_task(self, agent, context):
        return Task(
            description="""
                Combine the final lyrics and the Musical Arrangement Guide into one single, detailed prompt for Google's Lyria model.
                The prompt should be structured as a clear instruction set.

                Start by describing the overall musical feel, referencing the genre, mood, tempo, and key instruments.
                Then, integrate the lyrics, perhaps suggesting the musical feel for each section (e.g., "The verse should be sparse with just piano and vocals...").
                
                This final prompt is the ultimate handover to the AI musician. Make it as clear and descriptive as possible.
            """,
            expected_output="A single, comprehensive text prompt, formatted and optimized for use with Google's Lyria generative music model.",
            agent=agent,
            context=context,
            output_file='final_lyria_prompt.txt'
        )

# --- CREW SETUP ---

def create_music_crew(genre, verses, topic):
    """
    Factory function to create and configure the music creation crew.
    """
    agents = MusicCreationAgents()
    tasks = MusicCreationTasks()

    # Instantiate Agents
    lyricist = agents.theological_lyricist()
    songwriter = agents.worship_songwriter()
    arranger = agents.music_arranger()
    prompt_technician = agents.lyria_prompt_technician()
    
    # Define Tasks
    # These two can run in parallel
    task1 = tasks.lyrical_concept_task(lyricist, verses, topic)
    task3 = tasks.arrangement_task(arranger, genre, topic)
    
    # This depends on the lyrical concept
    task2 = tasks.song_writing_task(songwriter, [task1])

    # The final task depends on the lyrics and the arrangement
    task4 = tasks.prompt_generation_task(prompt_technician, [task2, task3])

    # Assemble the Crew
    crew = Crew(
        agents=[lyricist, songwriter, arranger, prompt_technician],
        tasks=[task1, task3, task2, task4], # Note the order for parallel execution
        process=Process.sequential, # Still sequential overall, but tasks with shared context are grouped.
        verbose=2
    )
    return crew
import streamlit as st
from music_crew import create_music_crew

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Worship Song Studio",
    page_icon="🎶",
    layout="wide",
)

# --- Sidebar for Credentials ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("""
    Enter your credentials here. For this demo, the agents run on OpenAI, but a full Lyria integration would require Google Cloud credentials.
    """)
    st.text_input("Google Cloud Project ID", key="project_id", help="Required for future Lyria integration.")
    st.text_input("Gemini/Google AI API Key", key="gemini_key", type="password", help="Required for future Lyria integration.")
    st.info("Currently, the crew's 'thinking' is powered by OpenAI. Ensure your `.env` file has an `OPENAI_API_KEY`.")


# --- Header and Introduction ---
st.title("🎶 AI Worship Song Studio")
st.markdown("""
Welcome, Songwriter! Let's partner with the Holy Spirit and technology to create something beautiful.
Provide a theme or scripture, choose a genre, and our AI Worship Team will compose a song concept and generate a production-ready prompt for **Google's Lyria AI**.
""")


# --- User Input Section ---
st.header("Step 1: Share Your Inspiration")

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox(
        "**Select the Musical Genre:**",
        ("Worship (Hillsong/Bethel style)", "Praise (Elevation/Upbeat style)", "African Gospel Praise")
    )
    topic = st.text_input("**Core Topic or Theme:**", placeholder="e.g., God's Grace, Faithfulness, Salvation")

with col2:
    verses = st.text_area("**Enter Bible Verses or Inspirational Text:**", placeholder="e.g., John 3:16, Psalm 23", height=150)


# --- Crew Execution ---
st.header("Step 2: Compose the Song")

if st.button("Compose & Generate Lyria Prompt"):
    if not topic and not verses:
        st.error("🚨 Please provide a Topic or some Bible Verses to inspire the song.")
    else:
        with st.spinner("Your AI Worship Team is gathering... This may take a few minutes."):
            try:
                # Create and run the crew
                music_creation_crew = create_music_crew(genre, verses, topic)
                result = music_creation_crew.kickoff()

                st.success("Song concept and prompt created successfully!")
                
                st.subheader("✅ Your Final Lyria Prompt")
                st.info("Copy this prompt and use it with a tool that connects to Google's Lyria model to generate the music.", icon="📋")
                
                # Read the final prompt from the output file
                try:
                    with open('final_lyria_prompt.txt', 'r', encoding='utf-8') as file:
                        final_prompt = file.read()
                    st.code(final_prompt, language="text")
                except FileNotFoundError:
                    st.error("The prompt file was not found. Displaying raw result instead.")
                    st.write(result)
                
                with st.expander("👀 See the AI Team's Creative Process"):
                    st.markdown(result)

            except Exception as e:
                st.error(f"An error occurred during composition: {e}")
                st.error("Please check your OpenAI API key in the .env file.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by an AI Musician & Python Expert.")




import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize tools
search_tool = SerperDevTool()

# --- AGENT DEFINITIONS (Generalized Roles) ---

class MusicCreationAgents:
    """
    A class that encapsulates the definitions of all agents in our AI Music Collective.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    def lyrical_concept_developer(self):
        return Agent(
            role='Lyrical Concept Developer',
            goal='Analyze user-provided text and topics to extract core themes, emotions, and imagery to serve as the foundation for a song.',
            backstory=(
                "You are a master of expression, able to find the poetic heart of any idea. You unpack user input to find the raw, emotional, and narrative elements that can inspire a powerful song, no matter the genre."
            ),
            llm=self.llm,
            tools=[search_tool],
            allow_delegation=False,
            verbose=True
        )

    def genre_songwriter(self):
        return Agent(
            role='Genre-Versatile Songwriter',
            goal='Craft compelling, structured song lyrics (verse, chorus, bridge) based on the provided concepts, tailored to the chosen genre.',
            backstory=(
                "You are a chameleon-like songwriter from the halls of Berklee College of Music, having penned hits in every genre from Country to Hip-Hop to Pop. You understand the lyrical conventions, rhyme schemes, and storytelling styles unique to each genre."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def music_arranger(self):
        return Agent(
            role='Multi-Genre Music Arranger & Producer',
            goal='Define the musical arrangement and atmosphere for a song based on the chosen genre. Specify instrumentation, tempo, mood, and dynamics.',
            backstory=(
                "As a top-tier producer, you have a vast sonic vocabulary. Whether it\'s the gritty authenticity of the Blues, the 808-driven landscape of Hip-Hop, or the polished catchiness of Schlager, you can define the precise musical elements needed to bring any genre to life."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def lyria_prompt_technician(self):
        return Agent(
            role='Lyria Prompt Technician',
            goal='Synthesize the lyrics and musical arrangement into a single, comprehensive, and technically sound prompt for Google\'s Lyria music generation model.',
            backstory=(
                "You are a specialist in generative AI for music. You translate the creative vision of the team into a precise set of instructions that the AI can understand, ensuring the final output perfectly matches the intended genre and style."
            ),
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

# --- TASK DEFINITIONS (Updated Arrangement Task) ---

class MusicCreationTasks:
    """
    Defines the tasks for the music creation crew.
    """
    def lyrical_concept_task(self, agent, text_input, topic):
        return Task(
            description=f"""
                Analyze the following user input to create a Lyrical Concept Brief.
                - Inspirational Text/Keywords: "{text_input}"
                - Core Topic/Theme: "{topic}"

                Your brief must identify:
                1.  **Core Message:** The central idea or story.
                2.  **Key Emotions:** The primary feelings to convey (e.g., joy, heartbreak, confidence, nostalgia).
                3.  **Key Imagery:** Powerful metaphors or scenes.
            """,
            expected_output="A concise Lyrical Concept Brief with sections for Core Message, Key Emotions, and Key Imagery.",
            agent=agent
        )

    def song_writing_task(self, agent, context):
        return Task(
            description="""
                Using the Lyrical Concept Brief, write a complete song.
                The song must have a clear structure suitable for popular music, such as:
                - Verse 1
                - Chorus
                - Verse 2
                - Chorus
                - Bridge
                - Chorus
                
                The lyrics should be creative and fit the intended theme.
            """,
            expected_output="A complete song with clearly labeled sections (Verse 1, Chorus, etc.).",
            agent=agent,
            context=context
        )
    
    def arrangement_task(self, agent, genre, topic):
        # This is the core of the genre adaptation
        return Task(
            description=f"""
                Create a detailed Musical Arrangement Guide for a new song.
                - Genre: "{genre}"
                - Topic: "{topic}"

                **CRITICAL:** You must adhere strictly to the conventions of the specified genre.
                
                Your guide must specify:
                1.  **Tempo & Rhythm:** Describe it (e.g., "Slow, soulful 12/8 feel, around 60 BPM," or "Classic boom-bap Hip-Hop groove, 90 BPM," or "Driving 4/4 'foxtrot' rhythm, 128 BPM").
                2.  **Mood & Dynamics:** Describe the emotional arc of the song.
                3.  **Instrumentation:** List the key instruments that DEFINE the genre.

                **Genre-Specific Instructions:**
                - If 'Worship': Use atmospheric pads, delayed electric guitars, grand piano, solid bass, powerful drums.
                - If 'Praise': Use rhythmic acoustic guitar, punchy synths, clean electric guitars, driving bass and drums.
                - If 'African Gospel': Use prominent basslines, complex polyrhythmic percussion (djembe, congas), choir vocals, bright keys/organ.
                - If 'Blues': MUST mention a 12-bar blues structure. Use expressive, slightly overdriven electric guitar (like a Gibson ES-335), harmonica, upright bass, and a simple, shuffling drum beat.
                - If 'Hip-Hop': Specify the drum machine sound (like a TR-808). Mention a prominent bassline or a sampled melody. The focus should be on the beat and rhythm.
                - If 'German Schlager': Mention a strong, simple 4/4 beat (often a 'Discofox' rhythm). Use synthesizer brass, accordion, clean electric guitars, and often a melodic, memorable synth line. The mood is typically upbeat, positive, and danceable.
            """,
            expected_output="A detailed Musical Arrangement Guide with sections for Tempo/Rhythm, Mood/Dynamics, and genre-specific Instrumentation.",
            agent=agent
        )

    def prompt_generation_task(self, agent, context):
        return Task(
            description="""
                Combine the final lyrics and the Musical Arrangement Guide into one single, detailed prompt for Google's Lyria model.
                The prompt must be a clear, descriptive paragraph.

                Start by describing the overall musical feel, referencing the genre, mood, tempo, and key instruments from the arrangement guide.
                Then, integrate the lyrics, suggesting the musical feel for each section (e.g., "The verse is sparse with just a shuffling drum and bassline...").
                
                The final prompt must be a masterpiece of instruction, ensuring the AI captures the specific genre requested.
            """,
            expected_output="A single, comprehensive text prompt, formatted and optimized for use with Google's Lyria generative music model.",
            agent=agent,
            context=context,
            output_file='final_lyria_prompt.txt'
        )

# --- CREW SETUP ---

def create_music_crew(genre, text_input, topic):
    """
    Factory function to create and configure the music creation crew.
    """
    agents = MusicCreationAgents()
    tasks = MusicCreationTasks()

    # Instantiate Agents
    concept_dev = agents.lyrical_concept_developer()
    songwriter = agents.genre_songwriter()
    arranger = agents.music_arranger()
    prompt_technician = agents.lyria_prompt_technician()
    
    # Define Tasks
    task1 = tasks.lyrical_concept_task(concept_dev, text_input, topic)
    task3 = tasks.arrangement_task(arranger, genre, topic)
    task2 = tasks.song_writing_task(songwriter, [task1])
    task4 = tasks.prompt_generation_task(prompt_technician, [task2, task3])

    # Assemble the Crew
    crew = Crew(
        agents=[concept_dev, songwriter, arranger, prompt_technician],
        tasks=[task1, task3, task2, task4],
        process=Process.sequential,
        verbose=2
    )
    return crew
    
import streamlit as st
from music_crew import create_music_crew

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Music Creation Studio",
    page_icon="🎤",
    layout="wide",
)

# --- Sidebar for Credentials ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("""
    Enter your credentials here. For this demo, the agents run on OpenAI, but a full Lyria integration would require Google Cloud credentials.
    """)
    st.text_input("Google Cloud Project ID", key="project_id", help="Required for future Lyria integration.")
    st.text_input("Gemini/Google AI API Key", key="gemini_key", type="password", help="Required for future Lyria integration.")
    st.info("Currently, the crew's 'thinking' is powered by OpenAI. Ensure your `.env` file has an `OPENAI_API_KEY`.")

# --- Header and Introduction ---
st.title("🎤 AI Music Creation Studio")
st.markdown("""
Welcome, Producer! This is your all-genre music studio.
Provide a theme, choose a genre, and our AI Music Collective will write a song and generate a production-ready prompt for **Google's Lyria AI**.
""")

# --- User Input Section ---
st.header("Step 1: Share Your Vision")

col1, col2 = st.columns(2)
with col1:
    genre = st.selectbox(
        "**Select the Musical Genre:**",
        (
            "Worship (Hillsong/Bethel style)",
            "Praise (Elevation/Upbeat style)",
            "African Gospel Praise",
            "Blues",
            "Hip-Hop",
            "German Schlager",
            "Pop",
            "Country"
        )
    )
    topic = st.text_input("**Core Theme/Topic:**", placeholder="e.g., Heartbreak, A Road Trip, Overcoming Adversity")

with col2:
    text_input = st.text_area("**Enter Inspirational Text or Keywords:**", placeholder="e.g., 'Rainy nights, empty streets, faded photograph', 'cadillac, dusty highway, setting sun'", height=150)

# --- Crew Execution ---
st.header("Step 2: Start the Session")

if st.button("Compose & Generate Lyria Prompt"):
    if not topic and not text_input:
        st.error("🚨 Please provide a Topic or some Inspirational Text.")
    else:
        with st.spinner("Your AI Music Collective is warming up... This may take a few minutes."):
            try:
                # Create and run the crew
                music_creation_crew = create_music_crew(genre, text_input, topic)
                result = music_creation_crew.kickoff()

                st.success("Song concept and prompt created successfully!")
                
                st.subheader("✅ Your Final Lyria Prompt")
                st.info("Copy this prompt and use it with a tool that connects to Google's Lyria model to generate the music.", icon="📋")
                
                try:
                    with open('final_lyria_prompt.txt', 'r', encoding='utf-8') as file:
                        final_prompt = file.read()
                    st.code(final_prompt, language="text")
                except FileNotFoundError:
                    st.error("The prompt file was not found. Displaying raw result instead.")
                    st.write(result)
                
                with st.expander("👀 See the AI Team's Creative Process"):
                    st.markdown(result)

            except Exception as e:
                st.error(f"An error occurred during composition: {e}")
                st.error("Please check your OpenAI API key in the .env file.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by an AI Musician & Python Expert.")  