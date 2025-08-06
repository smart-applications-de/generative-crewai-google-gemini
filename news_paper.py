import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize the tool for web searches
search_tool = SerperDevTool()

# --- AGENT DEFINITIONS ---

class NewsAgents:
    """
    A class to encapsulate the definitions of all agents in our AI newsroom.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

    def managing_editor(self):
        return Agent(
            role='Managing Editor',
            goal='Oversee the creation of a newspaper, ensuring all content is high-quality, relevant to the selected scope (Local, National, Global), and factually accurate.',
            backstory=(
                "With decades of experience at major news outlets like the BBC and Reuters, you are the final word. "
                "You have a sharp eye for a compelling story, a stickler for journalistic integrity, and the ability to orchestrate a team of reporters to produce a cohesive, engaging newspaper."
            ),
            llm=self.llm,
            allow_delegation=True,
            verbose=True
        )

    def news_wire_service(self):
        return Agent(
            role='News Wire Service',
            goal='Continuously scan the web for the latest, most significant news stories across all topics and regions. Provide a stream of raw, up-to-the-minute headlines and data.',
            backstory=(
                "You are the digital equivalent of the Associated Press—a tireless, 24/7 service that is the first to know about any breaking event. "
                "Your job is not to write articles, but to find and provide the initial, verifiable information that the reporting team will build upon."
            ),
            llm=self.llm,
            tools=[search_tool],
            allow_delegation=False,
            verbose=True
        )

    def specialist_reporter(self, topic, scope="Global"):
        return Agent(
            role=f'{topic.title()} Reporter',
            goal=f'Develop in-depth, accurate, and engaging news articles on the topic of {topic}, tailored for a {scope} audience.',
            backstory=(
                f"You are a seasoned journalist with a deep specialization in {topic}. You know the key players, the underlying trends, and how to frame a story to make it understandable and compelling for your target audience. "
                "You take raw data from the news wire and turn it into a polished, insightful article."
            ),
            llm=self.llm,
            tools=[search_tool],
            allow_delegation=False, # Reporters work independently on their assigned stories
            verbose=True
        )

# --- TASK DEFINITIONS ---

class NewsTasks:
    """
    A class to define the tasks for the newspaper creation crew.
    """
    def fetch_news_task(self, agent, scope, location=""):
        # The query changes based on the scope to get relevant results
        query_location = location if scope == "Local" or scope == "National" else "world"
        return Task(
            description=f"""
                Fetch the most recent and significant news stories for a {scope} newspaper focused on {query_location}.
                The current date is {datetime.now().strftime('%Y-%m-%d')}. Your information must be as up-to-date as possible.
                Cover a wide range of topics including general news, politics, business, technology, sports, and culture.
                Compile a list of key headlines, sources, and a brief summary for each major story you find.
                This compiled data will serve as the source material for the specialist reporters.
            """,
            expected_output="A structured list of current news stories, each with a headline, a URL source, and a one-sentence summary.",
            agent=agent
        )

    def reporting_task(self, agent, topic, scope, context):
        return Task(
            description=f"""
                Using the provided news wire data, identify the single most important story related to your beat: '{topic}'.
                Write a concise and compelling news article on this story, suitable for a {scope} newspaper.
                
                Your article MUST include:
                1.  A catchy but informative headline.
                2.  A byline with your role (e.g., "By the Financial Reporter").
                3.  A 2-3 paragraph body summarizing the key information (who, what, when, where, why).
                
                Ensure your writing style is objective, clear, and engaging. Base your article strictly on the information from the news wire context.
            """,
            expected_output="A well-formatted news article with a headline, byline, and a 2-3 paragraph body.",
            agent=agent,
            context=context
        )

    def editing_task(self, agent, context):
        return Task(
            description="""
                Review all the drafted articles from the specialist reporters.
                Assemble them into a single, cohesive newspaper format.
                The final output should be a single block of text, formatted in Markdown.
                
                The structure should be:
                - A main title for the newspaper.
                - Each article presented clearly under a section heading (e.g., "## Top Story", "## Business").
                
                Ensure there are no formatting errors and the entire newspaper flows logically.
            """,
            expected_output="A single, well-formatted Markdown document containing the complete newspaper with all its articles.",
            agent=agent,
            context=context,
            output_file='final_newspaper.md'
        )

# --- CREW SETUP ---

def create_newspaper_crew(scope, location, topics):
    """
    Factory function to create and configure the newspaper crew.
    """
    agents = NewsAgents()
    tasks = NewsTasks()

    # Instantiate Agents
    editor = agents.managing_editor()
    wire_service = agents.news_wire_service()
    
    # Create specialist reporters for selected topics
    reporters = [agents.specialist_reporter(topic, scope) for topic in topics]

    # Define Tasks
    # 1. Fetch all news
    fetch_task = tasks.fetch_news_task(wire_service, scope, location)
    
    # 2. Each reporter writes their article based on the fetched news
    reporting_tasks = [
        tasks.reporting_task(reporter, topic, scope, [fetch_task])
        for reporter, topic in zip(reporters, topics)
    ]
    
    # 3. The editor assembles the final newspaper
    editing_task = tasks.editing_task(editor, reporting_tasks)
    
    # Assemble the Crew
    crew = Crew(
        agents=[editor, wire_service] + reporters,
        tasks=[fetch_task] + reporting_tasks + [editing_task],
        process=Process.sequential,
        verbose=2
    )

    return crew
import streamlit as st
from newspaper_crew import create_newspaper_crew

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Newsroom HQ",
    page_icon="📰",
    layout="wide",
)

# --- Header and Introduction ---
st.title("📰 AI Newsroom Headquarters")
st.markdown("""
Welcome, Editor-in-Chief! From this dashboard, you can commission a complete, up-to-the-minute digital newspaper.
Make your selections below, and our expert AI journalist crew will get to work.
""")

# --- User Input Section ---
st.header("Step 1: Define Your Newspaper's Focus")

scope_options = ["Global", "National", "Local"]
scope = st.selectbox("**Select Newspaper Scope:**", scope_options)

location = ""
if scope == "Local":
    location = st.selectbox("Select City:", ["Berlin", "Hamburg", "Munich", "Cologne", "Frankfurt"])
elif scope == "National":
    location = st.text_input("Enter Country:", "Germany")

st.markdown("**Select the sections to include in your newspaper:**")
topic_options = ["Top Story", "Business & Stock Market", "Sports", "Technology", "Fashion & Trends"]
selected_topics = [topic for topic in topic_options if st.checkbox(topic, True)]


# --- Crew Execution ---
st.header("Step 2: Go to Print!")

if st.button("Assemble Today's Newspaper"):
    if not selected_topics:
        st.error("Please select at least one topic to include in the newspaper.")
    else:
        with st.spinner("Your AI Newsroom is on the story... This will take a few minutes."):
            try:
                # Create and run the crew
                newspaper_creation_crew = create_newspaper_crew(scope, location, selected_topics)
                result = newspaper_creation_crew.kickoff()

                st.success("Today's edition is ready!")
                st.balloons()
                
                st.subheader(f"The {location if location else scope} Times")
                
                # Read the final newspaper from the output file
                try:
                    with open('final_newspaper.md', 'r', encoding='utf-8') as file:
                        final_output = file.read()
                    st.markdown(final_output)
                except FileNotFoundError:
                    st.error("The final newspaper file was not found. Displaying raw result instead.")
                    st.write(result)

            except Exception as e:
                st.error(f"An error occurred while running the AI crew: {e}")
                st.error("Please check your API keys and network connection.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by an AI News Anchor & Python Expert.")