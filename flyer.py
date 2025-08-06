import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# --- AGENT DEFINITIONS (New Agent Added) ---

class FlyerDesignAgents:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0.8)
        self.search_tool = SerperDevTool()

    def creative_brief_specialist(self):
        return Agent(
            role='Creative Brief Specialist',
            goal='Analyze user request to develop a clear creative brief defining target audience, desired emotion, and core message for a flyer.',
            backstory="With a background in marketing, you excel at distilling complex ideas into actionable creative briefs.",
            llm=self.llm, tools=[self.search_tool], allow_delegation=False, verbose=True
        )

    def visual_concept_developer(self):
        return Agent(
            role='Visual Concept Developer & Art Director',
            goal='Brainstorm and develop strong visual concepts based on a creative brief, defining color palettes, composition, and modern artistic styles.',
            backstory="As a seasoned Art Director, you have your finger on the pulse of modern aesthetics and can translate strategy into compelling visual language.",
            llm=self.llm, tools=[self.search_tool], allow_delegation=False, verbose=True
        )

    def imagen_prompt_crafter(self):
        return Agent(
            role='Google Imagen Prompt Engineer',
            goal='Craft a detailed, effective image generation prompt for Google\'s Imagen model that translates the creative concepts into a language the AI can execute beautifully.',
            backstory="A technical artist, you know precisely which keywords invoke cinematic lighting, hyperrealism, or specific graphic styles in generative models.",
            llm=self.llm, allow_delegation=False, verbose=True
        )
    
    # NEW AGENT
    def social_media_copywriter(self):
        return Agent(
            role='Social Media Copywriter',
            goal='Write a short, engaging, and effective social media post to accompany the flyer image. The copy should be tailored to the campaign\'s goals and encourage interaction.',
            backstory='A viral marketing specialist, you craft words that stop the scroll and drive engagement. You know how to use hashtags, ask questions, and create calls-to-action that work.',
            llm=self.llm, tools=[self.search_tool], allow_delegation=False, verbose=True
        )

# --- TASK DEFINITIONS (New Task and Final Output Structure) ---

class FlyerDesignTasks:
    def briefing_task(self, agent, topic, text_element, flyer_type):
        return Task(
            description=f"""
                Analyze the provided information to create a Creative Brief.
                - Topic: "{topic}"
                - Key Text/Slogan: "{text_element}"
                - Flyer Type: "{flyer_type}"
                Your brief must clearly define: Target Audience, Desired Emotion, and Core Message.
                The current date is {datetime.now().strftime('%Y-%m-%d')}.
            """,
            expected_output="A concise creative brief document.",
            agent=agent,
        )

    def visual_concept_task(self, agent, context):
        return Task(
            description="""
                Based on the creative brief, develop a full visual concept.
                Include: A core Visual Metaphor/Scenario, a mood-setting Color Palette, Composition ideas, and a modern Artistic Style.
            """,
            expected_output="A detailed visual concept document.",
            agent=agent, context=context
        )

    def prompt_crafting_task(self, agent, context):
        return Task(
            description="""
                Synthesize the Creative Brief and Visual Concept into a single, masterful image generation prompt for Google's Imagen model.
                The prompt must be a detailed descriptive paragraph, including specifics on subject, setting, lighting, colors, mood, and camera settings.
                **Crucially, do NOT include the actual text/slogan in the image prompt itself.** The image is the visual background.
                Your final output is ONLY the prompt.
            """,
            expected_output="A single, detailed paragraph containing the final image prompt.",
            agent=agent, context=context
        )

    # NEW TASK
    def copywriting_task(self, agent, context):
        return Task(
            description="""
                Based on the Creative Brief and the Visual Concept, write a compelling social media post to accompany the generated flyer image.
                The post should:
                1.  Grab attention with a strong hook.
                2.  Incorporate the key text/slogan: "{text_element}" naturally.
                3.  Reflect the desired emotion of the campaign.
                4.  Include 3-5 relevant and trending hashtags.
                5.  End with a clear call-to-action or a question to spark engagement.
            """,
            expected_output="A complete social media post, including the main text and hashtags.",
            agent=agent, context=context
        )

# --- CREW SETUP (Updated to include the new agent and aggregate the output) ---

def create_flyer_crew(topic, text_element, flyer_type):
    agents = FlyerDesignAgents()
    tasks = FlyerDesignTasks()

    # Instantiate Agents
    brief_specialist = agents.creative_brief_specialist()
    concept_developer = agents.visual_concept_developer()
    prompt_crafter = agents.imagen_prompt_crafter()
    copywriter = agents.social_media_copywriter()

    # Instantiate Tasks
    briefing = tasks.briefing_task(brief_specialist, topic, text_element, flyer_type)
    visualizing = tasks.visual_concept_task(concept_developer, context=[briefing])
    
    # These two tasks can potentially run in parallel after visualizing
    crafting_prompt = tasks.prompt_crafting_task(prompt_crafter, context=[briefing, visualizing])
    crafting_copy = tasks.copywriting_task(copywriter, context=[briefing, visualizing])

    # Assemble the Crew
    flyer_crew = Crew(
        agents=[brief_specialist, concept_developer, prompt_crafter, copywriter],
        tasks=[briefing, visualizing, crafting_prompt, crafting_copy],
        process=Process.sequential,
        verbose=2,
    )
    
    # The result will be a list of the outputs from each task.
    # The prompt is from task 3 (index 2) and the copy is from task 4 (index 3).
    return flyer_crew
import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from google.api_core.exceptions import PermissionDenied, ClientError

# The location is generally static, but could also be a parameter if needed.
LOCATION = "us-central1"

# The function is now self-contained and relies on parameters.
# Using st.cache_data is still a good practice to prevent re-generation for the same prompt.
@st.cache_data
def generate_image_with_imagen(project_id: str, prompt: str):
    """
    Generates an image using Google's Imagen model in Vertex AI.
    Requires the user's Google Cloud Project ID.
    """
    try:
        # Initialize Vertex AI with the user-provided project ID.
        # This will use the Application Default Credentials (ADC) from the environment.
        vertexai.init(project=project_id, location=LOCATION)
        
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        
        images = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
        )
        
        return images[0]._image_bytes

    except PermissionDenied:
        st.error("Authentication failed: Permission Denied.")
        st.error("Please ensure you have authenticated your environment by running 'gcloud auth application-default login' in your terminal and that the Vertex AI API is enabled for your project.")
        return None
    except ClientError as e:
        # Handle cases where the project ID might be invalid or the API not enabled.
        st.error(f"A client error occurred: {e}")
        st.error(f"Please double-check that your Project ID ('{project_id}') is correct and that the Vertex AI API is enabled in the Google Cloud Console.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during image generation: {e}")
        return None
        
        
 
import streamlit as st
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# The location is generally fixed for the model endpoint
LOCATION = "us-central1" 

@st.cache_data # Cache the result to avoid re-generating on every interaction
def generate_image_with_imagen(prompt: str, project_id: str):
    """
    Generates an image using Google's Imagen model in Vertex AI.
    Args:
        prompt (str): The detailed text prompt for image generation.
        project_id (str): The user's Google Cloud Project ID.
    """
    # The project_id check is now handled in the main app before calling this function.
    try:
        # Initialize Vertex AI with the user-provided project
        vertexai.init(project=project_id, location=LOCATION)
        
        # Load the model
        model = ImageGenerationModel.from_pretrained("imagegeneration@006")
        
        # Generate image
        images = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio="1:1",
        )
        
        # Return the raw image data
        return images[0]._image_bytes

    except Exception as e:
        st.error(f"An error occurred during image generation: {e}")
        st.warning("""
        **Troubleshooting Tips:**
        1.  Ensure the Project ID you entered is correct.
        2.  Make sure you have authenticated your local environment by running `gcloud auth application-default login` in your terminal.
        3.  Verify that the Vertex AI API is enabled in your Google Cloud project.
        """)
        return None
 
 import streamlit as st
from flyer_crew import create_flyer_crew
from image_generator import generate_image_with_imagen

# --- Page Configuration ---
st.set_page_config(page_title="AI Flyer Production Studio", page_icon="🚀", layout="wide")

# --- Sidebar for Credentials ---
st.sidebar.title("🔐 Credentials & Setup")
st.sidebar.markdown("""
To generate images, you need to provide your Google Cloud Project ID.
This application uses Application Default Credentials (ADC). Before you begin, please authenticate in your terminal:
""")
st.sidebar.code("gcloud auth application-default login")

# Input fields for credentials
project_id_input = st.sidebar.text_input(
    "Enter Your Google Cloud Project ID",
    help="You can find your Project ID on the dashboard of your Google Cloud Console.",
    value=st.session_state.get('project_id', '') # Persist value across reruns
)

# Although not used by the current ADC method, we include it as requested.
# It's good practice for other APIs that might be added later.
api_key_input = st.sidebar.text_input(
    "Enter API Key (Optional)",
    type="password",
    help="Currently not used for Imagen, but can be used for other services.",
    value=st.session_state.get('api_key', '')
)

# Store credentials in session state
st.session_state['project_id'] = project_id_input
st.session_state['api_key'] = api_key_input


# --- Main Page Content ---
st.title("🚀 AI Flyer Production Studio")
st.markdown("From idea to share-ready asset in minutes. Your expert AI crew will design a concept, generate a flyer, and write the social media copy.")

# --- User Input for Flyer ---
st.header("Step 1: Describe Your Flyer")
col1, col2 = st.columns(2)
with col1:
    topic = st.text_input("**Topic or Theme:**", placeholder="e.g., Street Evangelism, Green Party Campaign")
with col2:
    text_element = st.text_input("**Key Text (Verse, Slogan, etc.):**", placeholder='e.g., "John 3:16", "Vote for a Greener Tomorrow"')

flyer_type = st.selectbox("**Flyer Type:**", ("Social Media Post (Square)", "Poster (Portrait)", "Banner (Landscape)"))


# --- Execution Logic ---
st.header("Step 2: Produce & Distribute")

# The button is always visible, but the logic inside is gated.
if st.button("🚀 Generate Flyer and Social Copy"):
    # GATING LOGIC: Check if required credentials are provided
    if not st.session_state.get('project_id'):
        st.error("❌ Please enter your Google Cloud Project ID in the sidebar to proceed.")
    elif not topic or not text_element:
        st.error("❌ Please provide both a Topic and Key Text to proceed.")
    else:
        # If credentials are provided, run the full process
        crew_result = None
        with st.spinner("Your AI Design Studio is developing the concept..."):
            try:
                flyer_design_crew = create_flyer_crew(topic, text_element, flyer_type)
                crew_result = flyer_design_crew.kickoff()
                
                image_prompt = crew_result.tasks_output[2].exported_output
                social_copy = crew_result.tasks_output[3].exported_output
                
                st.success("Concept approved! Prompt and copy are ready.")
                st.subheader("🎨 Generated Image Prompt")
                st.code(image_prompt, language="text")

            except Exception as e:
                st.error(f"An error occurred during AI crew execution: {e}")
                crew_result = None

        if crew_result:
            image_bytes = None
            with st.spinner("Sending prompt to Google Imagen for final rendering..."):
                # Pass the project_id from session_state to the function
                image_bytes = generate_image_with_imagen(
                    project_id=st.session_state['project_id'],
                    prompt=image_prompt
                )

            if image_bytes:
                st.success("Rendering complete!")
                st.subheader("✅ Your Final Flyer")
                st.image(image_bytes, caption="Generated by Google Imagen")

                st.subheader("Step 3: Download & Share")
                st.download_button(
                    label="Download Flyer Image",
                    data=image_bytes,
                    file_name="generated_flyer.png",
                    mime="image/png"
                )
                st.text_area("✍️ Your Social Media Caption (Ready to Copy)", social_copy, height=150)

# --- Footer ---
st.markdown("---")
st.markdown("Developed by a CrewAI Expert & Digital Designer.")