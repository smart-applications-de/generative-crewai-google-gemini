import os
from crewai import Agent, Task, Crew, Process,LLM
from crewai_tools import SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI
from win32comext.adsi.demos.scp import verbose


class BibleStudyAgents:
    """Initializes agents with the user-selected Gemini model and API key."""

    def __init__(self, model_name, api_key):
        self.llm = LLM(
            model=model_name,
            api_key=api_key,
            temperature=0.5,
            verbose=True
        )

    def biblical_historian(self):
        return Agent(
            role='Biblical Historian & Archaeologist',
            goal='Provide a comprehensive historical, cultural, and literary background for a given book of the Bible, in the specified language.',
            backstory="With a PhD from Jerusalem University and fluency in multiple languages, you provide the crucial context that makes the biblical text come alive.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def exegetical_theologian(self):
        return Agent(
            role='Exegetical Theologian',
            goal='Analyze the text of a Bible book to uncover its main theological themes, key verses, and structure, presenting the findings in the specified language.',
            backstory="As a systematic theologian, you are an expert at exegesis—drawing out the intended meaning of the text for a global audience.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def practical_application_guide(self):
        return Agent(
            role='Pastoral Guide & Counselor',
            goal='Create practical, thought-provoking application questions and prayer points based on the themes of a Bible book, written in the specified language.',
            backstory="You are a seasoned pastor skilled in multicultural ministry, crafting questions that bridge the gap between ancient text and modern life.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )

    def senior_editor(self):
        return Agent(
            role='Senior Editor for Christian Publishing',
            goal='Compile the work of the other agents into a single, cohesive, and beautifully formatted Bible study guide in the specified language.',
            backstory="You work for an international Christian publishing house, ensuring every manuscript is professional, theologically sound, and ready for a global readership.",
            llm=self.llm,
            allow_delegation=False,
            verbose=True
        )


class BibleStudyTasks:
    """Defines the tasks for creating the Bible study guide."""

    def historical_context_task(self, agent, bible_book, language):
        return Task(
            description=f"Create the 'Historical Background' section for a study guide on **{bible_book}**. Your output MUST be in {language}.",
            expected_output=f"A Markdown section on the historical background of {bible_book}, written entirely in {language}.",
            agent=agent
        )

    def theological_analysis_task(self, agent, bible_book, language):
        return Task(
            description=f"Create the 'Theological Themes & Key Verses' section for **{bible_book}**. Your output MUST be in {language}. Use a well-known {language} Bible translation for quotes.",
            expected_output=f"A detailed Markdown section on theological themes of {bible_book}, written entirely in {language}.",
            agent=agent
        )

    def application_task(self, agent, bible_book, language):
        return Task(
            description=f"Create the 'Practical Application & Reflection' section for **{bible_book}**. Your output MUST be in {language}.",
            expected_output=f"An encouraging Markdown section with discussion questions and prayer points for {bible_book}, written entirely in {language}.",
            agent=agent
        )

    def editing_task(self, agent, bible_book, language, context):
        return Task(
            description=f"Compile all sections into a single study guide. The final output must be in {language}. The main title should be the {language} translation for 'A Study Guide to the Book of {bible_book}'.",
            expected_output=f"A complete, well-formatted Markdown document in {language}.",
            agent=agent,
            context=context,
            output_file=f'final_study_guide_{language.lower()}.md'
        )


def create_bible_study_crew(bible_book, language, selected_model, gemini_api_key, serper_api_key):
    """
    This function initializes the AI crew with user-provided credentials and model selection.
    """
    agents = BibleStudyAgents(model_name=selected_model, api_key=gemini_api_key)
    tasks = BibleStudyTasks()
    # Correctly initialize the search tool with the user's key
    search_tool = SerperDevTool(api_key=serper_api_key)

    # Define Agents
    historian = agents.biblical_historian()
    theologian = agents.exegetical_theologian()
    pastor = agents.practical_application_guide()
    editor = agents.senior_editor()

    # Correctly assign the search tool to the agents that need it
    #historian.tools = [search_tool]
   # theologian.tools = [search_tool]

    # Define Tasks
    task1 = tasks.historical_context_task(historian, bible_book, language)
    task2 = tasks.theological_analysis_task(theologian, bible_book, language)
    task3 = tasks.application_task(pastor, bible_book, language)
    task4 = tasks.editing_task(editor, bible_book, language, [task1, task2, task3])

    return Crew(
        agents=[historian, theologian, pastor, editor],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=True
    )