import json
import streamlit as st
import streamlit.components.v1 as components
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI

from pathlib import Path


st.set_page_config(
    page_title="::: Quiz GPT :::",
    page_icon="🧐",
)
st.title("Quiz GPT")

st.markdown(
    """
        Use this chatbot to test your knowledge with quiz challenge.

        1. Input your OpenAI API Key on the sidebar.
        2. Choose an AI model (gpt-4o-mini, ...).
        3. Choose a challenge level (Easy | Medium | Hard).
        4. Choose a favorite language (Korean, ...).
        5. Choose a knowledge source (wiki or file).
        6. Input search keyword or Upload your file.
        7. Challenge the quiz created by an AI tutor.
    """
)


# extension feature
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions & answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = PromptTemplate.from_template(
    """
        You are a helpful assistant that is playing a role of a famous teacher that absolutely loves quiz.
        Based ONLY on the following context, make 5 questions to test the user's knowledge about the text.
        Each question should have 4 answers, three of them must be incorrect and one should be correct.
        Be sure that all the questions & answers should be written in {language} Language.
        The difficulty level of the problem is '{level}'.

        Context: {context}
    """
)


@st.cache_resource(show_spinner="Making Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5, lang="ko")
    docs = retriever.invoke(term)
    return docs


@st.cache_resource(show_spinner="Loading your file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.files/{file.name}"
    Path("./.files").mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb+") as f:
        f.write(file_content)

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_resource(show_spinner="Making Quiz...")
def run_quiz_chain(_docs, _level, _language):
    chain = prompt | llm
    return chain.invoke(
        {
            "context": _docs,
            "level": _level,
            "language": _language,
        }
    )


# Define Reset Actions
def reset_all():
    # for key in st.session_state.keys():
    #     del st.session_state[key]
    # wiki_search.clear()
    # split_file.clear()
    # run_quiz_chain.clear()
    # st.experimental_rerun()
    # st.experimental_set_query_params(refresh="true")
    st.markdown(
        """
            <script>
                location.reload();
            </script>
        """,
        unsafe_allow_html=True,
    )
    components.html(
        """
            <script>
                window.location.reload();
            </script>
        """,
        height=0,
    )


with st.sidebar:
    docs = None
    topic = None
    response = None

    # Input LLM API Key
    openai_api_key = st.text_input("Input your OpenAI API Key", type="password")

    # Select AI Model
    selected_model = st.selectbox(
        "Choose your AI Model",
        ("gpt-4o-mini", "gpt-3.5-turbo"),
    )

    # Select Challenge Level
    challenge_level = st.selectbox(
        "Choose your challenge level",
        ("Easy", "Medium", "Hard"),
    )

    # Select Favorite Language
    language = st.selectbox(
        "Choose your favorite language",
        ("Korean", "English", "Spanish"),
    )

    # Select Quiz Target (Wiki or Custom File)
    quiz_target = st.selectbox(
        "Choose what you want to use",
        (
            "Wikipedia Article",
            "Your Custom Document",
        ),
    )

    # Upload Document File on Your Custom Document
    if quiz_target == "Your Document File":
        file = st.file_uploader(
            "Upload a txt, pdf or docx file",
            type=["docx", "pdf", "txt"],
        )
        if file:
            docs = split_file(file)

    # Input Search Keyword on Wikipedia Article
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic and len(topic) > 2:
            docs = wiki_search(topic)

    # Reset all settings
    if st.button("Reset"):
        reset_all()
        placeholder = st.empty()
        with placeholder.container():
            st.info("Content has been reset.")

    # Link to Github Repo
    st.markdown("---")
    github_link = (
        "https://github.com/toweringcloud/fullstack-gpt-v2/blob/main/challenge-06.py"
    )
    badge_link = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    st.write(f"[![Repo]({badge_link})]({github_link})")


if not openai_api_key:
    st.error("Please input your OpenAI API Key on the sidebar")
elif quiz_target == "Your Document File" and not docs:
    st.error("Please upload your document file!")
elif quiz_target == "Wikipedia Article" and not docs:
    st.error("Please input your keyword to search on Wikipedia!")
else:
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=selected_model,
        temperature=0.1,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
    ).bind(
        function_call={
            "name": "create_quiz",
        },
        functions=[
            function,
        ],
    )

    response = run_quiz_chain(docs, challenge_level, language)
    response = response.additional_kwargs["function_call"]["arguments"]

    with st.form("questions_form"):
        questions = json.loads(response)["questions"]
        question_count = len(questions)
        success_count = 0

        for idx, question in enumerate(questions):
            st.markdown(f'#### {idx+1}. {question["question"]}')
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )

            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
                success_count += 1
            elif value is not None:
                st.error("Wrong!")

        if question_count == success_count:
            st.balloons()

        button = st.form_submit_button("Submit")
