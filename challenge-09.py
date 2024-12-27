import json
import streamlit as st
from duckduckgo_search import DDGS
from openai import OpenAI, AssistantEventHandler
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from typing_extensions import override


st.set_page_config(
    page_title="::: Research Assistant :::",
    page_icon="📜",
)
st.title("Research Assistant")

st.markdown(
    """
        Use this chatbot to research something you're curious about.

        1. Choose a favorite language (Korean, ...).
        2. Choose an AI model (gpt-4o-mini, ...).
        3. Input your OpenAI API Key on the sidebar.
        4. Ask questions to research something you wonder.
    """
)
st.divider()


with st.sidebar:
    # Select Favorite Language
    language = st.selectbox(
        "Choose your favorite language",
        ("Korean", "English", "Spanish", "Chinese", "Japanese"),
    )

    # Select AI Model
    selected_model = st.selectbox(
        "Choose your AI Model",
        (
            "gpt-4o-mini",
            "gpt-3.5-turbo",
        ),
    )

    # Input LLM API Key
    openai_api_key = st.text_input(
        "Input your OpenAI API Key",
        type="password",
    )

    # Link to Github Repo
    st.markdown("---")
    github_link = (
        "https://github.com/toweringcloud/fullstack-gpt-v2/blob/main/challenge-09.py"
    )
    badge_link = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    st.write(f"[![Repo]({badge_link})]({github_link})")


# define function logic
def WikipediaSearchTool(params):
    query = params["query"]
    search = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    return search.invoke(query)


def DuckDuckGoSearchTool(params):
    query = params["query"]
    search = DDGS().text(query)
    return json.dumps(list(search))


def SearchResultParseTool(params):
    link = params["link"]
    loader = WebBaseLoader(link, verify_ssl=True)
    return loader.load()


# define function mapper
functions_map = {
    "wiki_search": WikipediaSearchTool,
    "ddg_search": DuckDuckGoSearchTool,
}

# define function schema
functions = [
    {
        "type": "function",
        "function": {
            "name": "wiki_search",
            "description": "Search information on wikipedia site",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "user's input",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ddg_search",
            "description": "Search information on duckduckgo site",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "user's input",
                    },
                },
                "required": ["query"],
            },
        },
    },
]


# https://platform.openai.com/docs/assistants/tools/function-calling?context=streaming
# assistant event handler with streaming
class EventHandler(AssistantEventHandler):
    message = ""

    @override
    def on_event(self, event):
        # Retrieve events that are denoted with 'requires_action'
        # since these will have our tool_calls
        if event.event == "thread.run.requires_action":
            run_id = event.data.id
            self.message_box = st.empty()
            self.handle_requires_action(event.data, run_id)

    def handle_requires_action(self, data, run_id):
        tool_outputs = []

        for tool in data.required_action.submit_tool_outputs.tool_calls:
            print(
                f"# tool: {tool.id} | {tool.function.name} | {tool.function.arguments}"
            )
            tool_outputs.append(
                {
                    "tool_call_id": tool.id,
                    "output": functions_map[tool.function.name](
                        json.loads(tool.function.arguments)
                    ),
                }
            )

        # Submit all tool_outputs at the same time
        self.submit_tool_outputs(tool_outputs, run_id)

    def submit_tool_outputs(self, tool_outputs, run_id):
        client = st.session_state["client"]

        # Use the submit_tool_outputs_stream helper
        with client.beta.threads.runs.submit_tool_outputs_stream(
            thread_id=self.current_run.thread_id,
            run_id=self.current_run.id,
            tool_outputs=tool_outputs,
            event_handler=EventHandler(),
        ) as stream:
            for text in stream.text_deltas:
                self.message += text
                self.message_box.markdown(self.message)
                print(text, end="", flush=True)
            print()
            self.save_research_result()

    def save_research_result(self):
        st.session_state["result"] = self.message
        file_path = "./challenge-09.result"
        with open(file_path, "w+", encoding="utf-8") as f:
            f.write(self.message)

        st.markdown(f"✨ research result saved at {file_path}")
        st.download_button("::: Download :::", st.session_state["result"])


def main():
    client = None

    if "client" in st.session_state:
        client = st.session_state["client"]
        assistant = st.session_state["assistant"]
        thread = st.session_state["thread"]

    else:
        if not openai_api_key:
            st.error("Please input your OpenAI API Key on the sidebar.")
            return

        client = OpenAI(api_key=openai_api_key)

        assistant = client.beta.assistants.create(
            name="Research Expert",
            instructions="""
                You are a professional web research assistant. 
                Search information by query and summarize the result.
                Please, all the chat messages should be written as {} language.
            """.format(
                language
            ),
            temperature=0.1,
            model=selected_model,
            tools=functions,
        )
        thread = client.beta.threads.create()

        st.session_state["client"] = client
        st.session_state["assistant"] = assistant
        st.session_state["thread"] = thread

        with st.chat_message("ai"):
            st.markdown("I'm ready! Ask away!")

    # show messages in the thread of your assistant
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    if messages:
        messages = list(messages)
        messages.reverse()
        for message in messages:
            st.chat_message(message.role).write(message.content[0].text.value)

    # ready to research your question
    question = st.chat_input("Ask anything you're curious about.")
    if question:
        with st.chat_message("human"):
            st.markdown(question)

        message = client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content=question
        )

        with st.chat_message("ai"):
            st.markdown("Researching about `{}`".format(question))

            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                event_handler=EventHandler(),
            ) as stream:
                stream.until_done()
    else:
        st.empty()


try:
    main()

except Exception as e:
    st.write(e)
