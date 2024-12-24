import streamlit as st
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import SitemapLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pathlib import Path
from urllib.parse import urlparse


st.set_page_config(
    page_title="::: Site GPT :::",
    page_icon="🖥️",
)
st.title("Site GPT")

st.markdown(
    """
        Welcome to Site GPT!

        Ask questions about the content of a website.\n
        Start by writing the URL of the website on the sidebar.
    """
)


answers_prompt = ChatPromptTemplate.from_template(
    """
        Using ONLY the following context, answer the user's question.
        If you can't just say you don't know, don't make anything up.
        Then, give a score to the answer between 0 and 5.
        If the answer answers the user question the score should be high, else it should be low.
        Make sure to always include the answer's score even if it's 0.

        Context: {context}

        Examples:

        Question: How far away is the moon?
        Answer: The moon is 384,400 km away.
        Score: 5
                                    
        Question: How far away is the sun?
        Answer: I don't know
        Score: 0

        Your turn!

        Question: {question}
    """
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    answers_chain = answers_prompt | llm

    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                Use ONLY the following pre-existing answers to answer the user's question.
                Use the answers that have the highest score (more helpful) and favor the most recent ones.
                Cite sources and return the sources of the answers as they are, do not change them.

                Answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    choose_chain = choose_prompt | llm

    condensed = "\n\n".join(
        f"{answer['answer']}\n- Source:{answer['source']}\n- Date:{answer['date']}\n"
        for answer in answers
    )

    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    header = soup.find("header")
    if header:
        header.decompose()

    footer = soup.find("footer")
    if footer:
        footer.decompose()

    return str(soup.get_text()).replace("\n", " ").replace("\xa0", " ")


@st.cache_resource(show_spinner="Loading website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = (
        SitemapLoader(
            url,
            filter_urls=[
                r"^(.*\/ai-gateway\/).*",
                r"^(.*\/vectorize\/).*",
                r"^(.*\/workers-ai\/).*",
            ],
            parsing_function=parse_page,
            continue_on_failure=True,
        )
        if "cloudflare" in sitemap_url
        else SitemapLoader(url, parsing_function=parse_page, continue_on_failure=True)
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)

    cache_path = "./.cache"
    embedding_path = f"{cache_path}/challenge-07"
    dir_path = f"{embedding_path}/{urlparse(url).netloc}"
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    cache_dir = LocalFileStore(dir_path)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


with st.sidebar:
    docs = None

    # Input LLM API Key
    openai_api_key = st.text_input("Input your OpenAI API Key", type="password")

    # Select LLM Model
    selected_model = st.selectbox(
        "Choose your AI Model",
        ("gpt-4o-mini", "gpt-3.5-turbo"),
    )

    # Select Site Map
    selected_sitemap = st.selectbox(
        "Choose a SiteMap",
        ("Cloudflare"),
    )

    # Link to Github Repo
    st.markdown("---")
    github_link = (
        "https://github.com/toweringcloud/fullstack-gpt-v2/blob/main/challenge-07.py"
    )
    badge_link = "https://badgen.net/badge/icon/GitHub?icon=github&label"
    st.write(f"[![Repo]({badge_link})]({github_link})")


if not openai_api_key:
    st.error("Please input your OpenAI API Key on the sidebar")
else:
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=selected_model,
        temperature=0.1,
        streaming=True,
        callbacks=[
            StreamingStdOutCallbackHandler(),
        ],
    )

    # Setup SiteMap URL
    if selected_sitemap:
        if selected_sitemap == "Cloudflare":
            sitemap_url = "https://developers.cloudflare.com/sitemap-0.xml"
        elif selected_sitemap == "Flutter":
            sitemap_url = "https://docs.flutter.dev/sitemap.xml"

    if sitemap_url:
        retriever = load_website(sitemap_url)

        st.divider()
        query = st.text_input("Ask a question to the website.")
        if query:
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )
            result = chain.invoke(query)
            st.markdown(result.content.replace("$", "\$"))
