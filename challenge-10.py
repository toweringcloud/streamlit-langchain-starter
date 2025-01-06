import glob
import imageio_ffmpeg
import io
import math
import os
import platform
import requests
import streamlit as st
import subprocess
from langchain.embeddings import CacheBackedEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import which
from pytubefix import YouTube
from pytubefix.exceptions import LiveStreamError


st.set_page_config(
    page_title="::: Meeting GPT :::",
    page_icon="💼",
)
st.title("Meeting GPT")

st.markdown(
    """
        Use this chatbot to ask any questions about your video.

        1. Choose a favorite language (Korean, ...).
        2. Choose an AI model (gpt-4o-mini, ...).
        3. Input your OpenAI API Key on the sidebar.
        4. Choose a video source (file or youtube).
        5. Upload your file or input video channel.
        6. Ask questions to research something you wonder.
    """
)


def check_website_availale(url):
    try:
        response = requests.get(url, timeout=3)
        return response.status_code
    except requests.exceptions.RequestException as e:
        print(f"Error checking {url}: {e}")
        return 500


def check_ffmpeg_installed():
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            return (
                True,
                result.stdout.splitlines()[0],
            )
        else:
            return False, result.stderr
    except FileNotFoundError:
        return False, "ffmpeg not found"


def install_ffmpeg_on_platform():
    """
    Detects the operating system and installs FFmpeg accordingly.
    """
    os_type = platform.system()

    try:
        if os_type == "Windows":
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            zip_path = "ffmpeg.zip"
            print(f"Downloading FFmpeg from {url}...")
            subprocess.run(["curl", "-L", "-o", zip_path, url], check=True)

            print("Extracting FFmpeg...")
            subprocess.run(["tar", "-xf", zip_path, "-C", "C:\\"], check=True)
            print("FFmpeg installed successfully on Windows.")

        elif os_type == "Linux":
            print("Installing FFmpeg using apt...")
            subprocess.run(["sudo", "apt", "update"], check=True)
            subprocess.run(["sudo", "apt", "install", "-y", "ffmpeg"], check=True)
            print("FFmpeg installed successfully on Linux.")

        elif os_type == "Darwin":
            print("Installing FFmpeg using Homebrew...")
            subprocess.run(["brew", "install", "ffmpeg"], check=True)
            print("FFmpeg installed successfully on macOS.")

        else:
            print(f"Unsupported OS: {os_type}")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing FFmpeg: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def install_ffmpeg():
    try:
        imageio_ffmpeg.get_ffmpeg_exe()
        is_installed, message = check_ffmpeg_installed()

        if is_installed:
            return True, "FFmpeg is installed."
        else:
            print(message)
            install_ffmpeg_on_platform()
            return check_ffmpeg_installed()

    except Exception as e:
        return False, str(e)


def has_transcript():
    fileEmpty = True
    fileExist = os.path.exists(transcript_path)
    if fileExist:
        fileEmpty = Path(transcript_path).stat().st_size == 0
    return fileExist and not fileEmpty


def extract_audio_from_video(video_path, audio_path):
    if has_transcript():
        return
    if os.path.exists(audio_path):
        print(f"audio({audio_path}) already exists!")
        return
    if not os.path.exists(video_path):
        print(f"video({video_path}) not available!")
        return

    # check ffmpeg utility installed
    is_installed, message = check_ffmpeg_installed()
    if is_installed:
        # FFmpeg is already installed: ffmpeg version 7.1-essentials_build-www.gyan.dev Copyright (c) 2000-2024 the FFmpeg developers
        st.success(f"FFmpeg is already installed: {message}")
        command = [ffmpeg_path, "-y", "-i", video_path, "-vn", audio_path]
        subprocess.run(command)
    else:
        with st.spinner("Installing FFmpeg..."):
            success, message = install_ffmpeg()
            if success:
                st.success(message)
                command = [ffmpeg_path, "-y", "-i", video_path, "-vn", audio_path]
                subprocess.run(command)
            else:
                st.error(f"Failed to install FFmpeg. Retry to install manually!")


@st.cache_data()
def cut_audio_in_chunks(audio_path, chunk_size, chunks_dir):
    if has_transcript():
        return
    if not os.path.exists(audio_path):
        print(f"audio({audio_path}) not available!")
        return

    AudioSegment.converter = which(ffmpeg_path)
    absolute_audio_path = os.path.abspath(audio_path)
    print(f"{absolute_audio_path} | {os.path.exists(absolute_audio_path)}")
    track = AudioSegment.from_mp3(absolute_audio_path)

    chunk_len = chunk_size * 60 * 1000
    chunks = math.ceil(len(track) / chunk_len)
    for i in range(chunks):
        start_time = i * chunk_len
        end_time = (i + 1) * chunk_len
        chunk = track[start_time:end_time]
        chunk.export(
            f"{chunks_dir}/chunk_{i}.mp3",
            format="mp3",
        )


@st.cache_resource(show_spinner="Transcribing audio...")
def transcribe_chunks(chunks_dir, destination):
    if has_transcript():
        return
    if not os.path.exists(audio_path):
        print(f"audio({audio_path}) not available!")
        return

    print(f"transcribe_chunks.i: {chunks_dir} | {destination}")
    files = glob.glob(f"{chunks_dir}/*.mp3")
    files.sort()
    print(f"transcribe_chunks.c: {len(files)} files\n")

    client = OpenAI(api_key=openai_api_key)

    for file in files:
        with open(file, "rb") as audio_file, open(destination, "a") as text_file:
            # https://platform.openai.com/docs/guides/speech-to-text
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
            text_file.write(transcription)
    print(f"transcribe_chunks.o: {Path(destination).stat().st_size} bytes")


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file_path):
    embedding_dir = f"{video_path}/embeddings"
    Path(embedding_dir).mkdir(parents=True, exist_ok=True)
    embedding_cache_dir = LocalFileStore(embedding_dir)
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, embedding_cache_dir
    )
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


with st.sidebar:
    cache_dir = "./.cache"
    work_dir = f"{cache_dir}/challenge-10"
    video_dir = None
    video_source = None

    os_type = platform.system()
    ffmpeg_path = (
        "C:/ffmpeg-7.1-essentials_build/bin/ffmpeg.exe"
        if os_type == "Windows"
        else "ffmpeg"
    )

    # Select Favorite Language
    language = st.selectbox(
        "Choose your favorite language",
        ("Korean", "English"),
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

    # Select Knowledge Target (File or Youtube)
    knowledge_target = None
    if os_type != "Linux":
        knowledge_target = st.selectbox(
            "Choose what you want to use",
            (
                "Your Video File",
                "Youtube Channel",
            ),
        )

    # Upload Video File or Input Youtube Channel
    if os_type == "Linux" or knowledge_target == "Your Video File":
        video_source = st.file_uploader(
            "Upload your Video file",
            type=["mp4", "avi", "mkv", "mov"],
        )
        if video_source:
            video_dir = f"{work_dir}/{video_source.name.split(".")[0]}"
            Path(video_dir).mkdir(parents=True, exist_ok=True)

    else:
        video_channel = st.text_input(
            "Input a Channel Name (after v=)",
        )
        if video_channel and len(video_channel) == 11:
            video_file = "{}.mp4".format(video_channel)
            video_dir = f"{work_dir}/{video_channel}"
            Path(video_dir).mkdir(parents=True, exist_ok=True)

            video_url = "https://youtube.com/watch?v={}".format(video_channel)
            if check_website_availale(video_url) == 200:
                try:
                    yt = YouTube(video_url)
                    stream = yt.streams.get_highest_resolution()
                    stream.download(output_path=video_dir, filename=video_file)

                    with open(f"{video_dir}/{video_file}", "rb") as file:
                        video_source = io.BytesIO(file.read())
                        video_source.name = video_file
                        video_source.type = "video/mpeg"
                        video_source.seek(0)

                except LiveStreamError as lse:
                    print("Unsupported on Live Streaming!")
                except Exception as e:
                    print(e)
            else:
                st.write(f"{video_url} not available!")

    # Github Repo Link
    st.markdown("---")
    github_link = (
        "https://github.com/toweringcloud/fullstack-gpt-v2/blob/main/challenge-10.py"
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
    )

if not video_source:
    st.error("Please upload your video or input youtube channel on the sidebar")

else:
    video_name = video_source.name
    video_extension = Path(video_name).suffix
    video_path = f"{video_dir}/{video_name}"
    audio_path = f"{video_path}".replace(video_extension, ".mp3")
    transcript_path = f"{video_path}".replace(video_extension, ".txt")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=800,
        chunk_overlap=100,
    )

    with st.status("Loading video...") as status:
        video_content = video_source.read()
        with open(video_path, "wb") as f:
            f.write(video_content)

        status.update(label="Extracting audio...")
        extract_audio_from_video(video_path, audio_path)

        if not os.path.exists(audio_path):
            print(f"audio({audio_path}) not available!")
        else:
            status.update(label="Cutting audio segments...")
            chunk_minutes = 3
            chunks_dir = f"{video_dir}/chunks"
            Path(chunks_dir).mkdir(parents=True, exist_ok=True)
            cut_audio_in_chunks(audio_path, chunk_minutes, chunks_dir)

            status.update(label="Transcribing audio...")
            transcribe_chunks(chunks_dir, transcript_path)

    transcript_tab, summary_tab, qa_tab = st.tabs(
        [
            "Transcript",
            "Summary",
            "Q&A",
        ]
    )

    with transcript_tab:
        if not os.path.exists(audio_path):
            print(f"audio({audio_path}) not available!")
        else:
            with open(transcript_path, "r") as file:
                st.write(file.read())

    with summary_tab:
        start = st.button("Generate summary")
        if start:
            if has_transcript():
                loader = TextLoader(transcript_path)
                docs = loader.load_and_split(text_splitter=splitter)

                first_summary_prompt = ChatPromptTemplate.from_template(
                    """
                    Write a concise summary of the following:
                    "{text}"
                    CONCISE SUMMARY:
                """
                )

                first_summary_chain = first_summary_prompt | llm | StrOutputParser()
                summary = first_summary_chain.invoke(
                    {"text": docs[0].page_content},
                )
                st.write(summary)

                refine_prompt = ChatPromptTemplate.from_template(
                    """
                    Your job is to produce a final summary.
                    We have provided an existing summary up to a certain point: {existing_summary}
                    We have the opportunity to refine the existing summary (only if needed) with some more context below.
                    ------------
                    {context}
                    ------------
                    Given the new context, refine the original summary.
                    If the context isn't useful, RETURN the original summary.
                """
                )

                refine_chain = refine_prompt | llm | StrOutputParser()

                with st.status("Summarizing...") as status:
                    for i, doc in enumerate(docs[1:]):
                        status.update(label=f"Processing document {i+1}/{len(docs)-1} ")
                        summary = refine_chain.invoke(
                            {
                                "existing_summary": summary,
                                "context": doc.page_content,
                            }
                        )
                        st.write(summary)

            else:
                print(f"{transcript_path} not available!")

    with qa_tab:
        if has_transcript():
            retriever = embed_file(transcript_path)
            question = st.text_input("Ask question about your audio script.")

            if question:
                docs = retriever.invoke(question)
                for doc in docs:
                    st.write(f"- {doc.page_content}")
        else:
            print(f"{transcript_path} not available!")
