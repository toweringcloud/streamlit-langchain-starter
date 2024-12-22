# fullstack-gpt-v2
langchain v0.3.10 based gpt or agent service with python v3.10 + streamlit v1.41 + openai v1.57


## features

### challenge-01 (2024.12.09) : Welcome To Langchain
### challenge-02 (2024.12.10) : Model I/O
### challenge-03 (2024.12.11) : Memory
### challenge-04 (2024.12.15) : RAG
### challenge-05 (2024.12.17) : Streamlit is 🔥
-   [demo] https://toweringcloud-document-gpt.streamlit.app
### challenge-06 (2024.12.22) : QuizGPT Turbo
-   [demo] https://toweringcloud-quiz-gpt.streamlit.app


## how to run

### setup

-   install python 3.10 ~ 3.12 LTS and add system path on python & pip

```sh
$ python --version
Python 3.10.11 (or 3.11.9 or 3.12.8)

$ pip --version
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
```

### config

-   set runtime environment

```sh
$ cat .env
OPENAI_API_KEY="..."
```

-   load runtime environment

```python
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

config = dotenv_values(".env")

chat = ChatOpenAI(
    openai_api_key=config['OPENAI_API_KEY'],
    ...
)
```

### launch

-   run jupyter app in virtual environment

```sh
$ python -m venv .venv
$ source env/bin/activate
$ pip install -r requirements.txt
$ pip list
$ touch main.ipynb && code .
$ deactivate
```

-   run jupyter app in poertry environment

```sh
$ poetry init
$ poetry shell
$ poetry install
$ poetry show
$ touch main.ipynb && code .
$ exit
```

-   run streamlit app in root environment

```sh
$ streamlit run main.py
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```
