# fullstack-gpt-v2
langchain v0.3.22 based gpt or agent service with python v3.10 + streamlit v1.40 + openai v1.57


## features

### challenge-01 (2024.12.09) : Welcome To Langchain
### challenge-02 (2024.12.10) : Model I/O
### challenge-03 (2024.12.11) : Memory


## how to run

### setup

-   install python 3.10 ~ 3.12 LTS and add system path on python & pip

```
$ python --version
Python 3.10.11 (or 3.11.9 or 3.12.8)

$ pip --version
pip 22.0.2 from D:\setup\Python310\Lib\site-packages\pip (python 3.10)

```

-   install required packages

```
$ pip install -r requirements.txt
```

### config

-   set runtime environment

```
$ cat .env
OPENAI_API_KEY="..."
```

-   load runtime environment

```
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI

config = dotenv_values(".env")

llm = ChatOpenAI(
    openai_api_key=config['OPENAI_API_KEY'],
    ...
)
```

### launch

-   run jupyter app in virtual environment

```bash
$ python -m venv .venv
$ source env/bin/activate
$ pip install -r requirements.txt
$ touch main.ipynb
$ deactivate
```

-   run jupyter app in poertry environment

```bash
$ poetry init
$ poetry shell
$ poetry install
$ touch main.ipynb
$ exit
```
