# fullstack-gpt-v2
langchain v0.3.22 based gpt or agent service with python v3.10 + streamlit v1.40 + openai v1.57

## features

### challenge-01 (2024.12.09) : Welcome To Langchain


## how to run

### setup

-   install python 3.10.12 and add system path on python & pip

```
$ python --version
Python 3.10.12

$ pip --version
pip 22.0.2 from D:\setup\Python311\Lib\site-packages\pip (python 3.10)

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
from langchain.chat_models import ChatOpenAI

config = dotenv_values(".env")

llm = ChatOpenAI(
    openai_api_key=config['OPENAI_API_KEY'],
    ...
)
```

### launch

-   run jupyter app in virtual environment

```
$ python -m venv ./env
$ source env/bin/activate
$ touch main.ipynb
! select runtime kernel as venv - python 3.11.6
! run code & debug for testing
$ deactivate
```
