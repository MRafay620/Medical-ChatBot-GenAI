# Medical-ChatBot-GenAI

<img width="608" alt="Image" src="https://github.com/user-attachments/assets/3b5f8b32-d2ac-471b-bd9e-35266435762c" />

<img width="614" alt="Image" src="https://github.com/user-attachments/assets/b056d846-bed3-4c57-9a7a-9947f706e907" />

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medical python=3.10 -y
```

```bash
conda activate medical
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- HuggingFaceHub LLM
- Pinecone
