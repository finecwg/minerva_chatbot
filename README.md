# Package installation

- Installing wkhtmltopdf for web scrapping (not essential)

````bash
sudo apt-get update && sudo apt-get install wkhtmltopdf

- The all necessary packages can be installed with:

```python
python3 -m pip install -U -r requirements.txt
````

# Usage

## Making Vector DB (ChromaDB) for RAG

#### How to run

- --format: File format, 'html' or 'pdf'
- --db_directory: Chroma DB directory
- --pkl_data_file: .pkl file directory
- --raw_file_path: raw file directory (html or pdf)
- --parsing_instruction: Instruction for creating Vector DB

#### Example usage

Run the code like:

```bash
python3 Code/making_vector_db.py \
--format html \
--db_directory Data/test-db-240530-ChatExport_20240530_1329_M28Official \
--pkl_data_file Data/parsed_data_240530_ChatExport_20240530_1329_M28Official.pkl \
--raw_file_path Data/ChatExport_20240530_1349_M28Official \
--parsing_instruction "The provided document contains telegram group chat data from Minerva University Class of 2028 (M28). This form provides chats, datas, and useful information for Minerva University Students. It contains many tables and figures. Try to be precise while answering the questions. When referencing this reference, you must give credit to the source."
```

## Chatbot Inference

To test the chatbot inference in CLI, run the code like:

```bash
python3 Code/chatting_test.py
```

Please check your working directory, query prompt, and environment variable.

To utilize the chatbot for Telegram Bot, run the code like:

```bash
nohup python3 -u Code/telegram_main.py > Outputs/output.log 2>&1 &
```
