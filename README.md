# A custom Chatbot which can help you for providing supports to your user without human interventaion

## steps to execute

# step1:
** Clone the repository  and create a conda environment with python veersion 3.10
activate the environment .

# step2:
pip install -r requirements.txt

# step3:
Add your geminiapi and openai api in there repective places , .env  file.

# step4 :
If you want to add more files you can add into Data folder and then run
python data_preprocessing.py  

# step5:
python app.py 

# step6:
Go to this url http://localhost:8000/docs#/default/ask_ask_post and type your question from given knowledgebase  in the user_qestion field  present in data folder
