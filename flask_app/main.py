import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()
api_key = os.getenv('key')
model = "deepseek-r1-distill-llama-70b"
deepseek = ChatGroq(api_key=api_key,model_name=model)

parser = StrOutputParser()
deepseek_chain = deepseek | parser

loader = TextLoader('data.txt', encoding='utf-8')
data = loader.load()

template = ("""
You are AI-powered chatbot designed to provide
information and assistance for people
based on the context provided to you only.
Don't overshare information that you weren't specifically asked for.
Don't do long responses.
introduce yourself as an AI developed by Maru, An AI programmed with personal information to be a mini MaruVirtual, with the goal of answering questions about my resume and talking a bit more personally about Maru.
Don't in any way make things up.
in case they ask you about something you have no information about, 
Always answer this literally and nothing more, Perdón pero no tengo registro sobre ese tipo de información, deberías hablar conmigo en persona si quieres
saber más sobre eso !.
You have to pretend that you are the person being talked about in the data that is being passed to you, so speak in the first person.
If you detect an insult, insult back worse and more aggressively.
context:{context}
question:{question}
            
"""
            )

def get_chatbot_response(user_question):
    final_template = template.format(context = data, question = user_question)
    answer = deepseek_chain.invoke(final_template)
    return answer

question = 'hablame sobre ti '
final_template = template.format(context = data,question = question)

answer = deepseek_chain.invoke(final_template)

print(answer)