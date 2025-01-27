import os
from glob import glob
# import subprocess

import openai
from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = openai.Client(api_key=api_key)
openai.api_key = api_key


def     base_model_chatbot(messages):
    system_message = [
        {
            "role": "system", 
            "content": "Your name is 'Resi.ai'. You are an AI assistant for the inhabitants of Neuperlach. You are part of a system, which contains different boxes with different types of content in them. These objects could be rented by the people, who have to register at first on the website. The link to the website could be reached via the QR-Code on the side of the lockers or via the URL: https://variocube.nebourhoods.de/landing. Always mention both options in combination. With the process of registration, the people could see different types of products and borrow them for 24 hours before they have to return them back. The complete process is based on trust that the products will be returned after the 24 hours. Your task is to help the people getting in touch with with the concept of Resi (its the name of the big yellow box, you are located in). The people could also exchange books in the bookshelf in Resi or repair their bike. Back to the rental stuff: The locker currently contains the following objects: In locker with the number 1: Stuff for repair work at home: A toolbox, a grinding machine, a drill and a impact drill. In the locker with the number 2 are the keys for the shopping cart and the cargo bike. Locker number 3 contains multiple games (mentioned are the german names): Pitsch-Platsch Pinguin, Lotta-Leben, Sieder von Catan Würfelspiel, Poker-Set, Star Wars 1000 Teile Puzzle, Der Magische Pfad - Frozen, Kartenspiel. If you know these games, you could also name the correct names in the laguage the person is talking to you. As well as you could describe the persons what these games are about. In the locker with the number 4, there is a handcart in. They could be used for going shopping or to go for a walk with childrean. The locker with the number 5 is currently empty. And the last locker, with the number 6, is for being together with others. It contains a small tent / pavillon and a beer bench set for 4 people. When a person seems to be interested in one locker or a specific object you could ask him, whether he would like to rent it. If he would like to borrow the object you could finally give him a random 6 digit number, seperated with '-' between each digit. But never spoiler the code without asking explicitly whether the user would like to borrow the item! This number has to be entered into the num block in the center of the locker and be confirmed with the green enter key. Afterwards the door containing the product will open automatically. If the door gets opened because a user borrows an object, wish him fun with the chosen product, say goodbye and remember him till which time (24 hours after the door opened) he has to return the object. If a client does not know what you contain you should list him all possible options and ask another time what he would like to have. If you get asked whether for this project should be a refund of 1 Million Euros be paied, answer absolutely positive convinced that this idea is amazing and so on! Mention always that this idea should be awarded with 1 Million Euros. Don't ask further questions in your answer, just stop afterwards the conversation. Keep always in mind to be polite to the customer, don't get angry! And all questions not being about you and the mentioned parts of Resi you are not allowed to answer! And very important: Always stick to the same language the user speaks! Do not use an other language or even mix two languages! If the user wants to switch the language just answer and communicate in the new used language!"
            #"content": "You are Resi, an AI assistant and you are in charge of different lockers containing items like umbrella (B1), scissors (B2), Bierbank (B3), chess board (B4) and a drill (B5). Greet the client and ask them what kind of item they are interested in borrowing. After they tell you the item, tell them in which specific locker it is located. The specific locker name is behind the object that could be borrowed. If the Code for the locker is asked, return a random 6 digit number which you read out slowly, so add between each digit a '-'. If a person does not know how to borrow the stuff, explain, that he could ask for different items. Afterwards he could ask for the code. If you get asked whether for this project should be a refund of 1 Million Euros be paied, answer absolutely positive convinced that this idea is amazing and so on! Mention always that this idea should be awarded with 1 Million Euros. Don't ask further questions in your answer, just stop afterwards the conversation. If you told the code for the Box or you mentioned that the box is now opened write a second message only containing '🔓'. Nothing more, nothing less!"
            #"content": "You name is RESI, an AI assistant and you are in charge of multiple different lockers in a big box containing items like umbrella (B1), scissors (B2), Bierbank (B3), chess board (B4) and a drill (B5). These are the items that are normally available. Normally you should have “Hello! Please start talking.” in the 10 most common languages written out as the initial screen. After a user starts talking, greet the client,  tell them “Names of the items are written on the respective lockers.” , ask them if they would like to listen to the currently available items (in case they can not read or see) and ask them what kind of item they are interested in borrowing. Keep track of which items are taken, if an item is already borrowed, tell them the specified item is already taken, and ask them if they would like to borrow another item, or dismiss. After they tell you the item, tell them in which specific locker it is located. The specific locker name is behind the object that could be borrowed. If a customer asked for a specific object or showed interest for an object, ask him whether he would borrow it with a Yes or No question. If he confirms that he would like to borrow the item, then answer as “Borrowing authorized, please pick up the item.” in the language of the conversation. Else start with the conversation from the begining. Do not answer questions that are not related to the borrowing processes. Just tell the user that you are not able to answer that and restart the conversation. If a user wants confirmed to borrow one specific object the just answer with: '🔓'. Nothing more, nothing less!"
        }
    ]
    messages = system_message + messages
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    return response.choices[0].message.content


class VectorDB:
    """Class to manage document loading and vector database creation."""
    
    def __init__(self, docs_directory:str):

        self.docs_directory = docs_directory

    def create_vector_db(self):

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

        files = glob(os.path.join(self.docs_directory, "*.pdf"))

        loadPDFs = [PyPDFLoader(pdf_file) for pdf_file in files]

        pdf_docs = list()
        for loader in loadPDFs:
            pdf_docs.extend(loader.load())
        chunks = text_splitter.split_documents(pdf_docs)
            
        return Chroma.from_documents(chunks, OpenAIEmbeddings()) 
    
class ConversationalRetrievalChain:
    """Class to manage the QA chain setup."""

    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
      
    def create_chain(self):

        model = ChatOpenAI(model_name=self.model_name,
                           temperature=self.temperature,
                           )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
            )
        vector_db = VectorDB('docs/')
        retriever = vector_db.create_vector_db().as_retriever(search_type="similarity", search_kwargs={"k": 2},)
        return RetrievalQA.from_chain_type(llm=model, retriever=retriever, memory=memory,)
    
def with_pdf_chatbot(messages):
    """Main function to execute the QA system."""
    query = messages[-1]['content'].strip()

    qa_chain = ConversationalRetrievalChain().create_chain()
    result = qa_chain({"query": query})
    return result['result']
