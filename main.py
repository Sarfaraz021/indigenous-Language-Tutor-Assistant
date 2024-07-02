import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
# from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from prompt import template


class RAGAssistant:
    def __init__(self):
        self.load_env_variables()
        self.setup_prompt_template()
        self.retriever = None  # Define retriever as an instance variable
        self.relative_path = 'data'
        self.filename = 'dummy.txt'
        self.default_documents_directory = os.path.join(
            self.relative_path, self.filename)
        # default_documents_directory = r"D:\DanyAIApp\code\data\dummy.txt"
        self.initialize_retriever(self.default_documents_directory)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

    def load_env_variables(self):
        """Loads environment variables from .env file."""
        load_dotenv('var.env')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    def setup_prompt_template(self):
        """Sets up the prompt template for chat completions."""
        # self.template
        self.prompt_template = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=template,
        )

    def initialize_retriever(self, directory_path):
        """Initializes the retriever with documents from the specified directory path."""
        loader = TextLoader(directory_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        Pinecone(api_key=self.pinecone_api_key, environment='eastus-azure')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    def finetune(self, file_path):
        """Determines the document type and uses the appropriate loader to fine-tune the model."""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        elif file_path.endswith('.csv'):
            loader = CSVLoader(file_path=file_path)
        elif file_path.endswith('.xlsx'):
            loader = UnstructuredExcelLoader(file_path, mode="elements")
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError("Unsupported file type.")

        documents = loader.load_and_split() if hasattr(
            loader, 'load_and_split') else loader.load()

        self.process_documents(documents)

    def process_documents(self, documents):
        """Process and index the documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        Pinecone(api_key=self.pinecone_api_key, environment='eastus-azure')
        vectbd = PineconeVectorStore.from_documents(
            docs, embeddings, index_name=self.pinecone_index_name)
        self.retriever = vectbd.as_retriever()

    def chat(self):
        """Starts a chat session with the AI assistant."""

        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type='stuff',
            retriever=self.retriever,  # Use the instance variable here
            chain_type_kwargs={"verbose": False, "prompt": self.prompt_template,
                               "memory": ConversationBufferMemory(memory_key="history", input_key="question")}
        )

        print("----------RAG Assistant----------\n")
        while True:
            prompt = input("Enter Prompt (or 'exit' to quit): ").strip()
            if prompt.lower() == "exit":
                print("Thanks!! Exiting...")
                break
            else:
                # Use 'chain' instead of 'self.chain'
                assistant_response = chain.invoke(prompt)  # type: ignore
                print(f"AI Assistant: {assistant_response['result']}")
                print("*********************************")

    def start(self):
        """Main function to start the assistant."""
        while True:
            choice = input(
                "\nEnter 1 for Chat Assistant or 2 to Fine-tune Model: ").strip()
            if choice == "1":
                self.chat()
            elif choice == "2":
                path = input(
                    "Enter directory path to fine-tune the Model with new data (Supported:pdfs, word doc, txt, excel and csv files.): ").strip()
                self.finetune(path)
                print(
                    "\nFine-tuning done successfully. You can now chat with the updated Chat Assistant.\n")
            else:
                print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    rag_assistant = RAGAssistant()
    rag_assistant.start()
