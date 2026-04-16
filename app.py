import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_core.messages import trim_messages  # 트리밍 추가
from langchain_community.vectorstores import FAISS  # Chroma → FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory

try:
    api_key = st.secrets['OPENAI_API_KEY']
    os.environ['OPENAI_API_KEY'] = api_key
except:
    load_dotenv('../data/.env')
    api_key = os.getenv('OPENAI_API_KEY')
model = ChatOpenAI(model='gpt-4o', temperature=0)
parser = StrOutputParser()

FAISS_PATH = './faiss_db'

@st.cache_resource
def process_pdf():
    loader = PyPDFLoader('./2024_KB_부동산_보고서_최종.pdf')
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(document)

@st.cache_resource
def initialize_vectorstore():
    embeddings = OpenAIEmbeddings()
    
    # 요구사항 2: 이미 저장된 FAISS가 있으면 load
    if os.path.exists(FAISS_PATH):
        return FAISS.load_local(
            FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    
    # 없으면 새로 만들고 현재 폴더에 저장
    chunk = process_pdf()
    vectorstore = FAISS.from_documents(chunk, embeddings)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore

# 세션별 대화 이력 저장소
chat_history = ChatMessageHistory()

@st.cache_resource
def initialize_chain():
    vectorstore = initialize_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    
    template = """당신은 kb부동산 전문가 입니다 다음 정보를 바탕으로 질문에 답변을 해주세요.
    컨택스트:{context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ('system', template),
        ('placeholder', '{chat_history}'),
        ('human', '{question}')
    ])
    
    def format_docs(docs):
        return '\n\n'.join(doc.page_content for doc in docs)
    
    # 요구사항 3: 최근 4개 메시지(발화)만 유지하는 트리머
    trimmer = trim_messages(
        strategy="last",
        max_tokens=4,
        token_counter=len  # 메시지 개수 기준으로 자름
    )
    
    # 트리밍된 chat_history를 사용하는 체인
    base_chain = (
        RunnablePassthrough.assign(
            chat_history=itemgetter('chat_history') | trimmer,
            context=lambda x: format_docs(retriever.invoke(x['question']))
        )
        | prompt
        | model
        | parser
    )
    
    return RunnableWithMessageHistory(
        base_chain,
        lambda session_id: chat_history,
        input_messages_key='question',
        history_messages_key='chat_history'
    )

def main():
    st.set_page_config(page_title='🏠🏠KB 부동산 AI 상담🏠🏠')
    st.title('🏠🏠KB 부동산 AI 상담🏠🏠')
    st.caption('2024버전입니다')
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    if prompt := st.chat_input('부동산관련 질문을 하세요'):
        with st.chat_message('user'):
            st.markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        chain = initialize_chain()
        
        with st.chat_message('assistant'):
            with st.spinner('답변생성중'):
                response = chain.invoke(
                    {'question': prompt},
                    {'configurable': {'session_id': 'streamlit_session'}}
                )
                st.markdown(response)
        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == '__main__':
    main()
