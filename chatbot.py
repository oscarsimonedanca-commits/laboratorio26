# Il mio chatbot online

import streamlit as st
import pdfplumber

# Langchain - insieme di librerie usate per ia generativa
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.header("Assistenza online")

st.image("Chatbot.webp", width=400)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #b0dcdf;
        color: #B1D4D8;
    }
    </style>
    """,
    unsafe_allow_html=True)

with st.sidebar:
    st.title("Il mio documento")
    documento = st.file_uploader("Carica il tuo pdf:", type=["pdf"])

if documento is not None: #none è un operatore buleano cioè non significa nulla - SE QUESTO DOC NON è VUOTO
    with pdfplumber.open(documento) as pdf: #with consente di accatastare una serie di istruzioni e poi esce
        #st.write(f"Pagine totali: {len(pdf.pages)} - Comincio la scansione...")
        testo = ""
        for pagina in pdf.pages:
            testo = testo + pagina.extract_text() + "\n"
            # testo += pagina.extract_text() + "\n"
   # st.write(testo)

    taglierina = RecursiveCharacterTextSplitter( 
        separators=["\n\n", "\n", ". ", " "],
        chunk_size=1000,
        chunk_overlap=200)

    frammenti = taglierina.split_text(testo)
#st.write(f"Totale frammenti creati: {len(frammenti)}")
#st.write(frammenti)

 # Generiamo gli embeddings
    # Puoi cambiare OpenAIEmbeddings e metterne altri
    # https://docs.langchain.com/oss/python/integrations/embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=st.secrets["OPENAI_API_KEY"])
    #st.write("Embedding recuperati!")

    # Salviamo gli embeddings in un vector store o vector db (es. FAISS, Pinecone, etc.)
    vettori = FAISS.from_texts(frammenti, embedding=embeddings) #vettori è un nuovo oggetto / FAISS è un vectorstore che prendo in prestito da facebook

    # Richiesta utente
    domanda_utente = st.text_input("Fai una domanda sul documento caricato:")

# Generazione della risposta in una chain di eventi
# domanda -> embedding -> similarity search -> risultati all'LLM -> risposta.         FARE ATTENZIONE A QUESTO!!! è LA SCALETTA
    def formatta_documento(documenti): #def -> definiamo una funzione che formatta i doc che mettiamo in input e lo fa
        return "\n\n".join([documento.page_content for documento in documenti]) #spaccando ogni documento in documento e il join appicica le varie pag
    
    #al prompt va aggiunto il contesto e la domanda che andiamo a generare

    #DOMANDA 
    prompt = ChatPromptTemplate.from_messages([ #predispone e da un'identià al nostro agente e gli dice dove deve andare a guardare per rispondere
        ("system", #il sistema è il chatbot, lo human siamo noi 
         '''Sei un assistente virtuale. 
    Usa il contesto fornito per rispondere alla domanda in modo conciso. 
    Non accedere a informazioni esterne, come Internet. 
    Se non conosci la risposta, dì semplicemente 'Non lo so'. 
    Contesto:\n{context}'''),
        ("human", "{question}")
        ])

#EMBEDDING 

    comparatore = vettori.as_retriever(  
    #I vettori sono gli embeddings come li abbiamo conservati nel deposito #SIMILARITY SEACH
    # mmr = maximal marginal relevance
        search_type="mmr", 
    # Ritorna i 4 frammenti più simili
        search_kwargs={"k": 4})

    modello_llm = ChatOpenAI( 
        model="gpt-5.4-nano",
        temperature=0.3, #Serve alla creatività del chatbot
        max_tokens=1000, #quanti token mi scrive in uscite
        openai_api_key=st.secrets["OPENAI_API_KEY"])

    catena = ( 
        {"context": comparatore | formatta_documento, 
         "question": RunnablePassthrough()}
        | prompt
        | modello_llm
        | StrOutputParser()
        )
    if domanda_utente:
        risposta = catena.invoke(domanda_utente)
        st.write(risposta)
