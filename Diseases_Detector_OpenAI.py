import os

api_key= "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"


import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.chains.conversation.prompt import \
    ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


os.environ['OPENAI_API_KEY'] = api_key
st.title("Diseases Checker Chatbot")

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []


prompt = st.text_input("Enter your data")


symptoms_template = PromptTemplate(
    input_variables=["symptoms"], template=" You are Doctor, I will give you a dictionary mapping from biological and medical concepts to values. Based on my symptoms, please list the top 5 most likely diagnoses ranked most to least likely with a single-line explanation. {symptoms}"
)

diagnosis_template = PromptTemplate(
    input_variables=["diagnosis", "wikipedia_research"], template="Once you determine the most likely diagnosis, you should give description of symptoms write suggest how I can confirm it. Additionally, you should describe the treatment that would be recommended for the most likely steps diagnosis. Finally, you should provide a link from NHS that offers information about the most likely diagnosis {diagnosis} also while leveraging this wikipedia research:{wikipedia_research}.  RESPONSE:"
)

symptoms_memory = ConversationBufferMemory(input_key='symptoms', memory_key='chat_history')
diagnosis_memory = ConversationBufferMemory(input_key='diagnosis', memory_key='chat_history')

llm = OpenAI(temperature=0.9)

if 'entity_memory' not in st.session_state:
            st.session_state.entity_memory = ConversationEntityMemory(llm=llm, k= 3)

Conversation = ConversationChain(
            llm=llm, 
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=st.session_state.entity_memory
        )  

# output = Conversation.run(input=prompt)  
# st.session_state.past.append(prompt)  
# st.session_state.generated.append(output) 

# with st.expander("Conversation", expanded=True):
#     for i in range(len(st.session_state['generated'])-1, -1, -1):
#         st.info(st.session_state["past"][i])
#         st.success(st.session_state["generated"][i])

symptoms_chain = LLMChain(llm=llm, prompt=symptoms_template, verbose=True, output_key='diagnosis', memory=symptoms_memory)
diagnosis_chain = LLMChain(llm=llm, prompt=diagnosis_template, verbose=True, output_key='diseases', memory=diagnosis_memory)

# sequential_chain = SequentialChain(chains=[symptoms_chain, diagnosis_chain], input_variables=['symptoms'], output_variables=['diseases', 'diagnosis'], verbose= True)

wiki = WikipediaAPIWrapper()
if prompt:
    # response = sequential_chain({'symptoms':prompt})
    symptoms = symptoms_chain.run(prompt)
    wikipedia_research = wiki.run(prompt)
    diagnosis  = diagnosis_chain.run(diagnosis=symptoms,  wikipedia_research=wikipedia_research)

    output = Conversation.run(input=prompt)  
    st.session_state.past.append(prompt)  
    st.session_state.generated.append(output) 

    with st.expander("Conversation", expanded=True):
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.success(st.session_state["past"][i])
            # st.success(st.session_state["generated"][i])
            st.info(st.session_state["generated"][i])
            st.info(diagnosis)

    st.write(symptoms)
    st.write(diagnosis)

    # with st.expander('Message History'):
    #     st.info(symptoms_memory.buffer)

    

    with st.expander('Script History'):
        st.info(diagnosis_memory.buffer)
    
    with st.expander("Wikipedia Research"):
        st.info(wikipedia_research)

    