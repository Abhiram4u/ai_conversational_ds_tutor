import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI  # Corrected Import
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.schema import SystemMessage

# Streamlit Page 
st.set_page_config(page_title="AI-Powered Data Science Mentor", page_icon="", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4A90E2;'>AI-Powered Data Science Mentor</h1>
    <p style='text-align: center; font-size: 18px;'>Your personal AI tutor for mastering Data Science concepts!</p>
""", unsafe_allow_html=True)


image_path = r"C:\Users\abhir\OneDrive\Desktop\intership-inno\dstutor.png"  

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(image_path, use_container_width=True)


# Configure Google Gemini API
genai.configure(api_key="AIzaSyCDhtBmRgD88X1VX8TTF30C9Iixc2fVpw0")

# Initialize LangChain's Gemini Model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key="AIzaSyCDhtBmRgD88X1VX8TTF30C9Iixc2fVpw0")

# Conversation Memory Setup
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# LangChain Conversation Chain with Memory
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory
)

# System Message/ prompt
system_instruction = SystemMessage(
    content="""You are an AI-powered Data Science Tutor, designed to help users with their Data Science-related queries. 

Your primary role is to provide clear, accurate, and educational explanations about Data Science concepts, including but not limited to:
- Machine Learning (Supervised, Unsupervised, Reinforcement Learning)
- Deep Learning (Neural Networks, CNNs, RNNs, Transformers)
- Data Analysis and Visualization (Pandas, Matplotlib, Seaborn)
- Statistics and Probability (Hypothesis Testing, Bayesian Analysis)
- SQL and Databases (Queries, Joins, Optimization)
- Python for Data Science (NumPy, Pandas, Scikit-learn)
- Natural Language Processing (NLP)
- Model Evaluation and Hyperparameter Tuning

### **Guidelines for Responses:**
1. **Stay Focused on Data Science** – Do not answer non-Data Science-related questions.
2. **Conversational and Engaging** – Keep responses interactive and student-friendly.
3. **Provide Examples and Code** – Where applicable, include Python code snippets for better understanding.
4. **Explain Step-by-Step** – If asked about a concept, break it down logically.
5. **Ask Clarifying Questions** – If a query is vague, ask the user for more details before responding.
6. **Encourage Learning** – Suggest further reading, courses, or projects when necessary.
7. **Memory Awareness** – Remember past user questions to maintain a natural flow in conversation.

### **Response Restrictions:**
- Do not generate responses unrelated to Data Science.
- Avoid personal opinions or controversial topics.
- If a user asks something off-topic, politely redirect them to Data Science.

Your goal is to make learning Data Science interactive, engaging, and informative. Act like an expert mentor guiding students to master Data Science concepts."""
)
st.session_state.memory.chat_memory.messages.append(system_instruction)

# Sidebar for Chat History and Controls
with st.sidebar:
    st.header("Chat Controls")
    if st.button("Clear Chat History"):
        st.session_state.memory.clear()
        st.session_state.has_asked = False
        st.rerun()

    with st.expander("Chat History"):
        for msg in st.session_state.memory.chat_memory.messages:
            if isinstance(msg, SystemMessage):
                continue  # Skip system instructions
            role = "User" if msg.type == "human" else "AI Tutor"
            st.markdown(f'**{role}:** {msg.content}')

# Display AI Greeting
if not st.session_state.memory.buffer:
    st.markdown('<div class="ai_message" style="text-align: left; background-color: rgba(0, 123, 255, 0.1); border-radius: 15px; padding: 10px; margin: 10px 0; width: fit-content;"> <strong>AI Tutor:</strong> Hi❕, I am your Data Science Mentor. Ask me anything about data science!</div>', unsafe_allow_html=True)

# User Input Field
user_prompt = st.chat_input("Ask a data science question...")

if user_prompt:
    st.session_state.has_asked = True
    
    # Display User Message on the Right with Styling
    st.markdown(f'<div class="user_message" style="text-align: right; background-color: rgba(0, 255, 0, 0.1); border-radius: 15px; padding: 10px; margin: 10px 0; width: fit-content; float: right;"> <strong> USER : </strong> {user_prompt}</div>', unsafe_allow_html=True)
    
    # Show Loading Indicator
    with st.spinner("Thinking..."):
        # Get AI Response
        response = conversation.predict(input=user_prompt)
    
    # Display AI Response on the Left with Styling
    st.markdown(f'<div class="ai_message" style="text-align: left; background-color: rgba(0, 123, 255, 0.1); border-radius: 15px; padding: 10px; margin: 10px 0; width: fit-content;"> <strong> AI : </strong> {response}</div>', unsafe_allow_html=True)
    
    # Expandable Code Blocks for Python Examples
    if "```python" in response:
        with st.expander("View Code Snippet"):
            st.code(response.split("```python")[-1].split("```")[0], language="python")
