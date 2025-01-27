from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

model = OllamaLLM(model="./Ollama/Instruction Bot/fine_tuned_llama3")

template = """
    Answer the question below:
    Here is the conversation history: {chat_history}
    Question: {question}

    Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

print("Bot: Welcome! I am your product instructions chatbot. How can I assist you?")

def conversation():
    context = "Bot: Welcome! I am your product instructions chatbot. How can I assist you?"
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break
        result = chain.invoke({"chat_history": context, "question": user_input})
        print("Bot:", result)
        context += f"\nUser: {user_input}\nBot: {result}"

conversation()