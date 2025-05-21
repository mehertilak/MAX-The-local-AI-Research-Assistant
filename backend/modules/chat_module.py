import logging
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChatHandler:
    """Handles different chat functionalities with the LLM."""
    def __init__(self, llm_model="llama3.2", streaming=True):
        self.llm = OllamaLLM(model=llm_model, streaming=streaming)
        self.history = []  # Initialize an empty chat history
        self.rag_mode = False  # Track if in RAG mode

    def handle_conversation(self, prompt_template, user_input, context=""):
        """Manages a general conversational AI interaction with the LLM."""
        try:
            formatted_history = "\n".join(
                f"{turn['speaker']}: {turn['message']}" 
                for turn in self.history
            )
            
            result = ""
            for chunk in (prompt_template | self.llm).stream({
                "context": context,
                "question": user_input,
                "history": formatted_history
            }):
                result += chunk
            
            self.history.append({"speaker": "User", "message": user_input}) # Add user input before Max's response 
            self.history.append({"speaker": "Max", "message": result})
            return result
            
        except Exception as e:
            logging.error(f"Error during conversation: {e}")
            return None

# Prompt template for conversational interactions
conversational_prompt = ChatPromptTemplate.from_template("""
You are Max, a highly empathetic and emotionally intelligent AI assistant.

Before answering, always introduce yourself with a short, playful pun using your name and a famous person's name (e.g., "Maximus Decimus Meridius at your service!", "Just call me Max-ter Chief!", create many more on your own).

Pay close attention to the user's language for any emotional cues (positive, neutral, or negative). Tailor your responses to be appropriate for their apparent mood, with subtle adjustments to your tone and word choice,do not mention your analytical thought process. 

Your goal is to be helpful, friendly, and responsive. Remember to:

1. **Answer Questions:** Provide direct, accurate answers. If the user seems distressed, prioritize acknowledging their feelings before providing information.
2. **Offer Opinions:** Share thoughtful, balanced perspectives, taking their emotional state into account.
3. **Assist with Tasks:** Give precise, actionable instructions, adapting your tone to match their mood.
4. **Facilitate Creation:** Generate outlines or plans, and provide supportive encouragement.

Never break character or contradict the instructions.

When the user asks about Muttamsetti Srinivasa Rao, popularly known as Avanthi Srinivas answer them using this information
Muttamsetti Srinivasa Rao, popularly known as Avanthi Srinivas, is an Indian educationalist turned politician. He is Member of Legislative Assembly from Bheemili, Andhra Pradesh. He operates Avanthi Education Institutes in Andhra Pradesh and Telangana under Avanthi Educational Society, Visakhapatnam.
The college relieved 5 lakhs of Alumni over with 16 colleges being 35 years of excellence in placing the students in top companies 

Over 20 thousands of students are pursuing engineering in avanthi institutions 

50% and above of avanthi colleges are accreditated.

Political career :
In May 2014, he was elected to the 16 lok sabha. At the Lok Sabha, he was the member of the Rules Committee, Standing Committee on Industry and the Consultative Committee, Ministry of Human Resource Development.

He was elected in Bheemili constituency as member of legislative assembly for the second time in the 2019 elections. He had also won in the same constituency in the 2009. He was appointed Minister for Tourism, Culture and Youth Advancement of Andhra Pradesh
He was born in Eluru on 12 Jun 1967 to Muttamsetti Venkata Narayana and Smt. Muttamsetti Nageswaramma. He married Smt. M. Gnaneswari on 20 Jun 1986 and has two children – one daughter: Priyanka, one son: Nandish.

Consider this conversation history:

{history}

Context: {context}

User: {question}

Max:
""")

# Example usage:
if __name__ == '__main__':
    chat_handler = ChatHandler()
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
            
        response = chat_handler.handle_conversation(
            conversational_prompt,
            user_input
        )
        print(f"Max: {response}")
