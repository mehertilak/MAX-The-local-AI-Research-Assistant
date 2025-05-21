import logging
import pandas as pd
import re
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_name(question):
    
    name = re.search(r'\b(Mr\.|Ms\.|Mrs\.|who\.|tell\.|talk\.)?\s*([A-Z][a-z]+)\s+([A-Z][a-z]+)\s*([A-Z][a-z]+)?\b', question)
    if name:
        return name.group()  
    else:
        return None

class ChatHandler:
    
    def __init__(self, df, llm_model="llama3.2", streaming=True): # nee model name pettu ko 
        self.llm = OllamaLLM(model=llm_model, streaming=streaming)
        self.df = df
        self.history = []  

    def handle_conversation(self, prompt_template, user_input):
        """Extracts name, retrieves person data, and uses LLM to answer."""
        name = extract_name(user_input)
        if not name:
            return "I could not detect the name from your question."
        
        
        person_data = self.df[self.df['Name of the Faculty'].str.contains(name, case=False, na=False)] 
        
        if person_data.empty:
            return "I am sorry, I can't find any information regarding this person."
        
        
        context = person_data.to_string()

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
            
            self.history.append({"speaker": "User", "message": user_input}) 
            self.history.append({"speaker": "Max", "message": result})
            return result
            
        except Exception as e:
            logging.error(f"Error during conversation: {e}")
            return None
# neeku nachi naa prompt evu 
name_prompt = ChatPromptTemplate.from_template(""" 
You are a helpful AI assistant. You are provided with a question and a small database about a particular person

Use the data to answer the question as accurated and concise as possible

Data: {context}
Question: {question}
""")

if __name__ == '__main__':
    
    csv_file = r"C:\Users\Tilak\Downloads\csdmdatacsv.csv" # nee csv file path chage chesuko 

    try:
        df = pd.read_csv(csv_file, encoding='latin1')  # Naku encoding issues vachavu so chage chesa 
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file, encoding='windows-1252')  # max vaka pothe nee encoding pettu ko 
        except:
           print(f"Error: Could not read CSV file with latin1 or windows-1252 encoding. Try other encodings")
           exit()
    except FileNotFoundError:
        print(f"Error: Could not find CSV file at '{csv_file}'. Please make sure the file exists and the path is correct.")
        exit()
    except Exception as e:
        print(f"Error: Could not read CSV file: {e}")
        exit()

    chat_handler = ChatHandler(df)
    
    while True:
        user_input = input("Ask about someone (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
            
        response = chat_handler.handle_conversation(
            name_prompt,
            user_input
        )
        print(f"Answer: {response}")
