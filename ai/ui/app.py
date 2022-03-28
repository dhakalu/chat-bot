
"""
    Entry point of the application. Use this when you want to quickly test the results 
    of your model through terminal.
"""
from chat import ChatBot
import os

MODEL_LOCATION = "model.pth"
model_path = os.path.join(os.path.dirname(__file__), MODEL_LOCATION)
ChatBot(model_path).start_chat_ui()

