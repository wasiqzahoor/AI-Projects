import pyttsx3
import speech_recognition as sr
import tkinter as tk
from tkinter import ttk, scrolledtext, simpledialog
import datetime
import webbrowser
import urllib.parse
import wikipedia
import random
import pyjokes
import string

class AssistantGUI: 
    def __init__(self, root):
        self.root = root
        self.root.title("AI Assistant")
        self.root.geometry("800x600")
        
        # Initialize TTS engine
        self.engine = pyttsx3.init()
        self.voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', self.voices[0].id)
        self.engine.setProperty('rate', 180)
        
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        self.chat_only_mode = True
        
        self.create_gui()
        
    def open_website(self, site):
        websites = {
            "github": "https://github.com",
            "linkedin": "https://www.linkedin.com",
            "youtube": "https://www.youtube.com",
            "google": "https://www.google.com",
            "whatsapp": "https://www.whatsapp.com",
            "instagram": "https://www.instagram.com"
        }

        url = websites.get(site.lower(), site)
        webbrowser.open(url)
    
    def create_gui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main chat tab
        self.chat_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chat_tab, text="Chat")
        self.setup_chat_tab()
        
        # Notes tab
        self.notes_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.notes_tab, text="Notes")
        self.setup_notes_tab()
        
        # Reminders tab
        self.reminders_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.reminders_tab, text="Reminders")
        self.setup_reminders_tab()
        
        # Settings tab
        self.settings_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_tab, text="Settings")
        self.setup_settings_tab()
    
    def setup_chat_tab(self):
        # Chat history
        self.chat_history = scrolledtext.ScrolledText(self.chat_tab, wrap=tk.WORD, height=20)
        self.chat_history.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Input area
        input_frame = ttk.Frame(self.chat_tab)
        input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.input_entry = ttk.Entry(input_frame)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind('<Return>', lambda e: self.process_input())
        
        ttk.Button(input_frame, text="Send", command=self.process_input).pack(side=tk.LEFT, padx=2)
        ttk.Button(input_frame, text="🎤", command=self.listen).pack(side=tk.LEFT, padx=2)
        
        # Quick actions
        actions_frame = ttk.LabelFrame(self.chat_tab, text="Quick Actions", padding=5)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        actions = [
            ("YouTube", lambda: self.open_website("https://www.youtube.com/")),
            ("Google", lambda: self.open_website("https://www.google.com/")),
            ("WhatsApp", lambda: self.open_website("https://www.whatsapp.com/")),
            ("Instagram", lambda: self.open_website("https://www.instagram.com/")),
            ("GitHub", lambda: self.open_website("github")),
            ("LinkedIn", lambda: self.open_website("linkedin")),
            ("Tell Joke", lambda: self.speak(pyjokes.get_joke()))
        ]
        
        for text, command in actions:
            ttk.Button(actions_frame, text=text, command=command).pack(side=tk.LEFT, padx=2)
    
    def setup_notes_tab(self):
        self.notes_text = scrolledtext.ScrolledText(self.notes_tab, wrap=tk.WORD)
        self.notes_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        button_frame = ttk.Frame(self.notes_tab)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Save Note", command=self.save_note).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Load Notes", command=self.load_notes).pack(side=tk.LEFT, padx=2)
    
    def setup_reminders_tab(self):
        self.reminder_text = scrolledtext.ScrolledText(self.reminders_tab, wrap=tk.WORD, height=5)
        self.reminder_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        button_frame = ttk.Frame(self.reminders_tab)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(button_frame, text="Set Reminder", command=self.set_reminder).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="View Reminders", command=self.view_reminders).pack(side=tk.LEFT, padx=2)
    
    def setup_settings_tab(self):
        settings_frame = ttk.Frame(self.settings_tab)
        settings_frame.pack(padx=10, pady=10)
        
        # Voice selection
        ttk.Label(settings_frame, text="Voice:").pack(pady=5)
        self.voice_var = tk.StringVar(value="Male")
        voice_combo = ttk.Combobox(settings_frame, textvariable=self.voice_var, 
                                 values=["Male", "Female"], state="readonly")
        voice_combo.pack(pady=5)
        voice_combo.bind('<<ComboboxSelected>>', self.change_voice)
        
        # Speech rate
        ttk.Label(settings_frame, text="Speech Rate:").pack(pady=5)
        self.rate_var = tk.IntVar(value=180)
        rate_scale = ttk.Scale(settings_frame, from_=100, to=300, 
                             variable=self.rate_var, orient=tk.HORIZONTAL)
        rate_scale.pack(fill=tk.X, pady=5)
        rate_scale.bind("<ButtonRelease-1>", self.change_rate)
        
        # Mode selection
        ttk.Label(settings_frame, text="Mode:").pack(pady=5)
        self.mode_var = tk.StringVar(value="Chat")
        ttk.Radiobutton(settings_frame, text="Chat Only", variable=self.mode_var, 
                       value="Chat", command=self.change_mode).pack()
        ttk.Radiobutton(settings_frame, text="Voice Enabled", variable=self.mode_var, 
                       value="Voice", command=self.change_mode).pack()
    
    def speak(self, text):
        self.chat_history.insert(tk.END, f"Assistant: {text}\n")
        self.chat_history.see(tk.END)
        if not self.chat_only_mode:
            self.engine.say(text)
            self.engine.runAndWait()
    
    def listen(self):
        try:
            with sr.Microphone() as source:
                self.speak("Listening...")
                audio = self.recognizer.listen(source)
                text = self.recognizer.recognize_google(audio)
                self.input_entry.delete(0, tk.END)
                self.input_entry.insert(0, text)
                self.process_input()
        except Exception as e:
            self.speak("Sorry, I couldn't understand that.")
    
    def process_input(self):
        query = self.input_entry.get().strip().lower()
        self.input_entry.delete(0, tk.END)
        
        if not query:
            return
        
        self.chat_history.insert(tk.END, f"You: {query}\n")
        
        # Handle different commands
        if 'whatsapp search' in query or 'whatsapp profile' in query:
            number = simpledialog.askstring("WhatsApp", "Enter phone number (with country code):")
            if number and number.isdigit():
                webbrowser.open(f"https://wa.me/{number}")
                self.speak("Opening WhatsApp chat.")
        
        elif 'instagram search' in query or 'insta profile' in query:
            profile = simpledialog.askstring("Instagram", "Enter profile name:")
            if profile:
                webbrowser.open(f"https://www.instagram.com/{profile}")
                self.speak("Opening Instagram profile.")
        
        elif 'watch video' in query or 'search video' in query or 'youtube search' in query or 'i want to watch a video' in query:
            video = simpledialog.askstring("YouTube", "Enter video search:")
            if video:
                webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(video)}")
                self.speak("Searching for videos.")
        
        elif 'open youtube' in query or 'youtube' in query or 'play youtube' in query:
            webbrowser.open("https://youtube.com")
            self.speak("Opening YouTube")
            return

        elif 'open google' in query and len(query.strip().split()) <= 2:
            webbrowser.open("https://google.com")
            self.speak("Opening Google")
            return
        
        elif 'open whatsapp' in query or 'whatsapp web' in query:
            webbrowser.open("https://web.whatsapp.com")
            self.speak("Opening WhatsApp Web")
            return
        
        elif 'I want to ask a question' in query or 'search' in query or 'ask' in query or 'search for' in query:
            search_query = query.replace('search', '').strip()
            if search_query:
                choice = simpledialog.askstring("Search", "Choose source (google/wikipedia):", 
                                              initialvalue="google")
                if choice:
                    if 'wikipedia' in choice.lower():
                        try:
                            result = wikipedia.summary(search_query, sentences=2)
                            self.speak(result)
                        except:
                            self.speak("Couldn't find that on Wikipedia.")
                    else:
                        webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(search_query)}")
                        self.speak("Here's what I found on Google.")
        
        else:
            response = self.get_response(query)
            if response:
                self.speak(response)
    
    def get_response(self, query):
        if "joke" in query:
            return pyjokes.get_joke()
        elif "whom do you admire the most" in query:
            return "I really admire Sir Mumtaz who is a great human being and a wonderful teacher."
        elif "time" in query:
            return datetime.datetime.now().strftime("%I:%M %p")
        elif "date" in query:
            return datetime.datetime.now().strftime("%B %d, %Y")
        elif "hello" in query or "hi" in query:
            return "Hello! How can I help you today?"
        elif "who are you" in query:
            return "I am your AI assistant, here to help you with various tasks."
        elif "how are you" in query:
            return random.choice(["I'm doing great!", "I'm here to help!", "Ready to assist you!"])
        elif "thank" in query:
            return "You're welcome!"
        elif "how are you" in query:
            return random.choice(["I'm doing well, thanks for asking!", "I'm functioning as expected!"])
        elif "who created you" in query or "who made you" in query:
            return "I was created by a smart human like you!"
        elif any(word in query for word in ["hello", "hi", "hey"]):
            return random.choice(["Hello there!", "Hi! How can I assist you today?", "Hey! What can I do for you?"])
        elif any(word in query for word in ["thank you", "thanks"]):
            return random.choice(["You're welcome!", "No problem!", "Glad to help!"])
        elif 'tell me a joke' in query or 'joke' in query:
            return pyjokes.get_joke()
        elif 'i am sad' in query or 'im sad' in query:
            self.speak("I'm sorry to hear that. Here's a joke to cheer you up:")
            return pyjokes.get_joke()
        elif "i love you" in query:
            return random.choice(["Aww, thank you! You're sweet! ❤️", "That's so kind of you! 😊", "I love you too! 💙"])
        elif "weather" in query:
            return "I'm still learning to check the weather, but you can ask me to search for today's forecast."
        elif any(word in query for word in ["bye", "goodbye", "see you"]):
            self.speak("Wait! I'd like to thank Dr Mumtaz and Sir Farhan for his support and guidance.")
            return random.choice(["Goodbye! Talk to you later!", "See you soon! Take care!", "Bye for now!"])
        else:
            #google search
            search_query = query.replace('search', '').strip()
            webbrowser.open(f"https://www.google.com/search?q={urllib.parse.quote(search_query)}")
            self.speak("Here's what I found on Google.")
            
    
    
    # Settings methods
    def change_voice(self, event=None):
        voice_index = 0 if self.voice_var.get() == "Male" else 1
        if voice_index < len(self.voices):
            self.engine.setProperty('voice', self.voices[voice_index].id)
    
    def change_rate(self, event=None):
        self.engine.setProperty('rate', self.rate_var.get())
    
    def change_mode(self):
        self.chat_only_mode = self.mode_var.get() == "Chat"
    
    # Notes methods
    def save_note(self):
        note = self.notes_text.get("1.0", tk.END).strip()
        if note:
            with open("notes.txt", "a") as file:
                file.write(f"[{datetime.datetime.now()}]\n{note}\n\n")
            self.speak("Note saved!")
    
    def load_notes(self):
        try:
            with open("notes.txt", "r") as file:
                self.notes_text.delete("1.0", tk.END)
                self.notes_text.insert("1.0", file.read())
        except FileNotFoundError:
            self.speak("No saved notes found.")
    
    # Reminders methods
    def set_reminder(self):
        reminder = self.reminder_text.get("1.0", tk.END).strip()
        if reminder:
            with open("reminder.txt", "a") as file:
                file.write(f"[{datetime.datetime.now()}]\n{reminder}\n\n")
            self.speak("Reminder set!")
            self.reminder_text.delete("1.0", tk.END)
    
    def view_reminders(self):
        try:
            with open("reminder.txt", "r") as file:
                self.reminder_text.delete("1.0", tk.END)
                self.reminder_text.insert("1.0", file.read())
        except FileNotFoundError:
            self.speak("No reminders found.")

if __name__ == "__main__":
    root = tk.Tk()
    app = AssistantGUI(root)
    root.mainloop()