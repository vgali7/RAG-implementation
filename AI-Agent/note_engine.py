from llama_index.core.tools import FunctionTool
import os

current_dir = os.getcwd()
note_file = os.path.join(current_dir, "data", "notes.txt")

def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, 'w')
    
    with open(note_file, 'a') as f:
        f.writelines(note + "\n")
    
    return "note saved"

note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name='note_saver',
    description="this tool saves a text based note to a file for the user"
)
