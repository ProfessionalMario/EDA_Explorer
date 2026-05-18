import os
import sys
from rich.console import Console
from cli_app.command_handler import handle_command
from utils.logger import logger
from vector_store.instruction_embedder import embed_analyze_instructions
import os
import gradio as gr
from cli_app.command_handler import handle_command
from vector_store.instruction_embedder import embed_analyze_instructions

console = Console()

def run_web():
    """The 'Space Way': A Gradio interface that acts like your CLI."""
    import gradio as gr
    
    def chat_interface(command):
        # Calls your existing logic exactly like the CLI
        return handle_command(command)

    demo = gr.Interface(
        fn=chat_interface,
        inputs=gr.Textbox(label="EDA Command", placeholder="Type your command here..."),
        outputs=gr.Code(label="Terminal Output", language="markdown"),
        title="EDA Explorer",
        description="Web terminal for EDA Explorer. Type 'help' or your analysis commands."
    )
    # HF Spaces uses port 7860
    demo.launch(server_name="0.0.0.0", server_port=7860)

def run_cli():
    embed_analyze_instructions() 
    console.print("\n[bold cyan]EDA Explorer[/bold cyan]")
    console.print("Type 'exit' to quit\n")

    while True:
        try:
            cmd = console.input("[bold yellow]> [/bold yellow]")
            if cmd.lower() == "exit":
                break
            
            result = handle_command(cmd)
            if result:
                console.print(result)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"CLI error | {e}")
            console.print(f"Error: {e}")

def run_space_interface():
    """The 'Space Way' using ChatInterface."""
    
    # Pre-embed instructions just like your run_cli() does
    embed_analyze_instructions()

    def chat_response(message, history):
        # 'message' is what the user typed in the box
        # We pass it to your existing handler
        result = handle_command(message)
        
        if result == "exit":
            return "Session ended. Refresh the page to restart."
        
        return str(result) if result else "Command executed with no output."

    # ChatInterface is the closest 'look and feel' to a CLI
    demo = gr.ChatInterface(
        fn=chat_response,
        title="EDA Explorer Terminal",
        description="Type your EDA commands below. Works just like the CLI version!",
        examples=["help", "analyze data.csv", "status"] # Optional: suggest commands
    )
    
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    if "SPACE_ID" in os.environ:
        run_space_interface()
    else:
        run_cli()

