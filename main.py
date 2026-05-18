from rich.console import Console
from cli_app.command_handler import handle_command
from utils.logger import logger
import os
import sys
from fastapi import FastAPI
import uvicorn


def run_web():
    """A tiny web wrapper so Render stays happy."""
    app = FastAPI()

    @app.get("/")
    def health_check():
        return {"status": "EDA Explorer is running"}

    @app.get("/cmd")
    def api_handle(command: str):
        # This calls your existing logic exactly like the CLI would
        return {"output": handle_command(command)}

    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

console = Console()
from vector_store.instruction_embedder import embed_analyze_instructions

# Ensure analyze instructions are embedded
def init_vector_store():
    return embed_analyze_instructions()

def run_cli():
    embed_analyze_instructions()  # Pre-embed instructions for analysis commands

    console.print("\n[bold cyan]EDA Explorer[/bold cyan]")
    console.print("Type 'exit' to quit\n")

    while True:

        try:

            cmd = console.input("[bold yellow]> [/bold yellow]")

            result = handle_command(cmd)

            if result == "exit":
                console.print("Exiting...")
                break

            if result:
                console.print(result)

        except KeyboardInterrupt:
            console.print("\nInterrupted by user")
            break

        except Exception as e:
            logger.error(f"CLI error | {e}")
            console.print(f"Error: {e}")


if __name__ == "__main__":
    if "PORT" in os.environ:
        run_web()
    else:
        # Otherwise, run your normal CLI behavior
        run_cli()
    # run_cli()