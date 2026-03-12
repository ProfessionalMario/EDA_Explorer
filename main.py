from rich.console import Console
from cli_app.command_handler import handle_command
from utils.logger import logger

console = Console()


def run_cli():

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
    run_cli()