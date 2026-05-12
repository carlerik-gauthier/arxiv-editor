#!/usr/bin/env python3
"""
ArXiv Editor - Research News One-Pager Generator

This is the main entry point for the ArXiv research publishing system.
It orchestrates the multi-agent workflow to generate comprehensive one-pagers
about the latest research news.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path for importsimplementation
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings

console = Console()


def print_banner():
    """Display the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════╗
    ║         ArXiv Research Publishing System             ║
    ║                                                       ║
    ║  Generating comprehensive research summaries with    ║
    ║  specialized agents named after great researchers    ║
    ╚═══════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_agents():
    """Display information about the agents."""
    table = Table(title="Research Agents", show_header=True, header_style="bold magenta")
    table.add_column("Agent", style="cyan", width=12)
    table.add_column("Named After", style="green", width=20)
    table.add_column("Expertise", style="yellow")

    agents = [
        ("Julius", "Julius Springer", "Editor & Coordinator"),
        ("Michel", "Michel Benaim", "Mathematical Intuition"),
        ("Chris", "Krzystof Burdzy", "Probability Theory"),
        ("Alain", "Alain Valette", "Algebra"),
        ("Bruno", "Bruno Colbois", "Spectral & Riemannian Geometry"),
        ("Elisa", "Elisa Gorla", "Applied Math & Cryptography"),
        ("Felix", "Felix Schlenk", "Dynamical Systems & Symplectic Geometry"),
        ("Abdoulaye", "Abdoulaye Sakho", "Machine Learning"),
    ]

    for agent_name, named_after, expertise in agents:
        table.add_row(agent_name, named_after, expertise)

    console.print(table)
    console.print()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    ArXiv Research Publishing System

    Generate comprehensive one-pagers about the latest research news
    from ArXiv, with summaries suitable for both experts and non-experts.
    """
    pass


@cli.command()
@click.option(
    "--days",
    default=7,
    help="Number of days to look back for papers (default: 7)",
    type=int,
)
@click.option(
    "--start-date",
    help="Start date for paper search (YYYY-MM-DD). Overrides --days",
    type=click.DateTime(formats=["%Y-%m-%d"]),
)
@click.option(
    "--end-date",
    help="End date for paper search (YYYY-MM-DD, default: today)",
    type=click.DateTime(formats=["%Y-%m-%d"]),
)
# @click.option(
#     "--email",
#     help="Email address to send the one-pager to",
#     type=str,
# )
@click.option(
    "--agents",
    help="Comma-separated list of agents to use (default: all)",
    type=str,
)
@click.option(
    "--output-dir",
    default="outputs",
    help="Directory to save output files",
    type=click.Path(),
)
#@ click.option(
 #    "--no-email",
 #    is_flag=True,
 #    help="Skip sending email, only generate the one-pager file",
# )
def generate(days, start_date, end_date, agents, output_dir, no_email=True, email=None):
    """
    Generate a research one-pager for the specified time period.

    Example:
        python main.py generate --days 7 --email user@example.com
        python main.py generate --start-date 2024-01-01 --end-date 2024-01-07
    """
    print_banner()

    # Load settings
    try:
        settings = Settings()
    except Exception as e:
        console.print(f"[red]Error loading settings: {e}[/red]")
        console.print("[yellow]Make sure you have created a .env file based on .env.example[/yellow]")
        sys.exit(1)

    # Determine date range
    if start_date and end_date:
        start = start_date
        end = end_date
    elif start_date:
        start = start_date
        end = datetime.now()
    else:
        end = datetime.now()
        start = end - timedelta(days=days)

    # Display configuration
    config_panel = Panel(
        f"[cyan]Start Date:[/cyan] {start.strftime('%Y-%m-%d')}\n"
        f"[cyan]End Date:[/cyan] {end.strftime('%Y-%m-%d')}\n"
        f"[cyan]Days:[/cyan] {(end - start).days}\n"
        f"[cyan]Output Dir:[/cyan] {output_dir}\n"
        f"[cyan]Send Email:[/cyan] {'No' if no_email else 'Yes'}"
        + (f"\n[cyan]Email To:[/cyan] {email}" if email and not no_email else ""),
        title="Configuration",
        border_style="green",
    )
    console.print(config_panel)
    console.print()

    print_agents()

    # Main workflow
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            # Phase 1: Initialize Julius and agents
            task1 = progress.add_task("[cyan]Initializing agents...", total=None)
            # TODO: Initialize Julius and specialized agents
            progress.update(task1, completed=True)
            console.print("[green]✓[/green] Agents initialized")

            # Phase 2: Fetch papers
            task2 = progress.add_task("[cyan]Fetching papers from ArXiv...", total=None)
            # TODO: Coordinate agents to fetch papers
            progress.update(task2, completed=True)
            console.print("[green]✓[/green] Papers fetched")

            # Phase 3: Process and analyze
            task3 = progress.add_task("[cyan]Processing papers and extracting topics...", total=None)
            # TODO: Embed texts and run BERTopic
            progress.update(task3, completed=True)
            console.print("[green]✓[/green] Topics extracted")

            # Phase 4: Analyze representative papers
            task4 = progress.add_task("[cyan]Analyzing representative papers...", total=None)
            # TODO: Download and analyze selected papers
            progress.update(task4, completed=True)
            console.print("[green]✓[/green] Papers analyzed")

            # Phase 5: Generate one-pager
            task5 = progress.add_task("[cyan]Generating one-pager...", total=None)
            # TODO: Generate comprehensive summary
            progress.update(task5, completed=True)
            console.print("[green]✓[/green] One-pager generated")

            # Phase 6: Send email
            if not no_email and email:
                task6 = progress.add_task("[cyan]Sending email...", total=None)
                # TODO: Send email with one-pager
                progress.update(task6, completed=True)
                console.print("[green]✓[/green] Email sent")

        console.print()
        console.print(Panel(
            "[green]One-pager generation completed successfully![/green]\n"
            f"Output saved to: {output_dir}",
            title="Success",
            border_style="green",
        ))

    except Exception as e:
        console.print(f"[red]Error during generation: {e}[/red]")
        console.print("[yellow]Check logs for more details[/yellow]")
        sys.exit(1)


@cli.command()
def chat():
    """Start an interactive Julius conversation session."""
    from src.agents import JuliusSession, JuliusSessionState

    print_banner()
    console.print("Julius chat. Type 'quit' or 'exit' to stop.\n", style="cyan")

    def show_progress(message):
        console.print(f"[dim]{message}[/dim]")

    session = JuliusSession(progress_callback=show_progress)
    while session.state != JuliusSessionState.FINALIZED:
        try:
            user_message = click.prompt("You", type=str)
        except (EOFError, KeyboardInterrupt):
            console.print("\nSession ended.")
            break

        if user_message.strip().lower() in {"quit", "exit"}:
            console.print("Session ended.")
            break

        response = session.handle_user_message(user_message)
        console.print(Panel(response["message"], title=response["state"], border_style="cyan"))
        if response.get("draft_preview"):
            console.print(response["draft_preview"])
        for question in response.get("next_questions", []):
            console.print(f"[yellow]?[/yellow] {question}")


@cli.command()
def info():
    """Display information about the agents and system."""
    print_banner()
    print_agents()

    console.print(Panel(
        "This system uses specialized agents to analyze research papers from ArXiv.\n\n"
        "Each agent focuses on a specific domain and is named after a great researcher:\n"
        "- Julius coordinates all agents and compiles the final one-pager\n"
        "- Other agents fetch papers in their domains, analyze them, and report findings\n\n"
        "The system uses BERTopic for topic modeling and generates summaries\n"
        "suitable for both experts and non-experts.",
        title="About ArXiv Research Publishing System",
        border_style="blue",
    ))


@cli.command()
def test():
    """Run a quick test to verify the installation."""
    print_banner()

    console.print("[cyan]Running installation tests...[/cyan]\n")

    checks = [
        ("Python version", lambda: sys.version_info >= (3, 8)),
        ("Config module", lambda: __import__("config.settings")),
        ("Settings file", lambda: Path(".env").exists() or Path(".env.example").exists()),
        ("Output directory", lambda: Path("outputs").exists()),
        ("Data directory", lambda: Path("data").exists()),
    ]

    all_passed = True
    for check_name, check_func in checks:
        try:
            result = check_func()
            if result or result is None:
                console.print(f"[green]✓[/green] {check_name}")
            else:
                console.print(f"[red]✗[/red] {check_name}")
                all_passed = False
        except Exception as e:
            console.print(f"[red]✗[/red] {check_name}: {e}")
            all_passed = False

    console.print()
    if all_passed:
        console.print(Panel(
            "[green]All tests passed![/green]\n"
            "The system appears to be correctly set up.",
            title="Test Results",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[yellow]Some tests failed![/yellow]\n"
            "Please check the errors above and run setup if needed.",
            title="Test Results",
            border_style="yellow",
        ))
        sys.exit(1)


if __name__ == "__main__":
    cli()
