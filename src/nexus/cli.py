"""
CLI for Nexus - interactive chat, server, and training commands.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

app = typer.Typer(
    name="nexus",
    help="Nexus Conversational AI - Command Line Interface",
    add_completion=False,
)
console = Console()


@app.command()
def chat(
    model: str = typer.Option(
        "default",
        "--model", "-m",
        help="Model configuration to use",
    ),
) -> None:
    """Start an interactive chat session."""
    console.print(Panel.fit(
        "[bold cyan]Nexus Conversational AI[/bold cyan]\n"
        "[dim]Type 'quit' or 'exit' to end the conversation[/dim]",
        border_style="cyan",
    ))
    
    asyncio.run(_chat_loop())


async def _chat_loop() -> None:
    """Main chat loop."""
    from nexus.core.engine import ConversationEngine
    
    # Initialize engine
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading AI models...", total=None)
        
        engine = ConversationEngine()
        await engine.initialize()
        
        progress.update(task, description="Ready!")
    
    # Create session
    session = await engine.create_session()
    
    console.print("\n[green]✓[/green] Ready! Start chatting.\n")
    
    while True:
        try:
            # Get user input
            user_input = Prompt.ask("[bold blue]You[/bold blue]")
            
            if user_input.lower() in ("quit", "exit", "bye"):
                console.print("\n[cyan]Goodbye! Have a great day! 👋[/cyan]\n")
                break
            
            if not user_input.strip():
                continue
            
            # Process message
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True,
            ) as progress:
                progress.add_task("Thinking...", total=None)
                response = await engine.process(user_input, session=session)
            
            # Display response
            console.print(f"\n[bold green]Nexus[/bold green]: {response.text}")
            
            # Show suggestions if available
            if response.suggestions:
                console.print(
                    f"[dim]Suggestions: {' | '.join(response.suggestions)}[/dim]"
                )
            
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n\n[cyan]Session ended. Goodbye![/cyan]\n")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {str(e)}[/red]\n")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
) -> None:
    """Start the API server."""
    import uvicorn
    
    console.print(Panel.fit(
        f"[bold cyan]Starting Nexus API Server[/bold cyan]\n"
        f"[dim]Host: {host}:{port}[/dim]",
        border_style="cyan",
    ))
    
    uvicorn.run(
        "nexus.api.app:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
    )


@app.command()
def train(
    data_path: str = typer.Option(
        "data/training.json",
        "--data", "-d",
        help="Path to training data",
    ),
    output_dir: str = typer.Option(
        "models",
        "--output", "-o",
        help="Output directory for trained model",
    ),
    epochs: int = typer.Option(10, "--epochs", "-e", help="Number of training epochs"),
) -> None:
    """Train or fine-tune the model."""
    console.print(Panel.fit(
        "[bold cyan]Model Training[/bold cyan]\n"
        f"[dim]Data: {data_path}[/dim]\n"
        f"[dim]Output: {output_dir}[/dim]",
        border_style="cyan",
    ))
    
    asyncio.run(_train_model(data_path, output_dir, epochs))


async def _train_model(data_path: str, output_dir: str, epochs: int) -> None:
    """Run model training."""
    from nexus.training.trainer import ModelTrainer
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing trainer...", total=None)
        
        trainer = ModelTrainer(output_dir=output_dir)
        await trainer.load_data(data_path)
        
        progress.update(task, description=f"Training for {epochs} epochs...")
        metrics = await trainer.train(epochs=epochs)
        
        progress.update(task, description="Complete!")
    
    # Display results
    table = Table(title="Training Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in metrics.items():
        table.add_row(key, f"{value:.4f}" if isinstance(value, float) else str(value))
    
    console.print(table)


@app.command()
def analyze(
    text: str = typer.Argument(..., help="Text to analyze"),
) -> None:
    """
    Analyze text for intent, entities, and sentiment.
    """
    asyncio.run(_analyze_text(text))


async def _analyze_text(text: str) -> None:
    """Run text analysis."""
    from nexus.core.engine import ConversationEngine
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading models...", total=None)
        
        engine = ConversationEngine()
        await engine.initialize()
        
        progress.update(task, description="Analyzing...")
        
        # Run NLU
        intent = await engine._intent_classifier.classify(text)
        entities = await engine._entity_extractor.extract(text)
        sentiment, score = await engine._sentiment_analyzer.analyze(text)
    
    # Display results
    console.print(Panel(f"[bold]{text}[/bold]", title="Input", border_style="blue"))
    
    # Intent
    table = Table(title="Intent Classification")
    table.add_column("Intent", style="cyan")
    table.add_column("Confidence", style="green")
    table.add_row(intent.name, f"{intent.confidence:.2%}")
    console.print(table)
    
    # Entities
    if entities:
        table = Table(title="Extracted Entities")
        table.add_column("Text", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Confidence", style="green")
        
        for entity in entities:
            table.add_row(entity.text, entity.type, f"{entity.confidence:.2%}")
        
        console.print(table)
    else:
        console.print("[dim]No entities detected[/dim]")
    
    # Sentiment
    table = Table(title="Sentiment Analysis")
    table.add_column("Sentiment", style="cyan")
    table.add_column("Score", style="green")
    table.add_row(sentiment.value, f"{score:+.2f}")
    console.print(table)


@app.command()
def info() -> None:
    """
    Display information about the Nexus installation.
    """
    from nexus import __version__
    
    table = Table(title="Nexus Conversational AI")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Version", __version__)
    table.add_row("Python Package", "nexus-conversational-ai")
    table.add_row("Documentation", "https://github.com/raghavshuklaofficial/Conversation_AI_Chatbot")
    
    console.print(table)


if __name__ == "__main__":
    app()
