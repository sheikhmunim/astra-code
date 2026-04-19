import os
import sys

# Force UTF-8 output on Windows so Unicode art renders correctly
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import click
from langchain_core.messages import AIMessage

from agent.graph import build_graph
from agent.parser import parse_react_action
from rich.rule import Rule
from rich.table import Table
from cli.interface import console, print_banner, StreamingRenderer
from config.manager import load_config, get_active_provider, get_provider_cfg
from config.providers import build_llm, PROVIDER_DISPLAY
from training.logger import ask_rating, log_example


# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
@click.option("--model", "-m", default=None, help="Override model for this session")
@click.option("--provider", "-p", default=None,
              type=click.Choice(["ollama", "anthropic", "openai", "groq", "minmax"], case_sensitive=False),
              help="Override provider for this session")
@click.option("--show-tools/--no-show-tools", default=True, help="Show tool calls and results")
def cli(ctx, model, provider, show_tools):
    """Astra — local coding agent. Run `astra config` to configure providers."""
    if ctx.invoked_subcommand is None:
        ctx.ensure_object(dict)
        _run_chat(model=model, provider=provider, show_tools=show_tools)


@cli.command("config")
def config_cmd():
    """Configure providers, API keys, and models."""
    from cli.config_ui import run_configure
    run_configure()


# ── Chat session ──────────────────────────────────────────────────────────────

def _run_chat(model=None, provider=None, show_tools=True):
    cwd = os.getcwd()

    # Load config and apply any CLI overrides
    cfg = load_config()
    if provider:
        from config.manager import set_active_provider
        cfg = set_active_provider(cfg, provider)
    if model:
        active_p = get_active_provider(cfg)
        from config.manager import set_provider_field
        cfg = set_provider_field(cfg, active_p, "model", model)

    # Check Ollama is installed and ready (only when using Ollama)
    if get_active_provider(cfg) == "ollama":
        from cli.startup import check_ollama
        if not check_ollama():
            return

    # Build LLM from config
    try:
        llm, tool_mode = build_llm(cfg)
    except Exception as e:
        console.print(f"[red]Failed to initialise model:[/red] {e}")
        console.print("[dim]Run [bold]astra config[/bold] to set up your provider.[/dim]")
        return

    active_provider = get_active_provider(cfg)
    pcfg = get_provider_cfg(cfg, active_provider)
    active_model = pcfg.get("model", "")
    provider_label = PROVIDER_DISPLAY.get(active_provider, active_provider)
    model_label = f"{active_model} ({provider_label})"

    print_banner(model_label, cwd)

    graph, memory_store = build_graph(llm=llm, tool_mode=tool_mode, model_label=model_label, cwd=cwd)
    thread_id = "session-1"
    graph_config = {"configurable": {"thread_id": thread_id}}
    force_plan = False

    while True:
        try:
            console.print(Rule(style="dim cyan"))
            user_input = console.input("[bold cyan]>[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        # ── Slash commands (/ or \ prefix) ───────────────────────────────────
        if user_input.startswith("/") or user_input.startswith("\\"):
            result = _handle_slash(user_input, cfg, cwd, show_tools, memory_store)
            if result == "exit":
                break
            elif result == "force_plan":
                force_plan = True
                console.print("[dim cyan]Planning enabled for next message.[/dim cyan]")
            elif result is not None:
                llm, tool_mode, model_label, graph, memory_store, cfg = result
                graph_config = {"configurable": {"thread_id": "session-1"}}
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            console.print("[dim]Goodbye.[/dim]")
            break

        # ── Agent invocation ─────────────────────────────────────────────────
        state_input = {
            "messages":         [{"role": "user", "content": user_input}],
            "cwd":              cwd,
            "iterations":       0,
            "plan":             "",
            "force_plan":       force_plan,
            "reflection_count": 0,
        }
        force_plan = False  # reset after use

        renderer = StreamingRenderer(show_tools=show_tools)
        renderer.thinking()

        try:
            for msg_chunk, metadata in graph.stream(
                state_input, config=graph_config, stream_mode="messages"
            ):
                node = metadata.get("langgraph_node", "")
                content = getattr(msg_chunk, "content", "") or ""

                # Anthropic returns content as a list of blocks — flatten to string
                if isinstance(content, list):
                    content = "".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                    )

                if not content:
                    continue

                if node == "agent":
                    if isinstance(msg_chunk, AIMessage) and hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                        continue
                    renderer.on_agent_token(content)

                elif node == "tools":
                    obs = content
                    if obs.startswith("Observation: "):
                        obs = obs[13:]
                        tool_name, tool_args = parse_react_action(renderer._full_buf)
                        if tool_name:
                            renderer.on_tool_run(tool_name, tool_args or {})
                        renderer.on_tool_result(obs)
                        renderer.thinking()
                    else:
                        tool_name = getattr(msg_chunk, "name", "tool")
                        renderer.on_tool_run(tool_name, {})
                        renderer.on_tool_result(obs)
                        renderer.thinking()

            renderer.flush()
            _rate_and_log(renderer, user_input, active_model, active_provider)

        except KeyboardInterrupt:
            renderer.flush()
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            renderer.flush()
            console.print(f"[red]Error:[/red] {e}")


# ── Rating + logging ──────────────────────────────────────────────────────────

def _rate_and_log(renderer: StreamingRenderer, user_input: str, model: str, provider: str):
    answer = renderer.final_answer
    if not answer:
        return
    rating = ask_rating()
    if rating is not None:
        log_example(user=user_input, response=answer, rating=rating,
                    model=model, provider=provider)


# ── Slash command handler ─────────────────────────────────────────────────────

def _handle_slash(user_input: str, cfg: dict, cwd: str, show_tools: bool, memory_store=None):
    """
    Handle /commands typed inside the chat.
    Returns:
      "exit"  → break the chat loop
      tuple   → (llm, tool_mode, model_label, graph, memory_store, cfg) after switch
      None    → command handled, continue loop
    """
    parts = user_input.lstrip("/\\").split(None, 1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("exit", "quit"):
        return "exit"

    elif cmd == "help":
        from rich.panel import Panel
        console.print(Panel(
            "[bold]/configure[/bold]              — switch model or provider, set / delete API keys\n"
            "[bold]/status[/bold]                 — show current provider & model\n"
            "[bold]/plan[/bold]                   — force planning for the next message\n"
            "[bold]/remember <fact>[/bold]        — save a fact to long-term memory\n"
            "[bold]/memories[/bold]               — show all memories for this project\n"
            "[bold]/forget[/bold]                 — clear all memories for this project\n"
            "[bold]/help[/bold]                   — show this message\n"
            "[bold]/exit[/bold]                   — quit",
            title="[cyan]Slash Commands[/cyan]",
            border_style="cyan",
            expand=False,
        ))

    elif cmd == "status":
        active = get_active_provider(cfg)
        pcfg = get_provider_cfg(cfg, active)
        console.print(
            f"[dim]Provider:[/dim] [cyan]{PROVIDER_DISPLAY.get(active, active)}[/cyan]  "
            f"[dim]Model:[/dim] [green]{pcfg.get('model', '?')}[/green]"
        )

    elif cmd == "plan":
        return "force_plan"

    elif cmd == "remember":
        if not arg:
            console.print("[yellow]Usage: /remember <fact to remember>[/yellow]")
        elif memory_store and memory_store.ready:
            memory_store.save(arg, category="explicit")
            console.print(f"[green]✓ Remembered:[/green] {arg}")
        else:
            console.print("[yellow]Memory not available — install chromadb: pip install chromadb[/yellow]")

    elif cmd == "memories":
        if not memory_store or not memory_store.ready:
            console.print("[yellow]Memory not available — install chromadb: pip install chromadb[/yellow]")
        else:
            memories = memory_store.list_all()
            if not memories:
                console.print("[dim]No memories yet for this project.[/dim]")
            else:
                table = Table(border_style="cyan", show_lines=True, expand=False)
                table.add_column("Fact", min_width=40)
                table.add_column("Type",  style="dim", width=10)
                table.add_column("Date",  style="dim", width=12)
                for m in memories:
                    table.add_row(m["fact"], m["category"], m["timestamp"])
                console.print(table)

    elif cmd == "forget":
        if memory_store and memory_store.ready:
            memory_store.clear()
            console.print("[green]✓ All memories cleared for this project.[/green]")
        else:
            console.print("[yellow]Memory not available.[/yellow]")

    elif cmd == "configure":
        from cli.config_ui import run_configure
        changed = run_configure()
        if changed is not None:
            new_cfg = load_config()
            return _rebuild(new_cfg, cwd, show_tools)

    else:
        console.print(f"[yellow]Unknown command:[/yellow] /{cmd}  (type /help for list)")

    return None


def _rebuild(cfg: dict, cwd: str, show_tools: bool):
    """Rebuild graph after a provider switch."""
    try:
        llm, tool_mode = build_llm(cfg)
    except Exception as e:
        console.print(f"[red]Failed to switch model:[/red] {e}")
        return None

    active = get_active_provider(cfg)
    pcfg = get_provider_cfg(cfg, active)
    model_label = f"{pcfg.get('model', '')} ({PROVIDER_DISPLAY.get(active, active)})"
    graph, memory_store = build_graph(llm=llm, tool_mode=tool_mode, model_label=model_label, cwd=cwd)
    console.print("[dim]Session reset — new conversation started.[/dim]")
    return llm, tool_mode, model_label, graph, memory_store, cfg


if __name__ == "__main__":
    cli()
