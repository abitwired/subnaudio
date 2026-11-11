"""
Subnaudio — Unsupervised classifier for underwater aquatic sounds.
Rich-based TUI front-end.
"""

import csv
import glob as _glob
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from .classifier import cluster_events
from .detector import detect_events, load_audio
from .features import extract_event_features
from .models import AudioEvent
from .report import generate_report
from .session import save_session

console = Console()

# Audio formats supported by soundfile + audioread (via librosa)
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".ogg", ".flac", ".aiff", ".aif",
    ".au", ".caf", ".w64", ".rf64", ".mp4", ".m4a",
}

BANNER = """[bold cyan]
  ███████╗██╗   ██╗██████╗ ███╗   ██╗ █████╗ ██╗   ██╗██████╗ ██╗ ██████╗
  ██╔════╝██║   ██║██╔══██╗████╗  ██║██╔══██╗██║   ██║██╔══██╗██║██╔═══██╗
  ███████╗██║   ██║██████╔╝██╔██╗ ██║███████║██║   ██║██║  ██║██║██║   ██║
  ╚════██║██║   ██║██╔══██╗██║╚██╗██║██╔══██║██║   ██║██║  ██║██║██║   ██║
  ███████║╚██████╔╝██████╔╝██║ ╚████║██║  ██║╚██████╔╝██████╔╝██║╚██████╔╝
  ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝ ╚═════╝
[/bold cyan]
[dim]  Unsupervised classifier for underwater aquatic sounds[/dim]
"""

# Cluster colours for display
CLUSTER_COLOURS = [
    "bright_cyan", "bright_green", "bright_yellow", "bright_magenta",
    "bright_red", "bright_blue", "cyan", "green", "yellow", "magenta",
]


def _cluster_colour(cluster_id: Optional[int]) -> str:
    if cluster_id is None:
        return "dim"
    if cluster_id == -1:
        return "red"
    return CLUSTER_COLOURS[cluster_id % len(CLUSTER_COLOURS)]


class SubnAudio:
    def __init__(self) -> None:
        self.files: List[Path] = []
        self.events: List[AudioEvent] = []
        self.classified = False
        self.settings = {
            "top_db": 40.0,
            "min_duration": 0.1,
            "merge_gap": 0.5,
            "cluster_method": "auto",
            "dbscan_eps": None,  # None = auto-compute from data
            "dbscan_min_samples": 2,
            "n_clusters": None,  # None = auto for K-Means
        }

    # ------------------------------------------------------------------ UI

    def _header(self) -> None:
        console.clear()
        console.print(BANNER)

    def _status_bar(self) -> None:
        parts = [
            f"[bold]Files:[/bold] {len(self.files)}",
            f"[bold]Events:[/bold] {len(self.events)}",
        ]
        if self.classified and self.events:
            ids = {e.cluster_id for e in self.events if e.cluster_id is not None and e.cluster_id >= 0}
            parts.append(f"[bold]Clusters:[/bold] {len(ids)}")
        console.print("  " + "   │   ".join(parts) + "\n", style="dim")

    def _menu(self) -> str:
        self._header()
        self._status_bar()
        console.print(Panel(
            "\n".join([
                "  [bold][1][/bold] Add audio files",
                "  [bold][2][/bold] Analyze  [dim](detect events + extract features + classify)[/dim]",
                "  [bold][3][/bold] View results",
                "  [bold][4][/bold] Generate report  [dim](HTML with spectrograms per cluster)[/dim]",
                "  [bold][5][/bold] Launch Explorer  [dim](interactive browser UI with annotations)[/dim]",
                "  [bold][6][/bold] Export results to CSV",
                "  [bold][7][/bold] Settings",
                "  [bold][q][/bold] Quit",
            ]),
            title="[bold cyan]Main Menu[/bold cyan]",
            expand=False,
        ))
        return Prompt.ask("\nChoice", choices=["1", "2", "3", "4", "5", "6", "7", "q"])

    # ------------------------------------------------------------------ Actions

    def run(self) -> None:
        while True:
            choice = self._menu()
            if choice == "1":
                self._add_files()
            elif choice == "2":
                self._analyze()
            elif choice == "3":
                self._view_results()
            elif choice == "4":
                self._generate_report()
            elif choice == "5":
                self._launch_explorer()
            elif choice == "6":
                self._export_results()
            elif choice == "7":
                self._edit_settings()
            elif choice == "q":
                console.print("\n[bold cyan]🌊  Goodbye![/bold cyan]\n")
                sys.exit(0)

    # ------------------------------------------------------------------ Add files

    def _add_files(self) -> None:
        self._header()
        ext_list = "  ".join(sorted(SUPPORTED_EXTENSIONS))
        console.print(Panel(
            "Enter audio file paths one per line.\n"
            "Glob patterns are supported (e.g. [cyan]/data/*.wav[/cyan]  [cyan]recordings/**/*.mp3[/cyan]).\n"
            "Leave blank and press Enter when done.\n\n"
            f"[dim]Supported formats: {ext_list}[/dim]",
            title="[bold cyan]Add Audio Files[/bold cyan]",
            expand=False,
        ))

        added = 0
        while True:
            raw = Prompt.ask("\nPath (blank to finish)", default="")
            if not raw.strip():
                break
            is_glob = any(c in raw for c in ("*", "?", "["))
            if is_glob:
                matches = _glob.glob(raw, recursive=True)
                expanded = [Path(m) for m in sorted(matches)]
                if not expanded:
                    console.print(f"  [red]✗ No files matched:[/red] {raw}")
                    continue
            else:
                expanded = [Path(raw)]
            for p in expanded:
                if not p.exists():
                    console.print(f"  [red]✗ Not found:[/red] {p}")
                elif p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    console.print(
                        f"  [yellow]✗ Unsupported format[/yellow] [dim]({p.suffix})[/dim]: {p.name}\n"
                        f"    Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
                    )
                elif p.resolve() in self.files:
                    console.print(f"  [dim]Already added:[/dim] {p}")
                else:
                    self.files.append(p.resolve())
                    console.print(f"  [green]✓[/green] {p}")
                    added += 1

        console.print(f"\n[green]Added {added} file(s). Total: {len(self.files)}[/green]")
        Prompt.ask("\nPress Enter to continue", default="")

    # ------------------------------------------------------------------ Analyze

    def _analyze(self) -> None:
        self._header()
        if not self.files:
            console.print("[yellow]No files loaded. Use option 1 first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        console.print(Panel(
            f"Processing [bold]{len(self.files)}[/bold] file(s)…",
            title="[bold cyan]Analyze[/bold cyan]",
            expand=False,
        ))

        # --- Step 1: Detect events & extract features -----------------------
        all_events: List[AudioEvent] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            file_task = progress.add_task("Loading files…", total=len(self.files))

            for file_path in self.files:
                progress.update(file_task, description=f"[cyan]{file_path.name}[/cyan]")
                try:
                    y, sr = load_audio(file_path)
                    events = detect_events(
                        y, sr, file_path,
                        top_db=self.settings["top_db"],
                        min_duration=self.settings["min_duration"],
                        merge_gap=self.settings["merge_gap"],
                    )
                    if events:
                        feat_task = progress.add_task(
                            f"  Extracting features ({len(events)} events)…",
                            total=len(events),
                        )
                        extract_event_features(
                            events, y, sr,
                            progress=lambda done, total: progress.update(feat_task, completed=done),
                        )
                        progress.remove_task(feat_task)
                    all_events.extend(events)
                except Exception as exc:
                    console.print(f"  [red]Error processing {file_path.name}:[/red] {exc}")
                finally:
                    progress.advance(file_task)

        self.events = all_events
        self.classified = False

        console.print(f"\n[green]✓ Detected {len(self.events)} event(s).[/green]")

        if len(self.events) < 2:
            console.print("[yellow]Need at least 2 events to classify.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        # --- Step 2: Cluster ------------------------------------------------
        console.print("\n[bold]Classifying events…[/bold]")
        with console.status("Running clustering…"):
            self.events, stats = cluster_events(
                self.events,
                method=self.settings["cluster_method"],
                dbscan_eps=self.settings["dbscan_eps"],
                dbscan_min_samples=self.settings["dbscan_min_samples"],
                n_clusters=self.settings["n_clusters"],
            )

        if "error" in stats:
            console.print(f"[red]Clustering error:[/red] {stats['error']}")
        else:
            self.classified = True
            console.print(f"[green]✓ Found {stats['n_clusters']} cluster(s).[/green]")
            if stats.get("n_noise"):
                console.print(f"  [dim]{stats['n_noise']} event(s) marked as noise/outliers.[/dim]")

            tbl = Table(show_header=True, box=None, padding=(0, 2))
            tbl.add_column("Cluster", style="bold")
            tbl.add_column("Count", justify="right")
            for label, count in stats["cluster_counts"].items():
                colour = "red" if label == "Noise/Outlier" else "cyan"
                tbl.add_row(f"[{colour}]{label}[/{colour}]", str(count))
            console.print(tbl)

            # Auto-save session for the Explorer
            session_path = Path("subnaudio_session.json")
            try:
                save_session(self.events, session_path)
                console.print(f"\n[dim]Session saved → {session_path}  (use [5] Launch Explorer)[/dim]")
            except Exception as exc:
                console.print(f"[yellow]Could not save session: {exc}[/yellow]")

        Prompt.ask("\nPress Enter to continue", default="")

    # ------------------------------------------------------------------ View results

    def _view_results(self) -> None:
        self._header()
        if not self.events:
            console.print("[yellow]No events yet. Run Analyze first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        # Group events by cluster
        clusters: dict = {}
        for e in self.events:
            cid = e.cluster_id
            clusters.setdefault(cid, []).append(e)

        tbl = Table(
            title="[bold cyan]Detected Events[/bold cyan]",
            show_lines=False,
            expand=False,
        )
        tbl.add_column("#", justify="right", style="dim")
        tbl.add_column("File")
        tbl.add_column("Time range")
        tbl.add_column("Duration", justify="right")
        tbl.add_column("Class", justify="center")

        for i, event in enumerate(self.events, 1):
            colour = _cluster_colour(event.cluster_id)
            tbl.add_row(
                str(i),
                event.file_path.name,
                event.time_range,
                f"{event.duration:.2f}s",
                Text(event.label, style=f"bold {colour}"),
            )

        console.print(tbl)

        if self.classified:
            console.print(f"\n[dim]Total events: {len(self.events)} "
                          f"| Clusters: {len([k for k in clusters if k is not None and k >= 0])}[/dim]")

        Prompt.ask("\nPress Enter to continue", default="")

    # ------------------------------------------------------------------ Report

    def _generate_report(self) -> None:
        self._header()
        if not self.events:
            console.print("[yellow]No events yet. Run Analyze first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return
        if not self.classified:
            console.print("[yellow]Events not yet classified. Run Analyze first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_dir = f"subnaudio_report_{timestamp}"
        out_dir = Path(Prompt.ask("Output directory", default=default_dir))

        console.print(f"\nGenerating spectrograms for [bold]{len(self.events)}[/bold] event(s)…")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Rendering spectrograms…", total=len(self.events))

            index_path = generate_report(
                self.events,
                out_dir,
                progress_cb=lambda done, total: progress.update(task, completed=done),
            )

        console.print(f"\n[green]✓ Report saved to[/green] [cyan]{index_path.resolve()}[/cyan]")
        console.print(f"  Open in a browser: [dim]open {index_path.resolve()}[/dim]")
        Prompt.ask("\nPress Enter to continue", default="")

    # ------------------------------------------------------------------ Explorer

    def _launch_explorer(self) -> None:
        self._header()
        if not self.events:
            console.print("[yellow]No events yet. Run Analyze first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return
        if not self.classified:
            console.print("[yellow]Events not yet classified. Run Analyze first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        session_path = Path("subnaudio_session.json").resolve()
        try:
            save_session(self.events, session_path)
        except Exception as exc:
            console.print(f"[red]Could not save session:[/red] {exc}")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        explorer_script = Path(__file__).parent / "explorer.py"
        env = {**os.environ, "SUBNAUDIO_SESSION": str(session_path)}

        console.print(Panel(
            f"Starting Acoustic Event Explorer…\n\n"
            f"  [bold cyan]http://localhost:8501[/bold cyan]\n\n"
            f"  Session: [dim]{session_path}[/dim]\n\n"
            f"  Press [bold]Ctrl+C[/bold] in this terminal to stop the explorer\n"
            f"  and return to the menu.",
            title="[bold cyan]Launching Explorer[/bold cyan]",
            expand=False,
        ))

        try:
            proc = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", str(explorer_script),
                 "--server.headless", "false", "--browser.gatherUsageStats", "false"],
                env=env,
            )
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
        except FileNotFoundError:
            console.print("[red]streamlit not found.[/red] Install it: [cyan]pip install streamlit[/cyan]")

        console.print("\n[dim]Explorer stopped. Returning to menu.[/dim]")
        Prompt.ask("\nPress Enter to continue", default="")

    # ------------------------------------------------------------------ Export

    def _export_results(self) -> None:
        self._header()
        if not self.events:
            console.print("[yellow]No events to export. Run Analyze first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        default_path = "subnaudio_results.csv"
        out_path = Prompt.ask("Output CSV path", default=default_path)
        path = Path(out_path)

        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["event_id", "file", "start_time_s", "end_time_s", "duration_s", "cluster_id", "label"])
            for i, event in enumerate(self.events, 1):
                writer.writerow([
                    i,
                    str(event.file_path),
                    f"{event.start_time:.4f}",
                    f"{event.end_time:.4f}",
                    f"{event.duration:.4f}",
                    event.cluster_id if event.cluster_id is not None else "",
                    event.label,
                ])

        console.print(f"\n[green]✓ Exported {len(self.events)} events to[/green] [cyan]{path.resolve()}[/cyan]")
        Prompt.ask("\nPress Enter to continue", default="")

    # ------------------------------------------------------------------ Settings

    def _edit_settings(self) -> None:
        self._header()
        s = self.settings

        console.print(Panel(
            "\n".join([
                f"  [bold][1][/bold] Silence threshold (top_db)      [cyan]{s['top_db']} dB[/cyan]",
                f"  [bold][2][/bold] Min event duration               [cyan]{s['min_duration']} s[/cyan]",
                f"  [bold][3][/bold] Merge gap                        [cyan]{s['merge_gap']} s[/cyan]",
                "",
                f"  [bold][4][/bold] Clustering method                [cyan]{s['cluster_method']}[/cyan]  [dim](auto/agglomerative/kmeans/dbscan)[/dim]",
                f"  [bold][5][/bold] DBSCAN epsilon (0 = auto)         [cyan]{s['dbscan_eps'] or 'auto'}[/cyan]",
                f"  [bold][6][/bold] DBSCAN min samples               [cyan]{s['dbscan_min_samples']}[/cyan]",
                f"  [bold][7][/bold] Fixed cluster count (0 = auto)   [cyan]{s['n_clusters'] or 'auto'}[/cyan]",
                "",
                "  [bold][b][/bold] Back",
            ]),
            title="[bold cyan]Settings[/bold cyan]",
            expand=False,
        ))

        choice = Prompt.ask("\nChoice", choices=["1", "2", "3", "4", "5", "6", "7", "b"])

        if choice == "1":
            s["top_db"] = FloatPrompt.ask("Silence threshold (dB below peak)", default=s["top_db"])
        elif choice == "2":
            s["min_duration"] = FloatPrompt.ask("Min event duration (seconds)", default=s["min_duration"])
        elif choice == "3":
            s["merge_gap"] = FloatPrompt.ask("Merge gap (seconds)", default=s["merge_gap"])
        elif choice == "4":
            method = Prompt.ask("Clustering method", choices=["auto", "agglomerative", "kmeans", "dbscan"], default=s["cluster_method"])
            s["cluster_method"] = method
        elif choice == "5":
            eps = FloatPrompt.ask("DBSCAN epsilon (0 = auto)", default=s["dbscan_eps"] or 0.0)
            s["dbscan_eps"] = None if eps == 0.0 else eps
        elif choice == "6":
            s["dbscan_min_samples"] = IntPrompt.ask("DBSCAN min samples", default=s["dbscan_min_samples"])
        elif choice == "7":
            k = IntPrompt.ask("Fixed cluster count (0 = auto)", default=s["n_clusters"] or 0)
            s["n_clusters"] = None if k == 0 else k
        # 'b' — fall through, return to menu


def main() -> None:
    try:
        SubnAudio().run()
    except KeyboardInterrupt:
        console.print("\n\n[bold cyan]Interrupted. Goodbye![/bold cyan]\n")
        sys.exit(0)
