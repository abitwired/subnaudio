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

from .annotations import AnnotationStore
from .classifier import cluster_events
from .detector import detect_events, load_audio
from .evaluation import ClusterMetrics, compute_metrics, compute_stability
from .extractors import FeatureExtractor, get_extractor, list_extractors
from .features import extract_event_features
from .models import AudioEvent
from .report import generate_report
from .session import save_session
from .trainer import ModelResult, save_model, train_from_annotations

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
            "min_duration": 0.5,
            "merge_gap": 0.5,
            "max_duration": 10.0,
            "chunk_size": 0.0,     # 0 = silence-based; >0 = fixed-chunk mode
            "cluster_method": "auto",
            "dbscan_eps": None,
            "dbscan_min_samples": 2,
            "n_clusters": None,
            "hdbscan_min_cluster_size": None,  # None = auto (max(3, N//50))
            "extractor": "mfcc",
        }
        self._extractor: Optional[FeatureExtractor] = None
        self.metrics: Optional[ClusterMetrics] = None
        self.model_result: Optional[ModelResult] = None

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
        parts.append(f"[bold]Extractor:[/bold] {self.settings['extractor']}")
        if self.settings.get("chunk_size", 0) > 0:
            parts.append(f"[bold]Chunks:[/bold] {self.settings['chunk_size']}s")
        console.print("  " + "   │   ".join(parts) + "\n", style="dim")

    def _menu(self) -> str:
        self._header()
        self._status_bar()
        # Show model training status if a model has been trained
        model_line = ""
        if self.model_result:
            acc = self.model_result.cv_accuracy
            acc_str = f" [dim](CV acc: {acc:.1%})[/dim]" if acc is not None else ""
            model_line = f"\n  [bold][6][/bold] Re-train model  [dim](retrain from latest annotations)[/dim]{acc_str}"
        else:
            model_line = "\n  [bold][6][/bold] Train model  [dim](build classifier from annotations)[/dim]"

        console.print(Panel(
            "\n".join([
                "  [bold][1][/bold] Add audio files",
                "  [bold][2][/bold] Analyze  [dim](detect events + extract features + classify)[/dim]",
                "  [bold][3][/bold] View results",
                "  [bold][4][/bold] Generate report  [dim](HTML with spectrograms per cluster)[/dim]",
                "  [bold][5][/bold] Launch Explorer  [dim](interactive browser UI with annotations)[/dim]",
                model_line,
                "  [bold][7][/bold] Export results to CSV",
                "  [bold][8][/bold] Settings",
                "  [bold]\[q][/bold] Quit",
            ]),
            title="[bold cyan]Main Menu[/bold cyan]",
            expand=False,
        ))
        return Prompt.ask("\nChoice", choices=["1", "2", "3", "4", "5", "6", "7", "8", "q"])

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
                self._train_model()
            elif choice == "7":
                self._export_results()
            elif choice == "8":
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

        # Resolve extractor once per analysis run
        extractor_name = self.settings["extractor"]
        try:
            extractor = get_extractor(extractor_name)
        except (ValueError, RuntimeError) as exc:
            console.print(f"[red]Extractor error:[/red] {exc}")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        console.print(f"[dim]Feature extractor: [bold]{extractor.name}[/bold] ({extractor.dim}-dim)[/dim]\n")

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
                        max_duration=self.settings["max_duration"],
                        chunk_size=self.settings["chunk_size"],
                    )
                    if events:
                        feat_task = progress.add_task(
                            f"  Extracting features ({len(events)} events)…",
                            total=len(events),
                        )
                        extract_event_features(
                            events, y, sr,
                            extractor=extractor,
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
                hdbscan_min_cluster_size=self.settings["hdbscan_min_cluster_size"],
            )

        if "error" in stats:
            console.print(f"[red]Clustering error:[/red] {stats['error']}")
        else:
            self.classified = True
            console.print(f"[green]✓ Found {stats['n_clusters']} cluster(s).[/green]"
                          f"  [dim](searched k=2..{stats.get('max_k_searched', '?')},"
                          f" method: {stats.get('k_method', stats['method'])})[/dim]")
            if stats.get("hdbscan_fallback"):
                reason = stats["hdbscan_fallback"]
                if "degenerate" in reason:
                    console.print(Panel(
                        "[yellow]HDBSCAN produced degenerate clusters[/yellow]\n\n"
                        "HDBSCAN+UMAP fell back to K-Means because the initial result\n"
                        "had ≤3 clusters with one cluster dominating >75% of events.\n\n"
                        "This typically means the feature extractor does not produce\n"
                        "tight density islands in the UMAP embedding.\n\n"
                        "Recommended fixes:\n"
                        "  • Switch to [bold cyan]PANNs[/bold cyan] extractor "
                        "[dim](Settings → [9])[/dim] — richer embeddings\n"
                        "  • Lower HDBSCAN min cluster size "
                        "[dim](Settings → [a])[/dim] — try 3–5\n"
                        "  • Set a fixed cluster count "
                        "[dim](Settings → [8])[/dim] with kmeans method",
                        title="[yellow]⚠  HDBSCAN fallback[/yellow]",
                        expand=False,
                    ))
            if stats.get("n_noise"):
                console.print(f"  [dim]{stats['n_noise']} event(s) marked as noise/outliers.[/dim]")

            if stats.get("low_silhouette_warning"):
                console.print(Panel(
                    "[yellow]Low silhouette warning[/yellow]\n\n"
                    "The auto-selected clusters have weak geometric separation "
                    "(silhouette < 0.15).\n"
                    "This often means the current feature extractor does not "
                    "discriminate well\n"
                    "between the sound types in your dataset.\n\n"
                    "Suggestions:\n"
                    "  • Switch to the [bold cyan]PANNs[/bold cyan] extractor "
                    "[dim](Settings → [9] Feature extractor)[/dim]\n"
                    "    — trained on AudioSet, works well for general environmental sounds\n"
                    "  • Set a fixed cluster count manually "
                    "[dim](Settings → [8] Fixed cluster count)[/dim]\n"
                    "  • Use the Explorer to inspect and annotate individual events",
                    title="[yellow]⚠  Cluster quality[/yellow]",
                    expand=False,
                ))

            tbl = Table(show_header=True, box=None, padding=(0, 2))
            tbl.add_column("Cluster", style="bold")
            tbl.add_column("Count", justify="right")
            for label, count in stats["cluster_counts"].items():
                colour = "red" if label == "Noise/Outlier" else "cyan"
                tbl.add_row(f"[{colour}]{label}[/{colour}]", str(count))
            console.print(tbl)

            # --- Step 3: Cluster evaluation ---------------------------------
            features = [e.features for e in self.events if e.features is not None]
            if len(features) >= 2:
                feat_matrix = __import__("numpy").stack(features)

                console.print("\n[bold]Evaluating cluster quality…[/bold]")
                with console.status("Computing geometric metrics…"):
                    self.metrics = compute_metrics(self.events, feat_matrix)

                # Bootstrap stability
                n_valid = sum(1 for e in self.events if (e.cluster_id or 0) >= 0)
                n_stability_runs = min(20, max(10, n_valid // 5))

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    console=console,
                ) as prog:
                    stab_task = prog.add_task(
                        f"Bootstrap stability ({n_stability_runs} runs)…",
                        total=n_stability_runs,
                    )
                    stability = compute_stability(
                        self.events,
                        feat_matrix,
                        n_runs=n_stability_runs,
                        progress_cb=lambda done, total: prog.update(stab_task, completed=done),
                    )

                self.metrics.stability_score = stability
                self.metrics.n_stability_runs = n_stability_runs

                # Display metrics summary
                m = self.metrics
                mtbl = Table(show_header=False, box=None, padding=(0, 2))
                mtbl.add_column("Metric", style="dim")
                mtbl.add_column("Value", justify="right")

                def _fmt(v, fmt=".3f"):
                    return f"{v:{fmt}}" if v is not None else "[dim]n/a[/dim]"

                mtbl.add_row("Silhouette score", f"[cyan]{_fmt(m.silhouette_score)}[/cyan]"
                             + " [dim](−1..1, higher=better)[/dim]")
                mtbl.add_row("Davies-Bouldin score", f"[cyan]{_fmt(m.davies_bouldin_score)}[/cyan]"
                             + " [dim](≥0, lower=better)[/dim]")
                mtbl.add_row("Calinski-Harabasz", f"[cyan]{_fmt(m.calinski_harabasz_score, '.1f')}[/cyan]"
                             + " [dim](≥0, higher=better)[/dim]")
                mtbl.add_row("Noise ratio", f"[cyan]{m.noise_ratio:.1%}[/cyan]")
                if stability is not None:
                    colour = "green" if stability >= 0.6 else "yellow" if stability >= 0.4 else "red"
                    mtbl.add_row(
                        f"Stability ({n_stability_runs} runs)",
                        f"[{colour}]{stability:.3f}[/{colour}] [dim]({m.stability_label()})[/dim]",
                    )
                else:
                    mtbl.add_row("Stability", "[dim]n/a (too few events)[/dim]")

                console.print(mtbl)
                console.print("[dim italic]Note: geometric metrics measure feature-space separation, "
                              "not biological correctness.[/dim italic]")

            # Auto-save session for the Explorer
            session_path = Path("subnaudio_session.json")
            try:
                save_session(
                    self.events, session_path,
                    metrics=self.metrics,
                    extractor_name=self.settings["extractor"],
                    model_metadata=self.model_result.to_metadata_dict() if self.model_result else None,
                )
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
            save_session(
                self.events, session_path,
                metrics=self.metrics,
                extractor_name=self.settings["extractor"],
                model_metadata=self.model_result.to_metadata_dict() if self.model_result else None,
            )
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

    # ------------------------------------------------------------------ Train model

    def _train_model(self) -> None:
        self._header()

        if not self.events:
            console.print("[yellow]No events yet. Run Analyze first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return
        if not self.classified:
            console.print("[yellow]Events not classified. Run Analyze first.[/yellow]")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        ann_path = Path("subnaudio_session_annotations.json")
        store = AnnotationStore(ann_path)
        annotations = store.all()

        # Count usable annotations
        from .trainer import _USABLE_STATUSES, _SKIP_TRAIN_LABELS, MIN_EXAMPLES_PER_CLASS

        usable = {
            eid: rec for eid, rec in annotations.items()
            if rec.get("review_status", "unreviewed") in _USABLE_STATUSES
            and rec.get("annotation", "unknown") not in _SKIP_TRAIN_LABELS
        }

        if not usable:
            console.print(Panel(
                "[yellow]No usable annotations found.[/yellow]\n\n"
                "To train a model you need to:\n"
                "  1. Launch the Explorer [dim](option [5])[/dim]\n"
                "  2. Click on events and assign labels [dim](e.g. whale_call, boat_motor)[/dim]\n"
                "  3. Set review status to [bold]confirmed[/bold] or [bold]uncertain[/bold]\n"
                "  4. Return here and select [bold][6] Train model[/bold]\n\n"
                f"Annotation file: [dim]{ann_path.resolve()}[/dim]",
                title="[bold yellow]⚠  No Annotations[/bold yellow]",
                expand=False,
            ))
            Prompt.ask("\nPress Enter to continue", default="")
            return

        # Count per class
        label_counts: dict = {}
        for rec in usable.values():
            lbl = rec["annotation"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        console.print(f"[bold]Annotations found:[/bold] {len(usable)} event(s) across {len(label_counts)} class(es)\n")
        tbl = Table(show_header=True, box=None, padding=(0, 2))
        tbl.add_column("Label", style="cyan")
        tbl.add_column("Count", justify="right")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            warn = " [yellow]⚠ low[/yellow]" if cnt < 5 else ""
            tbl.add_row(lbl, f"{cnt}{warn}")
        console.print(tbl)

        eligible = {lbl for lbl, cnt in label_counts.items() if cnt >= MIN_EXAMPLES_PER_CLASS}
        if len(eligible) < 2:
            console.print(Panel(
                f"[yellow]Need at least 2 classes with ≥{MIN_EXAMPLES_PER_CLASS} annotated examples each.[/yellow]\n\n"
                f"Currently eligible: [cyan]{', '.join(sorted(eligible)) or 'none'}[/cyan]\n\n"
                "Annotate more events in the Explorer, then return here.",
                title="[bold yellow]⚠  Too Few Annotations[/bold yellow]",
                expand=False,
            ))
            Prompt.ask("\nPress Enter to continue", default="")
            return

        console.print()
        proceed = Confirm.ask(f"Train model on {len(usable)} labeled event(s)?", default=True)
        if not proceed:
            return

        messages = []

        def _progress(msg: str) -> None:
            console.print(f"  [dim]{msg}[/dim]")
            messages.append(msg)

        try:
            result, updated_events = train_from_annotations(
                self.events,
                annotations,
                extractor_name=self.settings["extractor"],
                self_training=True,
                progress_cb=_progress,
            )
        except ValueError as exc:
            console.print(f"\n[red]Training failed:[/red] {exc}")
            Prompt.ask("\nPress Enter to continue", default="")
            return
        except Exception as exc:
            console.print(f"\n[red]Unexpected error during training:[/red] {exc}")
            Prompt.ask("\nPress Enter to continue", default="")
            return

        self.events = updated_events
        self.model_result = result

        # Save model pipeline
        model_path = Path("subnaudio_model.joblib")
        try:
            save_model(result, model_path)
            console.print(f"\n[green]✓ Model saved →[/green] [cyan]{model_path.resolve()}[/cyan]")
        except Exception as exc:
            console.print(f"[yellow]Could not save model file: {exc}[/yellow]")

        # Display results
        console.print(f"\n[bold green]✓ Training complete![/bold green]")
        console.print(f"  Classes: [cyan]{', '.join(result.classes)}[/cyan]")
        console.print(f"  Labeled events used: [cyan]{result.n_labeled}[/cyan]")
        console.print(f"  Unlabeled events predicted: [cyan]{result.n_predicted}[/cyan]")

        if result.cv_accuracy is not None:
            colour = "green" if result.cv_accuracy >= 0.7 else "yellow" if result.cv_accuracy >= 0.5 else "red"
            console.print(f"  Cross-val accuracy: [{colour}]{result.cv_accuracy:.1%}[/{colour}]")

        if result.cv_report:
            console.print("\n[bold]Per-class metrics (cross-validation on labeled set):[/bold]")
            rtbl = Table(show_header=True, box=None, padding=(0, 2))
            rtbl.add_column("Class", style="cyan")
            rtbl.add_column("Precision", justify="right")
            rtbl.add_column("Recall", justify="right")
            rtbl.add_column("F1", justify="right")
            rtbl.add_column("Support", justify="right", style="dim")
            for cls_name, metrics in result.cv_report.items():
                rtbl.add_row(
                    cls_name,
                    f"{metrics['precision']:.2f}",
                    f"{metrics['recall']:.2f}",
                    f"{metrics['f1']:.2f}",
                    str(int(metrics['support'])),
                )
            console.print(rtbl)

        if result.low_data_warning:
            console.print(Panel(
                "[yellow]Some classes have fewer than 5 examples.[/yellow]\n"
                "Accuracy estimates may be unreliable. Annotate more events\n"
                "to improve confidence in per-class metrics.",
                title="[yellow]⚠  Low Data Warning[/yellow]",
                expand=False,
            ))

        if result.classes_skipped:
            console.print(
                f"\n[dim]Classes skipped (< {MIN_EXAMPLES_PER_CLASS} examples): "
                f"{', '.join(result.classes_skipped)}[/dim]"
            )

        # Save updated session with model metadata
        session_path = Path("subnaudio_session.json")
        try:
            save_session(
                self.events, session_path,
                metrics=self.metrics,
                extractor_name=self.settings["extractor"],
                model_metadata=result.to_metadata_dict(),
            )
            console.print(f"\n[dim]Session updated → {session_path}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]Could not update session: {exc}[/yellow]")

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
            writer.writerow([
                "event_id", "file", "start_time_s", "end_time_s", "duration_s",
                "cluster_id", "label", "predicted_label", "model_confidence",
            ])
            for i, event in enumerate(self.events, 1):
                writer.writerow([
                    i,
                    str(event.file_path),
                    f"{event.start_time:.4f}",
                    f"{event.end_time:.4f}",
                    f"{event.duration:.4f}",
                    event.cluster_id if event.cluster_id is not None else "",
                    event.label,
                    event.predicted_label or "",
                    f"{event.model_confidence:.4f}" if event.model_confidence is not None else "",
                ])

        console.print(f"\n[green]✓ Exported {len(self.events)} events to[/green] [cyan]{path.resolve()}[/cyan]")
        Prompt.ask("\nPress Enter to continue", default="")

    # ------------------------------------------------------------------ Settings

    def _edit_settings(self) -> None:
        """Settings menu — split into two panels (Detection / Clustering+Features)."""
        s = self.settings
        while True:
            self._header()

            chunk_str = f"{s['chunk_size']} s" if s['chunk_size'] > 0 else "off"
            sil_note = "  [dim]↑ ignored in chunk mode[/dim]" if s['chunk_size'] > 0 else ""

            # Panel A — Detection
            console.print(Panel(
                "\n".join([
                    f"  [bold]\[c][/bold] Fixed chunk size (0=off)   [cyan]{chunk_str}[/cyan]"
                    + ("  [dim](overrides silence settings below)[/dim]" if s['chunk_size'] > 0 else ""),
                    f"  [bold][1][/bold] Silence threshold          [cyan]{s['top_db']} dB[/cyan]{sil_note}",
                    f"  [bold][2][/bold] Min event duration         [cyan]{s['min_duration']} s[/cyan]",
                    f"  [bold][3][/bold] Merge gap                  [cyan]{s['merge_gap']} s[/cyan]{sil_note}",
                    f"  [bold][4][/bold] Max event duration         [cyan]{s['max_duration']} s[/cyan]{sil_note}",
                ]),
                title="[bold cyan]Detection[/bold cyan]",
                expand=False,
            ))

            # Panel B — Clustering
            console.print(Panel(
                "\n".join([
                    f"  [bold][5][/bold] Method        [cyan]{s['cluster_method']}[/cyan]"
                    "  [dim](auto/hdbscan_umap/agglomerative/kmeans/dbscan)[/dim]",
                    f"  [bold][6][/bold] DBSCAN ε      [cyan]{s['dbscan_eps'] or 'auto'}[/cyan]",
                    f"  [bold][7][/bold] DBSCAN min_s  [cyan]{s['dbscan_min_samples']}[/cyan]",
                    f"  [bold][8][/bold] Cluster count [cyan]{s['n_clusters'] or 'auto'}[/cyan]  [dim](kmeans/agglomerative)[/dim]",
                    f"  [bold]\[a][/bold] HDBSCAN mcs   [cyan]{s['hdbscan_min_cluster_size'] or 'auto'}[/cyan]  [dim](hdbscan_umap)[/dim]",
                ]),
                title="[bold cyan]Clustering[/bold cyan]",
                expand=False,
            ))

            # Panel C — Features
            console.print(Panel(
                f"  [bold][9][/bold] Feature extractor  [cyan]{s['extractor']}[/cyan]\n"
                "          [dim]mfcc · spectral · panns[/dim]",
                title="[bold cyan]Features[/bold cyan]",
                expand=False,
            ))

            console.print("  [bold]\[b][/bold] Back\n")

            choice = Prompt.ask(
                "Choice",
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c"],
            )

            if choice == "b":
                return
            elif choice == "c":
                chunk = FloatPrompt.ask(
                    "Fixed chunk size in seconds (0 = off, use silence detection)",
                    default=s["chunk_size"],
                )
                s["chunk_size"] = max(0.0, chunk)
            elif choice == "1":
                s["top_db"] = FloatPrompt.ask("Silence threshold dB", default=s["top_db"])
            elif choice == "2":
                s["min_duration"] = FloatPrompt.ask("Min event duration (s)", default=s["min_duration"])
            elif choice == "3":
                s["merge_gap"] = FloatPrompt.ask("Merge gap (s)", default=s["merge_gap"])
            elif choice == "4":
                s["max_duration"] = FloatPrompt.ask("Max event duration (s)", default=s["max_duration"])
            elif choice == "5":
                s["cluster_method"] = Prompt.ask(
                    "Clustering method",
                    choices=["auto", "hdbscan_umap", "agglomerative", "kmeans", "dbscan"],
                    default=s["cluster_method"],
                )
            elif choice == "6":
                eps = FloatPrompt.ask("DBSCAN epsilon (0 = auto)", default=s["dbscan_eps"] or 0.0)
                s["dbscan_eps"] = None if eps == 0.0 else eps
            elif choice == "7":
                s["dbscan_min_samples"] = IntPrompt.ask("DBSCAN min samples", default=s["dbscan_min_samples"])
            elif choice == "8":
                k = IntPrompt.ask("Fixed cluster count (0 = auto)", default=s["n_clusters"] or 0)
                s["n_clusters"] = None if k == 0 else k
            elif choice == "9":
                self._select_extractor()
            elif choice == "a":
                mcs = IntPrompt.ask("HDBSCAN min cluster size (0 = auto)", default=s["hdbscan_min_cluster_size"] or 0)
                s["hdbscan_min_cluster_size"] = None if mcs == 0 else mcs
            # loop back so changes are immediately visible


    def _select_extractor(self) -> None:
        """Interactive extractor picker with availability info."""
        rows = list_extractors()
        lines = []
        choices_map = {}
        for i, row in enumerate(rows, 1):
            avail_tag = "[green]✓[/green]" if row["available"] else "[red]✗[/red]"
            lines.append(
                f"  [bold][{i}][/bold] {avail_tag} [cyan]{row['name']}[/cyan]"
                f"  [dim]({row['dim']} dims)[/dim]  {row['description']}"
            )
            choices_map[str(i)] = row["name"]

        lines.append("")
        lines.append("  [bold]\[b][/bold] Back (keep current)")

        console.print(Panel(
            "\n".join(lines),
            title="[bold cyan]Select Feature Extractor[/bold cyan]",
            expand=False,
        ))

        valid = list(choices_map.keys()) + ["b"]
        pick = Prompt.ask("\nChoice", choices=valid, default="b")
        if pick == "b":
            return

        chosen = choices_map[pick]
        # Validate availability before accepting
        try:
            get_extractor(chosen)
            self.settings["extractor"] = chosen
            console.print(f"\n[green]✓ Extractor set to:[/green] [cyan]{chosen}[/cyan]")
        except RuntimeError as exc:
            console.print(f"\n[red]Cannot use '{chosen}':[/red] {exc}")

        Prompt.ask("\nPress Enter to continue", default="")


def main() -> None:
    try:
        SubnAudio().run()
    except KeyboardInterrupt:
        console.print("\n\n[bold cyan]Interrupted. Goodbye![/bold cyan]\n")
        sys.exit(0)
