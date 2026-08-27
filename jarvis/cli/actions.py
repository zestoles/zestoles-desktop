"""Long-running operations the terminal can start, with their progress shown live.

Each one subscribes to the event log for the duration of the run and unsubscribes
in a `finally`. That matters more than it looks: a subscription left behind after a
failed run keeps printing another run's events into a prompt that has moved on.
"""

from __future__ import annotations

from ..research import refusal_report
from .theme import BOLD, DIM, GREEN, RED, RESET, YELLOW, format_event


def run_orchestration(agents, goal: str) -> None:
    """Run the agent team on a goal, with the team's own events shown live."""
    if agents is None:
        print(f"{YELLOW}ajan sistemi açık değil{RESET}\n")
        return

    def show(event) -> None:
        if event.source == "agent":
            print("  " + format_event(event), flush=True)

    unsubscribe = agents.events.subscribe(show)
    print()
    try:
        run = agents.run(goal)
    except Exception as exc:  # noqa: BLE001 - a failed run must not end the session
        print(f"{RED}orkestrasyon hata verdi: {exc}{RESET}\n")
        return
    finally:
        unsubscribe()

    print(f"\n{BOLD}Sonuç{RESET}  {DIM}{run.summary()}{RESET}")
    if run.plan_source != "planlayıcı":
        print(f"{DIM}plan kaynağı: {run.plan_source}"
              f"{f' ({run.skill_name})' if run.skill_name else ''}{RESET}")
    print()
    print(run.output or f"{DIM}(çıktı yok){RESET}")

    if run.verdict:
        colour = GREEN if run.verdict.ok else YELLOW
        print(f"\n{colour}{run.verdict.summary()}{RESET}")
        for issue in run.verdict.blocking[:5]:
            print(f"  {DIM}· {issue}{RESET}")
        if not run.verdict.ok:
            print(f"{DIM}bu çıktı doğrulanmadı — gerçek diye kabul etme{RESET}")
    print()


def run_research(research, question: str) -> None:
    if research is None:
        print(f"{YELLOW}araştırma sistemi açık değil{RESET}\n")
        return

    def show(event) -> None:
        if event.source == "research":
            print("  " + format_event(event), flush=True)

    unsubscribe = research.events.subscribe(show)
    print()
    try:
        report = research.investigate(question)
    except Exception as exc:  # noqa: BLE001
        print(f"{RED}araştırma hata verdi: {exc}{RESET}\n")
        return
    finally:
        unsubscribe()

    if report.error:
        print(f"{RED}{report.error}{RESET}\n")
        return

    print(f"\n{BOLD}Cevap{RESET}  {DIM}{report.summary()}{RESET}\n")
    print(report.answer or report.synthesis)

    if report.claims:
        print(f"\n{BOLD}İddia denetimi{RESET}")
        for claim in report.claims:
            colour = {"dogrulandi": GREEN, "celiskili": YELLOW}.get(claim.status, DIM)
            print(f"  {colour}●{RESET} {claim.text[:100]}")
            print(f"      {DIM}{claim.summary()}{RESET}")
            for source in claim.supported_by[:3]:
                print(f"      {DIM}↳ {source.domain} · {source.tier}{RESET}")
            for source in claim.contradicted_by[:2]:
                print(f"      {YELLOW}↳ çelişiyor: {source.domain}{RESET}")

    if report.injection_sources:
        print(f"\n{YELLOW}Bu kaynaklarda talimat enjeksiyonu denendi ve engellendi:{RESET}")
        for url in report.injection_sources:
            print(f"  {DIM}· {url}{RESET}")

    if report.failures:
        print(f"\n{DIM}okunamayan kaynaklar:{RESET}")
        for failure in report.failures[:4]:
            print(f"  {DIM}· {failure[:140]}{RESET}")

    if report.note_title:
        print(f"\n{GREEN}hafızaya yazıldı: {report.note_title}{RESET}")
    refused = refusal_report(report.admission)
    if refused:
        print(f"\n{DIM}{refused}{RESET}")
    print()
