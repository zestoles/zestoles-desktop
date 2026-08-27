"""Colours, banner and the few formatting helpers the terminal shares.

Split out first because every other CLI module needs it and nothing here needs
them back — which makes it the one file in this package that can be imported from
anywhere without thinking about order.
"""

from __future__ import annotations

import os
import sys

from ..autonomy.events import ERROR, SUCCESS, WARN

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
RED = "\033[31m"
GREEN = "\033[32m"

BANNER = f"""{CYAN}{BOLD}
   ██  ██████  ██████ ██    ██ ██ ███████
   ██ ██   ██ ██   ██ ██    ██ ██ ██
   ██ ███████ ██████  ██    ██ ██ ███████
██ ██ ██   ██ ██   ██  ██  ██  ██      ██
 ███  ██   ██ ██   ██   ████   ██ ███████
{RESET}{DIM}   çekirdek S0 · iki katmanlı beyin · /yardim{RESET}
"""

HELP = f"""
{BOLD}Komutlar{RESET}
  {CYAN}/yardim{RESET}            bu liste
  {CYAN}/durum{RESET}             sistem durumu, kota ve son yanıtın künyesi
  {CYAN}/yerel{RESET} <mesaj>     bu mesajı zorla yerel modele sor
  {CYAN}/claude{RESET} <mesaj>    bu mesajı zorla Claude katmanına sor
  {CYAN}/mod{RESET} [auto|local|cloud]   yönlendirme modunu göster veya değiştir
  {CYAN}/hafiza{RESET} [söz]      hafızayı ara, boş bırakırsan durumu gösterir
  {CYAN}/kasa{RESET}              hafıza klasörünü aç (Obsidian'da da açabilirsin)
  {CYAN}/otonom{RESET} [dur|devam|baslat]   otonom çalışma durumu ve kontrolü
  {CYAN}/gorev{RESET} [tür başlık]         görevleri listele veya yeni görev ekle
  {CYAN}/olaylar{RESET}           son aktivite kaydı
  {CYAN}/ajan{RESET} <hedef>      ajan ekibini bir hedefe koş (planla → yap → doğrula)
  {CYAN}/beceri{RESET}            öğrenilmiş tekrar kullanılabilir iş akışları
  {CYAN}/temizle{RESET}           konuşma geçmişini sıfırla
  {CYAN}/kisilik{RESET}           aktif kişilik dosyasının yolunu göster
  {CYAN}/cikis{RESET}             çık   (Ctrl+C de olur)

{DIM}Yönlendirme otomatiktir: kısa sohbet yerel modele, derinlik gerektiren
işler Claude katmanına gider. Neden öyle gittiğini /durum söyler.{RESET}
"""

LEVEL_COLOUR = {ERROR: RED, WARN: YELLOW, SUCCESS: GREEN}


def enable_ansi() -> None:
    """Windows consoles need VT processing switched on before ANSI works."""
    if os.name == "nt":
        os.system("")
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def tier_badge(tier: str, model: str) -> str:
    from ..brain import CLOUD

    colour = BLUE if tier == CLOUD else GREEN
    name = "claude" if tier == CLOUD else "yerel"
    return f"{DIM}{colour}[{name}·{model}]{RESET}"


def format_event(event) -> str:
    colour = LEVEL_COLOUR.get(event.level, DIM)
    return f"{DIM}{event.when}{RESET} {colour}{event.source}·{event.kind}{RESET} {event.message}"
