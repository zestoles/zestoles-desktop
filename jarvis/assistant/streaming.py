"""Akışkan cevap çözümleyicisi: ham JSON'dan cümleleri erkeden çıkarır.

Beyin her kararı tek bir şemalı JSON içinde döndürür; nihai cevap da bir karar:

    {"action": "reply", "message": "<kullanıcıya cevap>"}

Bu sınıf o JSON'un *token akışına* bağlanır ve en üst seviyedeki `message`
değeri büyürken tamamlanan cümleleri hemen verir -- ses katmanı, model daha
yazarken konuşmaya başlar. İkinci bir model çağrısı yok, protokol değişmez:
beslenen parçalar birleştiğinde ortaya chat()'ın döndürdüğü metnin kendisi
çıkar.

## Neden karakter karakter tarayıcı

Naif bir `"message":"..."` araması yanlış güvende bırakır: bir aracın
argümanları, kendi içinde `"action": "reply", "message": ...` geçen serbest
metin taşıyabilir; yanlış tetiklenen erken cümle, hiç istenmemiş bir cevabın
başını okutmak olur. Tarayıcı bu yüzden yapıyı bilir: dizge içi/dışı, kaçış,
yuva derinliği. `message` yalnızca derinlik 1'de yakalanır ve ateşleme kapısı
ikilidir -- önce `action` değerinin gerçekten `"reply"` olduğu görülür, ondan
sonra tek bir cümle dışarı çıkar. Araç kararı veren bir akıştan bu boru sessiz
geçer.

## Muhafazakar sınır

Cümle sonu yalnızca noktalama + boşluk ikilisinden sayılır -- bu tek başına
"3.14" gibi ondalıkları zaten eler, çünkü orada boşluk yoktur. Kısaltma ve baş
harf kuralları şüpheli sınırı *atlar*: içerik beklemede kalır, sonraki gerçek
sınırda daha uzun bir parça olarak çıkar. Yanlış bölünen cümle kulakta anında
fark edilir; şüphede dur, söyleme.
"""

from __future__ import annotations

import re

#: Cümle sonu adayı: noktalama ve arkasından gelen boşluk.
_BOUNDARY = re.compile(r"(?<=[.!?…])[ \n\r\t]+")

#: Sınır öncesi son sözcük bunlardansa cümle bitmemiştir.
_ABBREVIATIONS = frozenset({
    "vb", "vs", "dr", "prof", "doç", "av", "sn", "bkz", "örn", "yy", "no",
    "mr", "mrs", "st", "etc", "ör", "bk", "çev", "haz", "yay",
})

_SIMPLE_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
                   "n": "\n", "r": "\r", "t": "\t"}


class ReplySentenceStream:
    """Ham JSON parçalarını yutar; tamamlanan cevap cümlelerini döndürür."""

    def __init__(self) -> None:
        self.depth = 0
        self.in_string = False
        self.escape = False
        #: \uXXXX kaçışının bekleyen onaltılık gövdesi.
        self._escape_body = ""
        #: Derinlik 1'de sıradaki şeyin ne olduğu: key/keyname/colon/value/literal/comma
        self.expect = "key"
        self.key = ""              # son okunan en üst seviye anahtar adı
        self._key_buffer = ""
        #: \uXXXX kaçışının bekleyen hane sayısı ve gövdesi.
        self._unicode_left = 0
        self._escape_body = ""
        #: Değer dizgesi toplanırken hangi anahtar için: "", "action", "message"
        self._capturing = ""
        self._action_value = ""
        self.is_reply = False      # action değeri "reply" olarak görüldü
        self.done = False          # message kapanandı
        self.decoded = ""          # message'ın çözülmüş birikimi
        self.emitted = 0           # decoded içinden cümlesi ateşlenmiş uzunluk

    @property
    def message_open(self) -> bool:
        return self._capturing == "message"

    @property
    def text(self) -> str:
        """Şimdiye kadar çözülmüş mesaj; test ve teşhis için."""
        return self.decoded

    # ------------------------------------------------------------------ feed
    def feed(self, chunk: str) -> list[str]:
        """Bir ham parça yut; bununla TAMAMLANAN cümleleri döndür."""
        for ch in chunk:
            self._char(ch)
        return self._fire_ready()

    def finish(self) -> list[str]:
        """Akış bitti: bekleyen kuyruğu son parça olarak ver."""
        rest = self.decoded[self.emitted:].strip()
        self.emitted = len(self.decoded)
        return [rest] if rest else []

    # ------------------------------------------------------- durum makinesi
    def _char(self, ch: str) -> None:
        if self.in_string:
            self._string_char(ch)
            return
        if ch == '"':
            self.in_string = True
            self.escape = False
            self._escape_body = ""
            if self.depth == 1 and self.expect == "value":
                self._begin_value_string()
            elif self.depth == 1 and self.expect == "key":
                self._key_buffer = ""
                self.expect = "keyname"
            return
        if ch in "{[":
            self.depth += 1
            self.expect = "key"
            return
        if ch in "}]":
            self.depth -= 1
            self.expect = "comma"
            return
        if self.depth != 1:
            return  # iç yuvalardaki virgül/işaretler bizi ilgilendirmez
        if ch == ":" and self.expect == "colon":
            self.expect = "value"
            return
        if ch == "," and self.expect in ("comma", "literal"):
            self.expect = "key"
            return

    def _string_char(self, ch: str) -> None:
        """Dizge içindeyiz: kaçışı çöz, gerekiyorsa değeri biriktir."""
        if self.escape:
            self.escape = False
            if ch == "u":
                self._escape_body = ""
                self._unicode_left = 4
                return
            self._emit_char(_SIMPLE_ESCAPES.get(ch, ""))
            return
        if self._unicode_left:
            self._escape_body += ch
            self._unicode_left -= 1
            if not self._unicode_left:
                try:
                    self._emit_char(chr(int(self._escape_body, 16)))
                except ValueError:
                    pass
                self._escape_body = ""
            return
        if ch == "\\":
            self.escape = True
            return
        if ch == '"':
            self.in_string = False
            self._string_closed()
            return
        self._emit_char(ch)

    def _emit_char(self, ch: str) -> None:
        if self._capturing == "message":
            self.decoded += ch
        elif self._capturing == "action":
            self._action_value += ch
        elif self.expect == "keyname":
            self._key_buffer += ch

    def _string_closed(self) -> None:
        capturing, self._capturing = self._capturing, ""
        if capturing == "message":
            self.done = True
            self.expect = "comma"
            return
        if capturing == "action" and self.depth == 1:
            self.is_reply = self._action_value.strip().lower() == "reply"
            self._action_value = ""
            self.expect = "comma"
            return
        if self.depth == 1 and self.expect == "keyname":
            self.key = self._key_buffer
            self._key_buffer = ""
            self.expect = "colon"

    # ------------------------------------------------------- değer yakalama
    def _begin_value_string(self) -> None:
        if self.key == "message":
            self._capturing = "message"
        elif self.key == "action":
            self._capturing = "action"
            self._action_value = ""

    # ------------------------------------------------------------- cümleler
    def _fire_ready(self) -> list[str]:
        if not (self.is_reply and (self.message_open or self.done)):
            return []
        out: list[str] = []
        search_from = 0
        while True:
            tail = self.decoded[self.emitted:]
            match = _BOUNDARY.search(tail, search_from)
            if not match:
                break
            head = tail[:match.end()]
            if self._false_boundary(head):
                # Yalnızca ARAMA ilerler; içerik hâlâ beklemede ve sonraki
                # gerçek sınırla birlikte daha uzun bir parça olarak çıkar.
                search_from = match.end()
                continue
            piece = head.strip()
            if piece:
                if out and len(piece) < 3:
                    out[-1] = f"{out[-1]} {piece}"
                else:
                    out.append(piece)
            self.emitted += len(head)
        return out

    @staticmethod
    def _false_boundary(head: str) -> bool:
        """Kısaltma / baş harf: burada bölünmez.

        Sıra önemli: önce boşluk, sonra noktalama soyulur -- tersi olsaydı
        sondaki boşluk soyulmayı durdurur ve hiçbir kısaltma yakalanmazdı.
        """
        word = head.rstrip().rstrip(".!?…").rstrip(" \")']").rsplit(" ", 1)[-1].lower()
        if word in _ABBREVIATIONS:
            return True
        return len(word) == 1 and word.isalpha()


__all__ = ["ReplySentenceStream"]
