"""Model-facing access to the local document library."""

from __future__ import annotations

from . import LOW, SERVICES, ToolError, ToolResult, Workspace, tool


def _library():
    library = SERVICES.get("documents")
    if library is None:
        raise ToolError("belge kütüphanesi bu oturumda açık değil")
    return library


@tool("docs.index", risk=LOW, summary="Yerel belgeleri PDF, DOCX ve metin olarak indeksler")
def _index(*, workspace: Workspace, path: str = "") -> ToolResult:
    folder = workspace.resolve(path, must_exist=True) if path else None
    if folder is not None and not folder.is_dir():
        raise ToolError("indekslenecek yol bir klasör olmalı")
    result = _library().index(folder)
    return ToolResult(True, output=(f"{result['dosyalar']} belge, {result['parcalar']} parça "
                                    f"indekslendi · {result['klasor']}"), detail=result)


@tool("docs.search", risk=LOW, summary="İndeksli yerel belgelerde kanıt parçaları arar")
def _search(*, workspace: Workspace, query: str, limit: int = 5) -> ToolResult:
    hits = _library().search(query, limit=limit)
    if not hits:
        return ToolResult(True, output="Belgelerde eşleşme bulunamadı.", detail={"count": 0})
    blocks = [f"[{hit.source} · skor {hit.score:.3f}]\n{hit.text}" for hit in hits]
    return ToolResult(True, output="\n\n".join(blocks),
                      detail={"count": len(hits), "sources": [hit.source for hit in hits]})
