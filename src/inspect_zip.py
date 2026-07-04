"""
Inspeciona o ZIP oficial do material suplementar do U-RNN (Figshare) SEM extrair
o arquivo inteiro.

Motivo: o ZIP oficial (`data/raw/urbanflood24_api.zip`) tem ~18 GB compactado e
expande para ~116 GB, o que inviabiliza a extração completa no ambiente local
usado nesta atividade. Este script usa `zipfile.ZipFile` para ler apenas o
índice central do ZIP (lista de nomes, tamanhos, datas) sem descompactar
nenhum conteúdo, permitindo auditar o dataset oficial de forma leve.

Saída: outputs/zip_inventory.txt com um resumo do conteúdo do arquivo.

Este script NUNCA extrai o ZIP completo. Se encontrar candidatos pequenos e
úteis para uma amostra pontual, apenas sugere o comando de extração — a
extração real fica a critério do usuário, fora deste script.
"""

import os
import zipfile
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = PROJECT_ROOT / "data" / "raw" / "urbanflood24_api.zip"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "zip_inventory.txt"

# Extensões consideradas "úteis" para inspeção rápida (dados tabulares, metadados,
# imagens e arrays numéricos comuns em datasets de nowcasting/hidrologia).
USEFUL_EXTENSIONS = {
    ".npy", ".npz", ".csv", ".json", ".txt", ".png", ".tif", ".tiff", ".h5", ".yaml", ".yml",
}

# Acima deste tamanho (bytes) um arquivo não é sugerido para extração pontual,
# mesmo que a extensão seja "útil" — o objetivo é evitar reintroduzir o problema
# de espaço em disco que motivou este script.
MAX_SAMPLE_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

# Quantos arquivos mostrar na listagem "primeiros arquivos do ZIP".
PREVIEW_COUNT = 40


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} PB"


def inspect_zip() -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("INVENTARIO DO ZIP OFICIAL - U-RNN (material suplementar / Figshare)")
    lines.append("=" * 70)
    lines.append(f"Arquivo: {ZIP_PATH.relative_to(PROJECT_ROOT)}")

    if not ZIP_PATH.exists():
        lines.append("")
        lines.append("STATUS: ZIP NAO ENCONTRADO em data/raw/urbanflood24_api.zip.")
        lines.append(
            "Este projeto nao depende do ZIP oficial para funcionar: o pipeline "
            "principal (src/train_demo.py) usa dados sinteticos (src/synthetic_data.py)."
        )
        return "\n".join(lines)

    compressed_size_on_disk = ZIP_PATH.stat().st_size
    lines.append(f"Tamanho no disco (compactado): {human_size(compressed_size_on_disk)}")
    lines.append("")
    lines.append("Abrindo indice central do ZIP (sem extrair conteudo)...")

    # ZipFile le apenas o "central directory" no fim do arquivo para montar
    # infolist() -- nao descompacta nada até que .read()/.extract() seja chamado.
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        infos = zf.infolist()

        total_compressed = sum(i.compress_size for i in infos)
        total_uncompressed = sum(i.file_size for i in infos)
        n_files = sum(1 for i in infos if not i.is_dir())
        n_dirs = sum(1 for i in infos if i.is_dir())

        lines.append(f"Total de entradas no ZIP: {len(infos)} ({n_files} arquivos, {n_dirs} pastas)")
        lines.append(f"Tamanho total compactado (somado por entrada): {human_size(total_compressed)}")
        lines.append(f"Tamanho total estimado apos extracao completa: {human_size(total_uncompressed)}")
        lines.append("")

        # Distribuição de extensões
        ext_counter: Counter = Counter()
        ext_size: Counter = Counter()
        for info in infos:
            if info.is_dir():
                continue
            ext = os.path.splitext(info.filename)[1].lower() or "(sem extensao)"
            ext_counter[ext] += 1
            ext_size[ext] += info.file_size

        lines.append("-" * 70)
        lines.append("DISTRIBUICAO POR EXTENSAO (contagem e tamanho extraido)")
        lines.append("-" * 70)
        for ext, count in ext_counter.most_common(30):
            lines.append(f"  {ext:>15}  {count:>8} arquivos  {human_size(ext_size[ext]):>12}")

        lines.append("")
        lines.append("-" * 70)
        lines.append(f"PRIMEIROS {PREVIEW_COUNT} ARQUIVOS DO ZIP (ordem do indice)")
        lines.append("-" * 70)
        for info in infos[:PREVIEW_COUNT]:
            tag = "DIR " if info.is_dir() else "FILE"
            lines.append(f"  [{tag}] {info.filename}  ({human_size(info.file_size)})")

        lines.append("")
        lines.append("-" * 70)
        lines.append("ARQUIVOS COM EXTENSOES UTEIS ENCONTRADOS (.npy/.npz/.csv/.json/.txt/.png/.tif/...)")
        lines.append("-" * 70)
        useful = [
            i for i in infos
            if not i.is_dir() and os.path.splitext(i.filename)[1].lower() in USEFUL_EXTENSIONS
        ]
        lines.append(f"Total de arquivos com extensoes uteis: {len(useful)}")
        for info in useful[:PREVIEW_COUNT]:
            lines.append(f"  {info.filename}  ({human_size(info.file_size)})")
        if len(useful) > PREVIEW_COUNT:
            lines.append(f"  ... e mais {len(useful) - PREVIEW_COUNT} arquivo(s) nao listados.")

        lines.append("")
        lines.append("-" * 70)
        lines.append("SUGESTAO DE AMOSTRA PONTUAL (arquivos pequenos, <= 5MB, extensao util)")
        lines.append("-" * 70)
        small_candidates = [i for i in useful if i.file_size <= MAX_SAMPLE_SIZE_BYTES]
        small_candidates.sort(key=lambda i: i.file_size)
        if small_candidates:
            lines.append(
                "Os arquivos abaixo sao candidatos razoaveis para extracao pontual "
                "(pequenos o suficiente para nao reproduzir o problema de espaco em disco)."
            )
            lines.append("Extracao NAO executada automaticamente. Para extrair manualmente um arquivo especifico:")
            lines.append("")
            for info in small_candidates[:10]:
                lines.append(f"  Candidato: {info.filename}  ({human_size(info.file_size)})")
                lines.append(
                    f"    Comando sugerido (execute manualmente, apos revisar): "
                    f"python -c \"import zipfile; "
                    f"zipfile.ZipFile('data/raw/{ZIP_PATH.name}').extract('{info.filename}', 'data/sample/')\""
                )
        else:
            lines.append(
                "Nenhum arquivo pequeno (<= 5MB) com extensao util foi encontrado no indice do ZIP. "
                "Isso e esperado para datasets cientificos volumosos com arrays grandes por amostra."
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("CONCLUSAO: nenhuma extracao foi realizada por este script. O ZIP permanece")
    lines.append("integro em data/raw/ apenas como referencia. A solucao funcional desta")
    lines.append("entrega usa dados sinteticos (src/synthetic_data.py), conforme justificado")
    lines.append("no README principal.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = inspect_zip()
    print(report)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"\nResumo salvo em: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
