from __future__ import annotations

import argparse
import csv
import re
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# regex para capturar palavras com 3+ letras evitando digitos e tokens curtos
WORD_RE = re.compile(r"[A-Za-z]{3,}")


def iter_text_files(root: Path) -> Iterator[Path]:
    # itera arquivos dentro de um diretorio de forma recursiva ignorando pastas
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def tokenize(text: str) -> list[str]:
    # extrai tokens alfabeticos e converte para minusculas
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def normalize_tokens(
    tokens: Iterable[str],
    stopset: set[str],
    lemmatizer: WordNetLemmatizer
) -> Iterator[str]:
    # remove stopwords e aplica lematizacao simples no padrao noun
    for tok in tokens:
        if tok in stopset:
            continue
        yield lemmatizer.lemmatize(tok)


def count_words(email_dir: Path, language: str, progress_every: int) -> Counter:
    # carrega stopwords e inicializa o lematizador
    stopset = set(stopwords.words(language))
    lemmatizer = WordNetLemmatizer()

    # inicializa contador global e contador de arquivos processados
    counts: Counter = Counter()
    processed = 0

    for file_path in iter_text_files(email_dir):
        # le arquivo tolerando erros de encoding
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # tokeniza normaliza e acumula contagens
        tokens = tokenize(text)
        counts.update(normalize_tokens(tokens, stopset, lemmatizer))

        # mostra progresso a cada n arquivos se habilitado
        processed += 1
        if progress_every > 0 and processed % progress_every == 0:
            print(f"processados {processed} arquivos")

    return counts


def write_csv(counts: Counter, out_path: Path, min_count: int) -> None:
    # filtra por min_count e ordena por frequencia desc com desempate alfabetico
    rows = sorted(
        ((w, c) for w, c in counts.items() if c >= min_count),
        key=lambda x: (-x[1], x[0]),
    )

    # escreve o csv no formato palavra contador
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["palavra", "contador"])
        writer.writerows(rows)


def ensure_nltk_resources() -> None:
    # garante que o pacote stopwords esta disponivel no nltk
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")

    # garante que wordnet e omw estao disponiveis para lematizacao
    try:
        _ = WordNetLemmatizer().lemmatize("cars")
    except LookupError:
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extrai e conta palavras frequentes em um diretório de emails.")
    parser.add_argument("--emails_dir", type=Path, default=Path("C:\\Users\\fonso\\Downloads\\projetosGitHub\\filtroSpam\\emails"), help="Diretório com os emails.")
    parser.add_argument("--out", type=Path, default=Path("listaPalavras.csv"), help="Caminho do CSV de saída.")
    parser.add_argument("--min_count", type=int, default=100, help="Frequência mínima para entrar no CSV.")
    parser.add_argument("--lang", type=str, default="english", help="Idioma das stopwords (ex: english, portuguese).")
    parser.add_argument("--progress_every", type=int, default=100, help="Exibe progresso a cada N arquivos (0 desativa).")
    args = parser.parse_args()

    ensure_nltk_resources()

    t0 = time.time()
    counts = count_words(args.emails_dir, args.lang, args.progress_every)
    write_csv(counts, args.out, args.min_count)
    dt = time.time() - t0

    print(f"CSV gerado em: {args.out.resolve()}")
    print(f"Tempo total: {dt:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
