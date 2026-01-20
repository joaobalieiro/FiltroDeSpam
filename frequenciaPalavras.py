from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# regex para capturar palavras com 3+ letras evitando digitos e tokens curtos
WORD_RE = re.compile(r"[A-Za-z]{3,}")


def ensure_nltk_resources() -> None:
    # garante que stopwords esta disponivel no nltk
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


def tokenize(text: str) -> list[str]:
    # extrai tokens alfabeticos e converte para minusculas
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


def load_vocabulary(words_csv: Path, word_col: str) -> list[str]:
    # le o csv de vocabulario e retorna a lista de palavras
    df = pd.read_csv(words_csv, header=0)
    if word_col not in df.columns:
        raise ValueError(f"coluna {word_col} nao existe em {words_csv}")
    return df[word_col].astype(str).tolist()


def infer_label_from_path(file_path: Path, spam_hint: str, ham_hint: str) -> int:
    # infere rotulo pelo nome do arquivo terminando em ham txt ou spam txt
    name = file_path.name.lower()

    if ham_hint and name.endswith(f".{ham_hint.lower()}.txt"):
        return 1

    if spam_hint and name.endswith(f".{spam_hint.lower()}.txt"):
        return -1

    return 0


def build_frequency_rows(
    emails_dir: Path,
    vocab: list[str],
    language: str,
    progress_every: int,
    spam_hint: str,
    ham_hint: str,
) -> list[list[int]]:
    # prepara estruturas para contagem rapida
    vocab_index = {w: i for i, w in enumerate(vocab)}
    lemmatizer = WordNetLemmatizer()
    stopset = set(stopwords.words(language))

    rows: list[list[int]] = []
    processed = 0

    for file_path in sorted(emails_dir.rglob("*")):
        if not file_path.is_file():
            continue

        # ignora arquivos extras do mac que comecam com ._
        if file_path.name.startswith("._"):
            continue

        # processa apenas txt
        if file_path.suffix.lower() != ".txt":
            continue

        # le o arquivo tolerando erros de encoding
        try:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # vetor de contagens por arquivo
        counts = [0] * len(vocab)

        # tokeniza e acumula contagens apenas para palavras do vocabulario
        for tok in tokenize(text):
            if tok in stopset:
                continue
            tok = lemmatizer.lemmatize(tok)
            idx = vocab_index.get(tok)
            if idx is not None:
                counts[idx] += 1

        # define rotulo
        label = infer_label_from_path(file_path, spam_hint, ham_hint)

        # adiciona linha com rotulo no final
        rows.append(counts + [label])

        processed += 1
        if progress_every > 0 and processed % progress_every == 0:
            print(f"processados {processed} arquivos")

    return rows


def write_frequency_csv(out_csv: Path, vocab: list[str], rows: list[list[int]]) -> None:
    header = vocab + ["output"]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="gera matriz de frequencia por email usando um vocabulario fixo")
    parser.add_argument("--words_csv", type=Path, default=Path("listaPalavras.csv"), help="csv com o vocabulario")
    parser.add_argument("--word_col", type=str, default="palavra", help="nome da coluna com as palavras no csv")
    parser.add_argument("--emails_dir", type=Path, default=Path("emails"), help="diretorio com os emails")
    parser.add_argument("--out", type=Path, default=Path("frequencia.csv"), help="caminho do csv de saida")
    parser.add_argument("--lang", type=str, default="english", help="idioma das stopwords")
    parser.add_argument("--spam_hint", type=str, default="spam", help="palavra chave no nome para rotular como spam")
    parser.add_argument("--ham_hint", type=str, default="ham", help="palavra chave no nome para rotular como ham")
    parser.add_argument("--progress_every", type=int, default=100, help="exibe progresso a cada n arquivos 0 desativa")
    args = parser.parse_args()

    ensure_nltk_resources()

    # carrega vocabulario e gera matriz
    t0 = time.time()
    vocab = load_vocabulary(args.words_csv, args.word_col)
    rows = build_frequency_rows(
        emails_dir=args.emails_dir,
        vocab=vocab,
        language=args.lang,
        progress_every=args.progress_every,
        spam_hint=args.spam_hint,
        ham_hint=args.ham_hint,
    )
    write_frequency_csv(args.out, vocab, rows)
    dt = time.time() - t0

    print(f"csv gerado em {args.out.resolve()}")
    print(f"tempo total {dt:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())