from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC


def parse_grid(values: str) -> list[float]:
    if not values:
        return []
    return [float(x.strip()) for x in values.split(",") if x.strip()]


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    # convencao do projeto ham = 1 spam = -1
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 1) & (y_pred == -1)))
    fn = int(np.sum((y_true == -1) & (y_pred == 1)))
    tn = int(np.sum((y_true == -1) & (y_pred == -1)))
    return tp, fp, fn, tn


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    precision = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    recall = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return precision, recall


def matrix_text(tp: int, fp: int, fn: int, tn: int) -> str:
    lines = []
    lines.append("confusion matrix")
    lines.append("           pred_ham  pred_spam")
    lines.append(f"true_ham    {tp:7d}  {fp:8d}")
    lines.append(f"true_spam   {fn:7d}  {tn:8d}")
    return "\n".join(lines)


def load_xy(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    # le csv e separa x e y assumindo output na ultima coluna
    df = pd.read_csv(csv_path, header=0)
    data = df.to_numpy()
    x = data[:, :-1].astype(float)
    y = data[:, -1].astype(int)

    # remove linhas com label 0 se existirem
    mask = y != 0
    x = x[mask]
    y = y[mask]

    # garante labels em {-1, 1}
    y = np.where(y > 0, 1, -1).astype(int)

    return x, y


def split_train_test(x: np.ndarray, y: np.ndarray, train_frac: float, shuffle: bool, seed: int):
    n = x.shape[0]
    train_n = int(n * train_frac)
    if train_n <= 0 or train_n >= n:
        raise ValueError("train_frac invalido")

    if shuffle:
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        x = x[idx]
        y = y[idx]

    x_train = x[:train_n]
    y_train = y[:train_n]
    x_test = x[train_n:]
    y_test = y[train_n:]

    return x_train, y_train, x_test, y_test


def build_model(model_name: str, c_value: float, alpha: float, max_iter: int, tol: float, class_weight: str):
    # usa escalonamento leve sem centralizar
    scaler = MaxAbsScaler()

    if model_name == "linearsvc":
        # dual false eh melhor quando n_samples > n_features
        svc = LinearSVC(
            C=c_value,
            dual=False,
            max_iter=max_iter,
            tol=tol,
            class_weight=(class_weight if class_weight != "none" else None),
        )
        return Pipeline([("scaler", scaler), ("clf", svc)])

    if model_name == "sgd":
        sgd = SGDClassifier(
            loss="hinge",
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=0,
            class_weight=(class_weight if class_weight != "none" else None),
        )
        return Pipeline([("scaler", scaler), ("clf", sgd)])

    raise ValueError("model invalido")


def write_block(resultados_path: Path, header: str, body: str) -> None:
    with resultados_path.open("a", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(body + "\n\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="treina svm otimizado usando csv de frequencia")
    parser.add_argument("--frequency_csv", type=Path, default=Path("frequencia.csv"), help="csv com features e output")
    parser.add_argument("--resultados_path", type=Path, default=Path("resultados.txt"), help="arquivo de saida dos resultados")
    parser.add_argument("--model", type=str, default="linearsvc", choices=["linearsvc", "sgd"], help="modelo a usar")
    parser.add_argument("--train_frac", type=float, default=0.70, help="fracao para treino")
    parser.add_argument("--shuffle", action="store_true", help="embaralha antes do split")
    parser.add_argument("--seed", type=int, default=42, help="seed do embaralhamento")
    parser.add_argument("--c", type=float, default=0.1, help="c para linearsvc")
    parser.add_argument("--c_grid", type=str, default="", help="lista de c separada por virgula para linearsvc")
    parser.add_argument("--alpha", type=float, default=1e-4, help="alpha para sgd")
    parser.add_argument("--alpha_grid", type=str, default="", help="lista de alpha separada por virgula para sgd")
    parser.add_argument("--max_iter", type=int, default=5000, help="max iteracoes do solver")
    parser.add_argument("--tol", type=float, default=1e-4, help="tolerancia do solver")
    parser.add_argument("--class_weight", type=str, default="none", choices=["none", "balanced"], help="peso de classe")
    args = parser.parse_args()

    global_start = time.time()
    args.resultados_path.write_text("", encoding="utf-8")

    x, y = load_xy(args.frequency_csv)
    x_train, y_train, x_test, y_test = split_train_test(x, y, args.train_frac, args.shuffle, args.seed)

    if args.model == "linearsvc":
        grid = parse_grid(args.c_grid) or [args.c]
        for c_value in grid:
            start = time.time()
            model = build_model(args.model, c_value=c_value, alpha=args.alpha, max_iter=args.max_iter, tol=args.tol, class_weight=args.class_weight)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            tp, fp, fn, tn = confusion_counts(y_test, y_pred)
            precision, recall = precision_recall(tp, fp, fn)
            elapsed = time.time() - start

            header = f"linearsvc  c={c_value}"
            body = (
                matrix_text(tp, fp, fn, tn)
                + "\n"
                + f"precision : {precision:.2f}\n"
                + f"recall : {recall:.2f}\n"
                + f"time spent for model : {elapsed:.2f}s"
            )
            write_block(args.resultados_path, header, body)
            print("done")

    if args.model == "sgd":
        grid = parse_grid(args.alpha_grid) or [args.alpha]
        for alpha in grid:
            start = time.time()
            model = build_model(args.model, c_value=args.c, alpha=alpha, max_iter=args.max_iter, tol=args.tol, class_weight=args.class_weight)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            tp, fp, fn, tn = confusion_counts(y_test, y_pred)
            precision, recall = precision_recall(tp, fp, fn)
            elapsed = time.time() - start

            header = f"sgd  alpha={alpha}"
            body = (
                matrix_text(tp, fp, fn, tn)
                + "\n"
                + f"precision : {precision:.2f}\n"
                + f"recall : {recall:.2f}\n"
                + f"time spent for model : {elapsed:.2f}s"
            )
            write_block(args.resultados_path, header, body)
            print("done")

    total_elapsed = time.time() - global_start
    with args.resultados_path.open("a", encoding="utf-8") as f:
        f.write("time spent for entire code : " + str(round(total_elapsed, 2)) + "s\n")

    print(f"resultados em {args.resultados_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
