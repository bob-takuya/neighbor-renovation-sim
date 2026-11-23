"""事前登録で凍結したファイルと、このリポジトリのファイルを照合する。

凍結の時点のハッシュは out/*_frozen.sha256 にある。公開にあたり .py のコメントと docstring を整理したので、
.py はファイルのハッシュでは一致しない。そこで、コメント・docstring・空行を除いた字句の列のハッシュ
（out/frozen_code.sha256。凍結したファイルから求めた）と、このリポジトリのファイルから求めた同じハッシュを比べる。
.json（事前登録の文）は手を加えていないので、ファイルのハッシュをそのまま比べる。

使い方：python3 tools/check_frozen.py   （リポジトリの最上位で実行する）
"""
import hashlib
import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING,
        tokenize.ENDMARKER}


def code_tokens(src):
    """コメント・docstring・改行を除いた字句の列。字下げは各行の頭の深さとして残す"""
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    out, depth, prev = [], 0, None
    for t in toks:
        if t.type == tokenize.INDENT:
            depth += 1
        elif t.type == tokenize.DEDENT:
            depth -= 1
        is_doc = (t.type == tokenize.STRING and prev in (None, tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT))
        if t.type not in (tokenize.COMMENT, tokenize.NL):
            prev = t.type
        if is_doc:
            prev = "doc"
            continue
        if t.type == tokenize.NEWLINE:
            out.append("\n")
            continue
        if t.type in SKIP:
            continue
        if out and out[-1] == "\n":
            out.append(f"<{depth}>")
        out.append(t.string)
    s = " ".join(out)
    while "\n \n" in s:
        s = s.replace("\n \n", "\n")
    return s


def code_hash(path):
    return hashlib.sha256(code_tokens(path.read_text(encoding="utf-8")).encode("utf-8")).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ok = True
    ref = {}
    for line in (ROOT / "out" / "frozen_code.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            ref[name.strip()] = h
    for f in sorted((ROOT / "out").glob("*_frozen.sha256")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                continue
            h, name = parts[0], parts[1].strip()
            p = ROOT / name
            if not p.exists():
                print(f"{f.name}  {name}: ファイルがない"); ok = False; continue
            if name.endswith(".py"):
                good = code_hash(p) == ref.get(name)
                kind = "コード"
            else:
                good = file_hash(p) == h
                kind = "ファイル"
            print(f"{f.name}  {name}: {kind}のハッシュ {'一致' if good else '不一致'}")
            ok &= good
    print("すべて一致" if ok else "一致しないものがある")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
