#!/usr/bin/env python3
from __future__ import annotations

import ast
import argparse
import sys
from typing import List, Optional


# ----------------------------- Formatting Helpers -----------------------------
class Indent:
    def __init__(self, step: int = 4):
        self.level = 0
        self.step = step

    def __enter__(self):
        self.level += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        self.level -= 1

    def pad(self) -> str:
        return " " * (self.step * self.level)


def join_nonempty(parts: List[str], sep: str = " ") -> str:
    return sep.join(p for p in parts if p)


# ----------------------------- Pseudocode Emitter -----------------------------
class PseudoEmitter(ast.NodeVisitor):
    def __init__(self, style: str = "plain", show_lineno: bool = False):
        self.indent = Indent()
        self.out: List[str] = []
        self.style = style
        self.show_lineno = show_lineno

    # ------------------------- Utilities -------------------------
    def emit(self, line: str = ""):
        self.out.append(f"{self.indent.pad()}{line}")

    def mark(self, node: ast.AST) -> str:
        return f"  [L{getattr(node, 'lineno', '?')}]" if self.show_lineno else ""

    def dump(self) -> str:
        return "\n".join(self.out).rstrip() + "\n"

    # ------------------------- Expression Printer -------------------------
    def e(self, node: Optional[ast.AST]) -> str:
        if node is None:
            return ""
        method = getattr(self, f"expr_{type(node).__name__}", None)
        if method:
            return method(node)
        # Fallback to AST dump when not implemented
        return ast.unparse(node) if hasattr(ast, "unparse") else f"<{type(node).__name__}>"

    def expr_Name(self, node: ast.Name) -> str:
        return node.id

    def expr_Constant(self, node: ast.Constant) -> str:
        return repr(node.value)

    def expr_Attribute(self, node: ast.Attribute) -> str:
        return f"{self.e(node.value)}.{node.attr}"

    def expr_Subscript(self, node: ast.Subscript) -> str:
        return f"{self.e(node.value)}[{self.e(node.slice)}]"

    def expr_Slice(self, node: ast.Slice) -> str:
        lower = self.e(node.lower)
        upper = self.e(node.upper)
        step = self.e(node.step)
        inside = ":".join(filter(None, [lower, upper]))
        return f"{inside}:{step}" if step else inside

    def expr_Tuple(self, node: ast.Tuple) -> str:
        return "(" + ", ".join(self.e(elt) for elt in node.elts) + ")"

    def expr_List(self, node: ast.List) -> str:
        return "[" + ", ".join(self.e(elt) for elt in node.elts) + "]"

    def expr_Set(self, node: ast.Set) -> str:
        return "{" + ", ".join(self.e(elt) for elt in node.elts) + "}"

    def expr_Dict(self, node: ast.Dict) -> str:
        pairs = [f"{self.e(k)}: {self.e(v)}" for k, v in zip(node.keys, node.values)]
        return "{" + ", ".join(pairs) + "}"

    def expr_BinOp(self, node: ast.BinOp) -> str:
        op = type(node.op).__name__
        ops = {
            "Add": "+", "Sub": "-", "Mult": "*", "MatMult": "@", "Div": "/",
            "FloorDiv": "//", "Mod": "%", "Pow": "**",
        }
        return f"({self.e(node.left)} {ops.get(op, op)} {self.e(node.right)})"

    def expr_UnaryOp(self, node: ast.UnaryOp) -> str:
        ops = {"Not": "NOT", "USub": "-", "UAdd": "+", "Invert": "~"}
        return f"({ops.get(type(node.op).__name__, '?')} {self.e(node.operand)})"

    def expr_BoolOp(self, node: ast.BoolOp) -> str:
        op = "AND" if isinstance(node.op, ast.And) else "OR"
        return f"({f' {op} '.join(self.e(v) for v in node.values)})"

    def expr_Compare(self, node: ast.Compare) -> str:
        parts = [self.e(node.left)]
        for op, comp in zip(node.ops, node.comparators):
            name = type(op).__name__
            table = {
                "Eq": "==", "NotEq": "!=", "Lt": "<", "LtE": "<=",
                "Gt": ">", "GtE": ">=", "Is": "IS", "IsNot": "IS NOT",
                "In": "IN", "NotIn": "NOT IN",
            }
            parts.append(table.get(name, name))
            parts.append(self.e(comp))
        return " ".join(parts)

    def expr_IfExp(self, node: ast.IfExp) -> str:
        return f"IF {self.e(node.test)} THEN {self.e(node.body)} ELSE {self.e(node.orelse)}"

    def expr_Call(self, node: ast.Call) -> str:
        args = [self.e(a) for a in node.args]
        kw = [f"{k.arg}={self.e(k.value)}" for k in node.keywords]
        args = ", ".join(args + kw)
        return f"{self.e(node.func)}({args})"

    def expr_Lambda(self, node: ast.Lambda) -> str:
        params = self._format_args(node.args)
        return f"LAMBDA {params} => {self.e(node.body)}"

    def expr_JoinedStr(self, node: ast.JoinedStr) -> str:
        # f-string
        parts = []
        for v in node.values:
            if isinstance(v, ast.FormattedValue):
                parts.append(f"{{{self.e(v.value)}}}")
            else:
                parts.append(getattr(v, 'value', ''))
        return 'f"' + ''.join(parts).replace('"', '\\"') + '"'

    # -------- Comprehensions --------
    def _format_comp(self, node, kind: str) -> str:
        # kind in {"list", "set", "dict", "gen"}
        def gen_clause(c: ast.comprehension) -> str:
            clause = f"FOR {self.e(c.target)} IN {self.e(c.iter)}"
            if c.ifs:
                clause += " WHERE " + " AND ".join(self.e(i) for i in c.ifs)
            if getattr(c, 'is_async', False):
                clause = "ASYNC " + clause
            return clause
        if kind == "dict":
            head = f"{{{self.e(node.key)}: {self.e(node.value)}}}"
        elif kind == "set":
            head = f"{{{self.e(node.elt)}}}"
        else:
            head = self.e(node.elt)
        clauses = "; ".join(gen_clause(c) for c in node.generators)
        return f"COMPREHENSION build {kind} of {head} :: {clauses}"

    def expr_ListComp(self, node: ast.ListComp) -> str:
        return self._format_comp(node, "list")

    def expr_SetComp(self, node: ast.SetComp) -> str:
        return self._format_comp(node, "set")

    def expr_DictComp(self, node: ast.DictComp) -> str:
        return self._format_comp(node, "dict")

    def expr_GeneratorExp(self, node: ast.GeneratorExp) -> str:
        return self._format_comp(node, "generator")

    # ------------------------- Argument Formatting -------------------------
    def _format_args(self, args: ast.arguments) -> str:
        parts = []
        # Positional-only (3.8+)
        if getattr(args, 'posonlyargs', []):
            parts.extend(a.arg for a in args.posonlyargs)
            parts.append("/")
        parts.extend(a.arg for a in args.args)
        if args.vararg:
            parts.append("*" + args.vararg.arg)
        elif args.kwonlyargs:
            parts.append("*")
        parts.extend(a.arg for a in args.kwonlyargs)
        if args.kwarg:
            parts.append("**" + args.kwarg.arg)
        return ", ".join(parts)

    # ------------------------- Statement Visitors -------------------------
    def visit_Module(self, node: ast.Module):
        self.emit("PSEUDOCODE START")
        with self.indent:
            for stmt in node.body:
                self.visit(stmt)
        self.emit("PSEUDOCODE END")

    def visit_Expr(self, node: ast.Expr):
        # Top-level expressions => ACTION
        self.emit(join_nonempty(["DO", self.e(node.value), self.mark(node)]))

    def visit_Assign(self, node: ast.Assign):
        targets = " = ".join(self.e(t) for t in node.targets)
        self.emit(join_nonempty([f"SET {targets} TO {self.e(node.value)}", self.mark(node)]))

    def visit_AnnAssign(self, node: ast.AnnAssign):
        target = self.e(node.target)
        ann = self.e(node.annotation)
        value = self.e(node.value) if node.value else None
        line = f"DECLARE {target}: {ann}"
        if value:
            line += f" INITIALIZED WITH {value}"
        self.emit(join_nonempty([line, self.mark(node)]))

    def visit_AugAssign(self, node: ast.AugAssign):
        ops = {
            ast.Add: "+=", ast.Sub: "-=", ast.Mult: "*=", ast.Div: "/=",
            ast.Mod: "%=", ast.Pow: "**=", ast.MatMult: "@=", ast.FloorDiv: "//=",
            ast.LShift: "<<=", ast.RShift: ">>=", ast.BitOr: "|=", ast.BitAnd: "&=", ast.BitXor: "^=",
        }
        op = next((v for k, v in ops.items() if isinstance(node.op, k)), type(node.op).__name__)
        self.emit(join_nonempty([f"UPDATE {self.e(node.target)} {op} {self.e(node.value)}", self.mark(node)]))

    def visit_Return(self, node: ast.Return):
        value = self.e(node.value)
        self.emit(join_nonempty([f"RETURN {value}" if value else "RETURN", self.mark(node)]))

    def visit_Pass(self, node: ast.Pass):
        self.emit(join_nonempty(["NO-OP", self.mark(node)]))

    def visit_Break(self, node: ast.Break):
        self.emit(join_nonempty(["BREAK", self.mark(node)]))

    def visit_Continue(self, node: ast.Continue):
        self.emit(join_nonempty(["CONTINUE", self.mark(node)]))

    def visit_Delete(self, node: ast.Delete):
        self.emit(join_nonempty(["DELETE " + ", ".join(self.e(t) for t in node.targets), self.mark(node)]))

    def _emit_block(self, header: str, body: List[ast.stmt], footer: Optional[str] = None, node: Optional[ast.AST] = None):
        self.emit(join_nonempty([header, self.mark(node) if node else ""]))
        with self.indent:
            if body:
                for s in body:
                    self.visit(s)
            else:
                self.emit("(empty)")
        if footer:
            self.emit(footer)

    def visit_If(self, node: ast.If):
        self._emit_block(f"IF {self.e(node.test)} THEN", node.body, node=node)
        if node.orelse:
            # elif chain vs else
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                # elif
                self.emit("ELSE")
                with self.indent:
                    self.visit(node.orelse[0])
            else:
                self._emit_block("ELSE", node.orelse)
        self.emit("END IF")

    def visit_For(self, node: ast.For):
        hdr = join_nonempty([f"FOR {self.e(node.target)} IN {self.e(node.iter)}:", self.mark(node)])
        self._emit_block(hdr, node.body)
        if node.orelse:
            self._emit_block("ELSE (no break):", node.orelse)
        self.emit("END FOR")

    def visit_While(self, node: ast.While):
        hdr = join_nonempty([f"WHILE {self.e(node.test)}:", self.mark(node)])
        self._emit_block(hdr, node.body)
        if node.orelse:
            self._emit_block("ELSE (no break):", node.orelse)
        self.emit("END WHILE")

    def visit_With(self, node: ast.With):
        items = []
        for item in node.items:
            ctx = self.e(item.context_expr)
            if item.optional_vars:
                items.append(f"{ctx} AS {self.e(item.optional_vars)}")
            else:
                items.append(ctx)
        self._emit_block("WITH " + ", ".join(items) + ":", node.body, node=node)
        self.emit("END WITH")

    def visit_Try(self, node: ast.Try):
        self._emit_block("TRY:", node.body, node=node)
        for h in node.handlers:
            name = self.e(h.type) if h.type else "Exception"
            alias = f" AS {h.name}" if h.name else ""
            self._emit_block(f"EXCEPT {name}{alias}:", h.body)
        if node.orelse:
            self._emit_block("ELSE:", node.orelse)
        if node.finalbody:
            self._emit_block("FINALLY:", node.finalbody)
        self.emit("END TRY")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        deco = [f"@{self.e(d)}" for d in node.decorator_list]
        for d in deco:
            self.emit(f"DECORATOR {d}")
        sig = self._format_args(node.args)
        hdr = join_nonempty([f"FUNCTION {node.name}({sig}):", self.mark(node)])
        self._emit_block(hdr, node.body)
        self.emit(f"END FUNCTION {node.name}")

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        deco = [f"@{self.e(d)}" for d in node.decorator_list]
        for d in deco:
            self.emit(f"DECORATOR {d}")
        sig = self._format_args(node.args)
        hdr = join_nonempty([f"ASYNC FUNCTION {node.name}({sig}):", self.mark(node)])
        self._emit_block(hdr, node.body)
        self.emit(f"END ASYNC FUNCTION {node.name}")

    def visit_ClassDef(self, node: ast.ClassDef):
        bases = ", ".join(self.e(b) for b in node.bases) if node.bases else "Object"
        for d in node.decorator_list:
            self.emit(f"DECORATOR @{self.e(d)}")
        hdr = join_nonempty([f"CLASS {node.name} < {bases}:", self.mark(node)])
        self._emit_block(hdr, node.body)
        self.emit(f"END CLASS {node.name}")

    def visit_Raise(self, node: ast.Raise):
        if node.exc:
            self.emit(join_nonempty([f"RAISE {self.e(node.exc)}", self.mark(node)]))
        else:
            self.emit(join_nonempty(["RAISE", self.mark(node)]))

    def visit_Assert(self, node: ast.Assert):
        msg = f" MESSAGE {self.e(node.msg)}" if node.msg else ""
        self.emit(join_nonempty([f"ASSERT {self.e(node.test)}{msg}", self.mark(node)]))

    def visit_Import(self, node: ast.Import):
        mods = ", ".join((f"{a.name} AS {a.asname}" if a.asname else a.name) for a in node.names)
        self.emit(join_nonempty([f"IMPORT {mods}", self.mark(node)]))

    def visit_ImportFrom(self, node: ast.ImportFrom):
        mods = ", ".join((f"{a.name} AS {a.asname}" if a.asname else a.name) for a in node.names)
        pkg = "." * (node.level or 0) + (node.module or "")
        self.emit(join_nonempty([f"FROM {pkg} IMPORT {mods}", self.mark(node)]))

    # Generic fallback
    def generic_visit(self, node: ast.AST):
        # Fallback for statements not explicitly handled
        if isinstance(node, ast.stmt):
            text = type(node).__name__
            if hasattr(ast, "unparse"):
                try:
                    text += f": {ast.unparse(node)}"
                except Exception:
                    pass
            self.emit(join_nonempty([f"[{text}]", self.mark(node)]))
        else:
            super().generic_visit(node)


# ----------------------------- Public API -----------------------------

def python_to_pseudocode(src: str, style: str = "plain", show_lineno: bool = False) -> str:
    tree = ast.parse(src)
    emitter = PseudoEmitter(style=style, show_lineno=show_lineno)
    emitter.visit(tree)
    return emitter.dump()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Translate Python source to pseudocode")
    p.add_argument("path", nargs="?", help="Input Python file. Reads stdin if omitted.")
    p.add_argument("-o", "--output", help="Write pseudocode to this file")
    p.add_argument("--style", default="plain", choices=["plain", "verbose"], help="Pseudocode style")
    p.add_argument("--show-lineno", action="store_true", help="Annotate with source line numbers")
    args = p.parse_args(argv)

    # Read input
    if args.path:
        with open(args.path, "r", encoding="utf-8") as f:
            src = f.read()
    else:
        src = sys.stdin.read()

    try:
        pseudo = python_to_pseudocode(src, style=args.style, show_lineno=args.show_lineno)
    except SyntaxError as e:
        sys.stderr.write(f"SyntaxError: {e}\n")
        return 2

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(pseudo)
    else:
        sys.stdout.write(pseudo)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
