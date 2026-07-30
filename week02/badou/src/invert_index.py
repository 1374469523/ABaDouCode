"""
倒排索引全文检索实现（主要针对中文）

支持：
- 中文分词（jieba）
- 布尔查询：AND (&)、OR (|)、NOT (-)
- 括号分组优先级
- 短语匹配（分词后的连续词匹配）
- 结果高亮
"""

import jieba

class InvertedIndex:
    """倒排索引全文检索类

    将一批文档分词建库，然后支持以布尔表达式检索文档 ID 集合，
    最终返回匹配的文档原文（可选高亮）。

    Attributes:
        index (dict[str, set[int]]):  词项 → 包含该词的文档 ID 集合
        doc_list (list[str]):        文档原文列表，下标即 doc_id
        max_id (int):                下一个新文档的 ID
    """

    def __init__(self, docs_file: str | None = None):
        """初始化索引。

        Args:
            docs_file: 可选的文档文件路径，每行一个文档。
                       若提供则自动读入建索引。
        """
        self.index: dict[str, set[int]] = {}
        self.doc_list: list[str] = []
        self.max_id: int = 0
        if docs_file is not None:
            self.load_docs(docs_file)

    def load_docs(self, docs_file: str) -> None:
        """从文件读入文档并构建倒排索引（覆盖原有数据）。

        自动跳过重复文档（空行也被忽略）。
        """
        self.index.clear()
        self.doc_list.clear()
        self.max_id = 0

        seen: set[str] = set()
        with open(docs_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    self.add_doc(line)

    def add_doc(self, doc: str) -> int:
        """向索引中添加一篇文档。

        Args:
            doc: 文档文本。

        Returns:
            该文档被分配的 ID（int）。
        """
        doc_id = self.max_id
        self.max_id += 1
        self.doc_list.append(doc)

        for term in jieba.cut(doc):
            if not term.strip():
                continue
            if term in self.index:
                self.index[term].add(doc_id)
            else:
                self.index[term] = {doc_id}
        return doc_id

    def _all_ids(self) -> set[int]:
        """返回所有文档 ID 的集合，供 NOT 运算使用。"""
        return set(range(len(self.doc_list)))

    def word_match(self, word: str) -> set[int]:
        """返回包含 *完整短语*（分词后全部词项）的文档 ID 集合。

        例如 word="北京大学" → jieba 切为 ["北京", "大学"] →
        返回同时包含 "北京" 和 "大学" 的文档集合（交集）。
        """
        terms = [t for t in jieba.cut(word) if t.strip()]
        if not terms:
            return set()

        result: set[int] | None = None
        for t in terms:
            post = self.index.get(t, set())
            if result is None:
                result = post.copy()
            else:
                result &= post
        return result if result is not None else set()

    def conv_query(self, query: str) -> str:
        """将用户输入的布尔查询字符串转换成 Python 集合运算表达式。

        支持运算符（不区分大小写）：
            AND / and   →  &   （交集）
            OR  / or    →  |   （并集）
            NOT / not   →  -   （差集）
            ( )         →  括号分组

        分词时可能将用户希望当作短语的词切开（例如 "北京大学" 被切成
        "北京" "大学"）；本方法会通过一个 **缓存** 变量将相邻的普通词
        合并回短语，再统一用 word_match() 处理。

        Args:
            query: 用户输入的布尔查询。

        Returns:
            可直接交给 eval() 求值的 Python 表达式字符串，
            例如 ``self.word_match('苹果') & (self.word_match('芯片') | self.word_match('高通'))``
        """
        tokens = list(jieba.cut(query))
        n = len(tokens)
        out: list[str] = []
        cache = ""                  # 暂存被切开的短语片段
        pending_not = False         # 遇到 NOT 后置为 True
        not_paren_depth = 0         # NOT 括号嵌套深度，用于补全多余的 )
        i = 0

        while i < n:
            t = tokens[i]

            # ── 括号 ──────────────────────────────────────────────────
            if t == "(":
                if pending_not:
                    # NOT (A or B) → (self._all_ids() - (A | B))
                    out.append("(self._all_ids() - (")
                    pending_not = False
                    not_paren_depth = 1
                else:
                    out.append("(")
                    if not_paren_depth:
                        not_paren_depth += 1
            elif t == ")":
                if not_paren_depth:
                    not_paren_depth -= 1
                    if not_paren_depth == 0:
                        out.append("))")   # 补全外层 (self._all_ids() - (...))
                    else:
                        out.append(")")
                else:
                    out.append(")")

            # ── 空格（保留以便 eval 正确解析 token 边界） ─────────────
            elif t == " ":
                out.append(" ")

            # ── 布尔操作符（不区分大小写） ─────────────────────────────
            elif t.upper() in ("AND",):
                out.append("&")
            elif t.upper() in ("OR",):
                out.append("|")
            elif t.upper() in ("NOT",):
                pending_not = True       # 与下一个词项合并处理

            # ── 普通词项（短语合并 + 处理 NOT） ──────────────────────
            else:
                # 下一个 token 也是普通词？→ 短语中段，暂存到 cache
                if i + 1 < n and tokens[i + 1] not in (" ", "(", ")",):
                    cache += t
                else:
                    phrase = cache + t
                    if pending_not:
                        # NOT word → (全集 - word_match('word'))
                        out.append(f"(self._all_ids() - self.word_match({phrase!r}))")
                        pending_not = False
                    else:
                        out.append(f"self.word_match({phrase!r})")
                    cache = ""
            i += 1

        return "".join(out)

    # ── 高亮 --------------------------------------------------------------

    def highlighter(self, doc: str, query: str) -> str:
        """将文档中匹配查询关键词的部分用 HTML 红字标记。

        Args:
            doc:   文档原文。
            query: 用户原始查询字符串。

        Returns:
            高亮后的 HTML 片段。
        """
        seen: set[str] = set()
        for part in jieba.cut(query):
            part = part.strip()
            if not part or part in ("(", ")", " ") or part.upper() in ("AND", "OR", "NOT"):
                continue
            if part in seen:
                continue
            seen.add(part)
            doc = doc.replace(
                part,
                f'<span style="color:red">{part}</span>',
            )
        return doc

    def search(self, query: str, highlight: bool = True) -> list[str]:
        """执行布尔查询，返回匹配的文档列表（可选高亮）。

        Args:
            query:     用户布尔查询字符串，例如 ``"苹果 and (芯片 or 高通)"``。
            highlight: 是否对结果做关键词高亮。

        Returns:
            匹配的文档原文列表（若 highlight=True 则包含 HTML 高亮标签）。
        """
        expr = self.conv_query(query)
        matched_ids: set[int] = eval(expr)  # 调用 word_match 返回的 set

        results: list[str] = []
        for doc_id in sorted(matched_ids):
            doc = self.doc_list[doc_id]
            if highlight:
                doc = self.highlighter(doc, query)
            results.append(doc)
        return results

if __name__ == "__main__":
    import os

    data_path = os.path.join(
        os.path.dirname(__file__),
        "..", "asserts", "爬虫-新闻标题.txt",
    )

    searcher = InvertedIndex(data_path)

    # 查看索引中某些词项的文档数
    for w in ("孟买", "贫民窟", "豪宅", "苹果"):
        n = len(searcher.index.get(w, set()))
        print(f"  term={w!r:6s}  docs={n}")

    # 简单查询
    print("\n" + "=" * 60)
    queries = [
        "苹果 and (芯片 or 高通)",
        "泰山 and (台北 or 高通)",
        "豪宅 and not 美国",
        "孟买 or 印度",
        "别墅 and (车库 or 花园)",
    ]
    for q in queries:
        print(f"\n查询: {q}")
        for doc in searcher.search(q):
            print(f"  -> {doc}")
