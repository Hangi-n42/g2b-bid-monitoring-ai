"""Select exact source spans without learning from evaluation records.

All offsets refer to the text supplied by the caller; normalize before calling
if NFC output is required. Only each individual record affects its selection.
"""
from __future__ import annotations

import re


# Overlapping item families deliberately receive separate selection turns.
# These patterns retrieve context; they do not decide whether a violation exists.
_QUERIES = (
    r"참가.{0,8}자격|비영리|공공기관|출연|연구기관|협회|법인",
    r"실적|수행.{0,8}경험|납품.{0,8}경험",
    r"실적|단일.{0,8}계약|배수|동등.{0,8}이상",
    r"실적|국가.{0,12}지방|공공.{0,8}기관|민간|해외",
    r"지역.{0,8}제한|본점|본사|소재지|주된.{0,8}영업",
    r"소재|관내|시.{0,5}군.{0,5}구|지역",
    r"인접|인근|지역|소재",
    r"실적|지역|본점|소재지",
    r"모델|제조사|제조업체|제조업자|품명|규격|동등|상표|정품|브랜드|[A-Za-z]{2,}[-_]?[0-9]{2,}",
    r"직접.{0,5}생산|직생|경쟁제품|세부품명",
    r"중소기업|확인서|경쟁제품",
    r"직접.{0,5}생산|경쟁제품|세부품명",
    r"소기업|소상공인|중소기업|경쟁제품",
    r"중소기업|추정.{0,5}가격|예산|기초.{0,5}금액",
    r"소기업|소상공인|판로|예외",
    r"중소기업|확인서|판로|예외",
    r"중소기업|소기업|소상공인",
    r"소기업|소상공인|확인서|판로|예외",
    r"확약|기술.{0,5}지원|물품.{0,5}공급|제조사|발급|입찰.{0,5}전",
    r"소프트웨어|소프트웨어사업자|SW|대기업|상호출자|사업.{0,5}금액|매출",
    r"공동|분담|이행|지분|출자|구성원|구성업체",
    r"설명회|현장.{0,5}설명|참석|협상",
    r"설명회|제안.{0,5}요청|제안.{0,5}제출|공고.{0,5}기간|접수|마감|긴급",
    r"예산|추정.{0,5}가격|기초.{0,5}금액|계약.{0,5}방법|지역|업종|면허|자격",
)
_PATTERNS = tuple(re.compile(q, re.I) for q in _QUERIES)
_BOUNDARY = re.compile(r"\n|[.!?。](?=\s|$)")
_MODEL_CODE = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9/-]*[A-Za-z])(?=[A-Za-z0-9/-]*[0-9])[A-Za-z0-9][A-Za-z0-9/-]{2,}(?![A-Za-z0-9])")
_MODEL_NAME = re.compile(r"\b[A-Z][a-z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)?\s+(?=[A-Za-z0-9/-]*\d)[A-Za-z0-9/-]+")
_MEASURE_CODE = re.compile(r"(?:\d+(?:GB|MB|TB|fps|set|mAh|Hz|dBm|km|mm|cm|Wh|kg|GHz|kHz|MHz|inch|x)|IP\d+)", re.I)
_EVENT = re.compile(r"설명회|현장\s*설명|사업\s*설명|제안\s*설명")
_DATE = re.compile(r"(?:20\d{2}\s*[.년/-]\s*)?\d{1,2}\s*[.월/-]\s*\d{1,2}\s*[.일]?")
_ELIGIBILITY = re.compile(r"입찰\s*참가\s*자격|참가\s*자격|직접\s*생산|중소기업|소기업|소상공인|지역\s*제한|실적\s*제한|공동\s*수급|대기업")
_EXCEPTION = re.compile(r"다만|단[,，\s]|예외|제외|적용하지|적용\s*제외|아니한|불구하고|면제|갈음|비영리|동등\s*이상")


def _complete_segments(rec: dict) -> list[dict]:
    """Partition every source character, with no gaps or duplicate coverage."""
    segments = []
    for doc_index, document in enumerate(rec.get("docs") or []):
        text = document.get("text") or ""
        start = 0
        while start < len(text):
            end = min(start + 360, len(text))
            if end < len(text):
                # A list marker such as "가." is not a sentence end. Prefer
                # complete physical lines before considering punctuation.
                boundaries = [m.end() for m in re.finditer(r"\n", text[start + 180:end])]
                if boundaries:
                    end = start + 180 + boundaries[-1]
                else:
                    boundaries = [m.end() for m in _BOUNDARY.finditer(text, start + 180, end)
                                  if not re.search(r"(?:^|\n)\s*(?:[가-힣]|\d+)\.$", text[start:m.end()])]
                    if boundaries:
                        end = boundaries[-1]
            segments.append({"doc": doc_index, "start": start, "end": end,
                             "text": text[start:end], "type": document.get("type", "기타")})
            start = end
    return segments


def segment_record(rec: dict) -> list[dict]:
    """Return document-order exact substrings of at most 360 characters.

    Windows overlap by up to 100 characters to retain conditions crossing a
    sentence/line boundary. Empty/whitespace-only spans are omitted.
    """
    segments = []
    for doc_index, document in enumerate(rec.get("docs") or []):
        text = document.get("text") or ""
        start = 0
        while start < len(text):
            while start < len(text) and text[start].isspace():
                start += 1
            if start >= len(text):
                break
            cap = min(start + 360, len(text))
            end = cap
            if cap < len(text):
                boundaries = [m.end() for m in _BOUNDARY.finditer(text, start + 180, cap)]
                if boundaries:
                    end = boundaries[-1]
            while end > start and text[end - 1].isspace():
                end -= 1
            if end > start:
                segments.append({"doc": doc_index, "start": start, "end": end,
                                 "text": text[start:end], "type": document.get("type", "기타")})
            if cap == len(text):
                break
            # Advance at least half a window even if a short boundary was found.
            next_start = max(start + 120, end - 100)
            # Prefer beginning the overlap at a real sentence/line boundary.
            overlap_boundaries = [m.end() for m in _BOUNDARY.finditer(text, next_start, min(end, next_start + 60))]
            start = overlap_boundaries[0] if overlap_boundaries else next_start
    return segments


def select_segments(rec: dict, max_chars: int = 14000, items: tuple | None = None) -> list[dict]:
    """Select spans using fixed item queries and document coverage, not labels.

    The budget counts span text (including overlap), excluding prompt headers.
    Output stays in original document/offset order and receives local sid 1..N.
    Optional items restricts the retrieval queries (integers 1..24 or v1..v24).
    Omitting items preserves the original all-item selection policy.
    """
    if items is None:
        active_indices = tuple(range(24))
    else:
        numbers = []
        for item in items:
            if isinstance(item, str) and re.fullmatch(r"v(?:[1-9]|1\d|2[0-4])", item):
                item = int(item[1:])
            if type(item) is not int or not 1 <= item <= 24:
                raise ValueError('items must contain integers 1..24 or keys v1..v24')
            numbers.append(item - 1)
        active_indices = tuple(sorted(set(numbers)))
        if not active_indices:
            raise ValueError('items must not be empty')
    if max_chars <= 0:
        return []
    total_chars = sum(len(d.get("text") or "") for d in rec.get("docs") or [])
    if total_chars <= max_chars:
        selected = _complete_segments(rec)
        for sid, seg in enumerate(selected, 1):
            seg["sid"] = sid
        return selected
    pool = segment_record(rec)
    if not pool:
        return []
    chosen: set[int] = set()
    used = 0
    by_doc: dict[int, list[int]] = {}
    for i, seg in enumerate(pool):
        by_doc.setdefault(seg["doc"], []).append(i)

    def add(index: int) -> bool:
        nonlocal used
        if index in chosen:
            return False
        seg = pool[index]
        if used + len(seg["text"]) > max_chars:
            return False
        # Skip nearly identical windows, while permitting boundary overlap.
        for j in chosen:
            other = pool[j]
            if other["doc"] != seg["doc"]:
                continue
            overlap = max(0, min(seg["end"], other["end"]) - max(seg["start"], other["start"]))
            if overlap >= 0.85 * len(seg["text"]):
                return False
        chosen.add(index)
        used += len(seg["text"])
        return True

    # Each document gets at least an initial opportunity; the public notice gets
    # two extra leading spans so its basic context is available before ranking.
    for indexes in by_doc.values():
        add(indexes[0])
    for indexes in by_doc.values():
        if pool[indexes[0]]["type"] == "공고문":
            for index in indexes[1:3]:
                add(index)

    rankings = []
    for item_index in active_indices:
        pattern = _PATTERNS[item_index]
        hits = []
        for i, seg in enumerate(pool):
            text = seg["text"]
            matches = list(pattern.finditer(text))
            model_codes = [m for m in _MODEL_CODE.finditer(text)
                           if not _MEASURE_CODE.fullmatch(m.group())] if item_index == 8 else []
            if not matches and not model_codes:
                continue
            # Distinct trigger forms help avoid a repeated word dominating.
            score = min(len(matches), 6) + len({m.group().lower() for m in matches})
            if item_index == 8 and seg["type"] in ("규격서", "과업지시서", "제안요청서"):
                score += 5
            if item_index == 8:
                # Model identifiers often contain spaces before a short mixed
                # alphanumeric suffix. Pure date/price numerals do not qualify.
                score += min(len(model_codes), 3) * 5
                if _MODEL_NAME.search(text):
                    score += 12
                if re.search(r"모델\s*명|제조사|브랜드|상표", text):
                    score += 3
                    if items is not None:
                        # Explicit manufacturer/model fields are more direct
                        # than incidental mixed letter-digit technical units.
                        score += 12
            if item_index == 21 and _EVENT.search(text):
                # Event attendance is relevant when it controls eligibility;
                # generic participation language elsewhere is less informative.
                score += 8
                if re.search(r"불참|참석|참가", text):
                    score += 5
                if re.search(r"제외|필수|한하|제한|자격|불허|반드시", text):
                    score += 7
            if item_index == 22 and _EVENT.search(text):
                # Preserve actual event dates and deadlines instead of merely
                # ranking repeated generic submission instructions highly.
                score += 8
                if _DATE.search(text):
                    score += 9
                if re.search(r"일시|일정|마감|공고일|개최", text):
                    score += 3
            if item_index in (18, 20, 21, 22) and seg["type"] == "공고문":
                score += 2
            hits.append((-score, seg["doc"], seg["start"], i))
        rankings.append([h[-1] for h in sorted(hits)])

    if items is not None and 8 in active_indices:
        # The specification-only item competes with six unrelated notice items
        # in its group. Give its strongest attachment candidates an early turn.
        specification_ranking = rankings[active_indices.index(8)]
        for index in specification_ranking[:3]:
            add(index)

    # Extra context is selected independently of labels. Conditions and their
    # exceptions can sit in adjacent windows: keep those windows together.
    condition_context = []
    for i, seg in enumerate(pool):
        if not _ELIGIBILITY.search(seg["text"]):
            continue
        if items is not None and not any(_PATTERNS[q].search(seg["text"]) for q in active_indices):
            continue
        neighbors = [j for j in (i - 1, i, i + 1)
                     if 0 <= j < len(pool) and pool[j]["doc"] == seg["doc"]]
        for j in neighbors:
            near = pool[j]
            if not _EXCEPTION.search(near["text"]):
                continue
            # Prioritize the actual exception and then the condition giving it
            # meaning. All candidates remain exact spans in their own document.
            condition_context.extend((j, i))
    condition_context = list(dict.fromkeys(condition_context))

    # Fair turns across all 24 item queries. Taking one candidate per turn
    # preserves nearby conditions without letting common eligibility words fill
    # the whole context before late-numbered items are considered.
    cursors = [0] * len(rankings)
    context_added = False
    while True:
        progressed = False
        for q, ranking in enumerate(rankings):
            added = 0
            while cursors[q] < len(ranking) and added < 1:
                i = ranking[cursors[q]]
                cursors[q] += 1
                if add(i):
                    added += 1
                    progressed = True
        if not context_added:
            # Every item gets its first turn before this reservation. Limit it
            # so exceptions do not crowd out rare item families or attachments.
            extra_start = used
            allowance = min(2000, max_chars // 8)
            for i in condition_context:
                if i not in chosen and used - extra_start + len(pool[i]["text"]) <= allowance:
                    add(i)
            context_added = True
        if not progressed:
            break

    # Use residual space for balanced source coverage, including text with no
    # expected keywords, which may carry an exception or an unusual model name.
    for offset in range(max(len(v) for v in by_doc.values())):
        for indexes in by_doc.values():
            if offset < len(indexes):
                add(indexes[offset])

    selected = [dict(pool[i]) for i in sorted(chosen)]
    for sid, seg in enumerate(selected, 1):
        seg["sid"] = sid
    return selected
