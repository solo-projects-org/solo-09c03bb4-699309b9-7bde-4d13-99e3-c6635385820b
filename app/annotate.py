"""变异注释逻辑：CDS TSV 解析、SNV VCF 解析、密码子/氨基酸推演。

本模块不做任何 I/O，全部基于传入的文本与参考序列字符串，保证一次上传的
参考序列、转录本、位点只存在于本次调用的局部变量中，从机制上避免并发批次混用。
任一输入非法即抛 AnnotateError，由调用方终止整次注释，不产生部分结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field

BASES = frozenset("ACGT")
MAX_SNVS = 200          # 每份 VCF 最多 200 条 SNV
MAX_TRANSCRIPTS = 5     # 每份 CDS TSV 最多 5 个转录本
MAX_SEGMENTS = 8        # 每个转录本最多 8 段

_COMPLEMENT = str.maketrans("ACGT", "TGCA")

# 标准遗传密码
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

AA_NAMES = {
    "A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys",
    "Q": "Gln", "E": "Glu", "G": "Gly", "H": "His", "I": "Ile",
    "L": "Leu", "K": "Lys", "M": "Met", "F": "Phe", "P": "Pro",
    "S": "Ser", "T": "Thr", "W": "Trp", "Y": "Tyr", "V": "Val",
    "*": "Ter",
}

CONSEQUENCE_LABELS = {
    "synonymous": "同义",
    "missense": "错义",
    "stop_gained": "终止获得",
    "stop_lost": "终止丢失",
    "outside_cds": "CDS 外",
}

_TSV_HEADER_FIRST_COLS = {"transcript", "transcript_id", "tid", "id", "name"}


class AnnotateError(Exception):
    """任一非法记录 / 非法输入导致的整次注释失败。"""


@dataclass(frozen=True)
class Segment:
    """1 基闭区间 [start, end]（基因组正链坐标）。"""

    start: int
    end: int


@dataclass
class Transcript:
    """一个转录本：区间按基因组坐标升序，CDS 已按转录方向拼接。"""

    tid: str
    strand: str                      # '+' | '-'
    segments: tuple[Segment, ...]    # 基因组坐标升序、互不重叠
    cds: str                         # 转录方向（5'→3'）拼接的 CDS 序列
    cds_index: dict[int, int] = field(default_factory=dict)  # 基因组 pos -> CDS 0-based 下标


@dataclass(frozen=True)
class Snv:
    """一条双等位 SNV（可能由多个等价 VCF 记录合并而来）。"""

    pos: int
    ref: str
    alt: str
    lines: tuple[int, ...]           # 原 VCF 行号（含表头的物理行号）


@dataclass(frozen=True)
class Annotation:
    """一条 SNV 对一个转录本的注释结果；transcript 为 None 表示 CDS 外。"""

    snv: Snv
    transcript: str | None
    cds_pos: int | None              # CDS 内 1-based 位置
    codon_number: int | None         # 第几个密码子（1-based）
    ref_codon: str | None
    new_codon: str | None
    aa_ref: str | None               # 三字母缩写
    aa_new: str | None
    consequence: str                 # synonymous | missense | stop_gained | stop_lost | outside_cds


def reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def parse_cds_tsv(text: str, ref_seq: str) -> list[Transcript]:
    """解析 CDS TSV，返回转录本列表（按文件中出现顺序）。非法则抛 AnnotateError。

    每行 4 列：transcript_id、strand(+/-)、start、end（1 基闭区间）。
    允许空行、'#' 注释行与一行表头。同一转录本须同向且区间互不重叠，
    区间须位于参考序列内；拼接后的 CDS 只含 A/C/G/T 且总长为 3 的倍数。
    """
    rows: list[tuple[int, str, str, int, int]] = []  # (行号, tid, strand, start, end)
    header_checked = False
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.rstrip("\r").strip()
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t") if "\t" in line else line.split()
        if not header_checked:
            header_checked = True
            # 首行数据若形如表头（首列为 transcript/id 类名称且第二列是 strand），跳过
            if (
                len(cols) == 4
                and cols[0].strip().lower() in _TSV_HEADER_FIRST_COLS
                and cols[1].strip().lower() == "strand"
            ):
                continue
        if len(cols) != 4:
            raise AnnotateError(
                f"CDS TSV 第 {lineno} 行应为 4 列（转录本 ID、正负链、起始、终止），实际 {len(cols)} 列"
            )
        tid, strand, start_s, end_s = (c.strip() for c in cols)
        if not tid:
            raise AnnotateError(f"CDS TSV 第 {lineno} 行转录本 ID 为空")
        if strand not in ("+", "-"):
            raise AnnotateError(
                f"CDS TSV 第 {lineno} 行正负链须为 '+' 或 '-'，实际为 '{strand}'"
            )
        if not start_s.isdigit() or not end_s.isdigit():
            raise AnnotateError(
                f"CDS TSV 第 {lineno} 行起始/终止须为正整数: '{start_s}' '{end_s}'"
            )
        start, end = int(start_s), int(end_s)
        if start < 1 or end < start:
            raise AnnotateError(
                f"CDS TSV 第 {lineno} 行区间非法（须满足 1 ≤ 起始 ≤ 终止）: {start}-{end}"
            )
        if end > len(ref_seq):
            raise AnnotateError(
                f"CDS TSV 第 {lineno} 行区间 {start}-{end} 超出参考序列范围（长度 {len(ref_seq)}）"
            )
        rows.append((lineno, tid, strand, start, end))

    if not rows:
        raise AnnotateError("CDS TSV 没有任何区间记录")

    # 按转录本分组（保持首次出现顺序）
    by_tid: dict[str, list[tuple[int, str, int, int]]] = {}
    for lineno, tid, strand, start, end in rows:
        by_tid.setdefault(tid, []).append((lineno, strand, start, end))
    if len(by_tid) > MAX_TRANSCRIPTS:
        raise AnnotateError(
            f"CDS TSV 转录本数 {len(by_tid)} 超过上限 {MAX_TRANSCRIPTS} 个"
        )

    transcripts: list[Transcript] = []
    for tid, segs in by_tid.items():
        if len(segs) > MAX_SEGMENTS:
            raise AnnotateError(
                f"转录本 '{tid}' 的区间数 {len(segs)} 超过上限 {MAX_SEGMENTS} 段"
            )
        strands = {s for _, s, _, _ in segs}
        if len(strands) > 1:
            raise AnnotateError(f"转录本 '{tid}' 的正负链不一致，同一转录本须同向")
        strand = segs[0][1]
        ordered = sorted(segs, key=lambda s: s[2])  # 按起始坐标升序
        segments: list[Segment] = []
        prev_lineno: int | None = None
        prev_seg: tuple[int, int] | None = None
        for lineno, _, start, end in ordered:
            if prev_seg is not None and start <= prev_seg[1]:
                raise AnnotateError(
                    f"转录本 '{tid}' 第 {lineno} 行的区间 {start}-{end} 与第 {prev_lineno} 行的"
                    f"区间 {prev_seg[0]}-{prev_seg[1]} 重叠，同一转录本区间不得重叠"
                )
            segments.append(Segment(start, end))
            prev_lineno, prev_seg = lineno, (start, end)

        # 按转录方向拼接 CDS：正链按坐标升序逐段取正链序列；
        # 负链按坐标降序逐段取反向互补序列
        pieces: list[str] = []
        cds_index: dict[int, int] = {}
        offset = 0
        directed = segments if strand == "+" else list(reversed(segments))
        for seg in directed:
            genomic = ref_seq[seg.start - 1 : seg.end]
            if strand == "+":
                pieces.append(genomic)
                for i, pos in enumerate(range(seg.start, seg.end + 1)):
                    cds_index[pos] = offset + i
            else:
                pieces.append(reverse_complement(genomic))
                for i, pos in enumerate(range(seg.end, seg.start - 1, -1)):
                    cds_index[pos] = offset + i
            offset += seg.end - seg.start + 1
        cds = "".join(pieces)

        bad = sorted(set(cds) - BASES)
        if bad:
            raise AnnotateError(
                f"转录本 '{tid}' 的 CDS 含非法字符: {''.join(bad)}（仅允许 A/C/G/T）"
            )
        if len(cds) % 3 != 0:
            raise AnnotateError(
                f"转录本 '{tid}' 的 CDS 总长 {len(cds)} 不是 3 的倍数"
            )
        transcripts.append(Transcript(tid, strand, tuple(segments), cds, cds_index))
    return transcripts


def parse_snv_vcf(text: str, ref_seq: str, ref_name: str) -> list[Snv]:
    """解析注释用 VCF，返回按坐标排序的 SNV 列表（完全相同的记录合并并保留全部行号）。

    仅接受双等位 SNV（REF/ALT 各一个 A/C/G/T 碱基），任一记录非法即抛
    AnnotateError，由调用方终止整次注释。
    """
    merged: dict[tuple[int, str, str], list[int]] = {}
    n_records = 0
    for lineno, raw in enumerate(text.split("\n"), 1):
        line = raw.rstrip("\r")
        if not line.strip() or line.startswith("#"):
            continue
        n_records += 1
        if n_records > MAX_SNVS:
            raise AnnotateError(
                f"VCF 记录数超过上限 {MAX_SNVS} 条（第 {lineno} 行）"
            )
        cols = line.split("\t") if "\t" in line else line.split()
        if len(cols) < 5:
            raise AnnotateError(
                f"VCF 第 {lineno} 行列数不足，至少需要 CHROM/POS/ID/REF/ALT 5 列"
            )
        chrom, pos_s, _id, ref, alt = cols[0], cols[1], cols[2], cols[3].upper(), cols[4].upper()
        if chrom != ref_name:
            raise AnnotateError(
                f"VCF 第 {lineno} 行染色体 '{chrom}' 与参考序列 '{ref_name}' 不一致"
            )
        if not pos_s.isdigit() or int(pos_s) < 1:
            raise AnnotateError(f"VCF 第 {lineno} 行 POS 不是正整数: '{pos_s}'")
        pos = int(pos_s)
        if "," in alt:
            raise AnnotateError(
                f"VCF 第 {lineno} 行 ALT 为多等位 '{cols[4]}'，仅支持双等位 SNV"
            )
        if len(ref) != 1 or len(alt) != 1 or ref not in BASES or alt not in BASES:
            raise AnnotateError(
                f"VCF 第 {lineno} 行不是双等位 SNV（REF/ALT 须各为一个 A/C/G/T 碱基）: "
                f"REF='{cols[3]}' ALT='{cols[4]}'"
            )
        if ref == alt:
            raise AnnotateError(f"VCF 第 {lineno} 行 REF 与 ALT 相同，不是变异")
        if pos > len(ref_seq):
            raise AnnotateError(
                f"VCF 第 {lineno} 行 POS {pos} 超出参考序列范围（长度 {len(ref_seq)}）"
            )
        actual = ref_seq[pos - 1]
        if actual != ref:
            raise AnnotateError(
                f"VCF 第 {lineno} 行 REF '{ref}' 与参考序列不符（{ref_name}:{pos} 实际为 '{actual}'）"
            )
        merged.setdefault((pos, ref, alt), []).append(lineno)
    return [
        Snv(pos, ref, alt, tuple(lines))
        for (pos, ref, alt), lines in sorted(merged.items())
    ]


def _classify(aa_ref: str, aa_new: str) -> str:
    if aa_ref == aa_new:
        return "synonymous"  # 含终止→终止（如 TAA→TAG）
    if aa_ref == "*":
        return "stop_lost"
    if aa_new == "*":
        return "stop_gained"
    return "missense"


def annotate(snv: Snv, transcript: Transcript) -> Annotation | None:
    """注释一条 SNV 对一个转录本的影响；未命中该转录本返回 None。"""
    idx = transcript.cds_index.get(snv.pos)
    if idx is None:
        return None
    alt = snv.alt if transcript.strand == "+" else snv.alt.translate(_COMPLEMENT)
    codon_number = idx // 3 + 1
    start = (idx // 3) * 3
    ref_codon = transcript.cds[start : start + 3]
    new_codon = ref_codon[: idx % 3] + alt + ref_codon[idx % 3 + 1 :]
    aa_ref = CODON_TABLE[ref_codon]
    aa_new = CODON_TABLE[new_codon]
    return Annotation(
        snv=snv,
        transcript=transcript.tid,
        cds_pos=idx + 1,
        codon_number=codon_number,
        ref_codon=ref_codon,
        new_codon=new_codon,
        aa_ref=AA_NAMES[aa_ref],
        aa_new=AA_NAMES[aa_new],
        consequence=_classify(aa_ref, aa_new),
    )


def annotate_all(snvs: list[Snv], transcripts: list[Transcript]) -> list[Annotation]:
    """对每条 SNV 列出各命中转录本的注释；未命中任何转录本的标为 CDS 外。"""
    rows: list[Annotation] = []
    for snv in snvs:
        hits = [a for t in transcripts if (a := annotate(snv, t)) is not None]
        if hits:
            rows.extend(hits)
        else:
            rows.append(
                Annotation(
                    snv=snv,
                    transcript=None,
                    cds_pos=None,
                    codon_number=None,
                    ref_codon=None,
                    new_codon=None,
                    aa_ref=None,
                    aa_new=None,
                    consequence="outside_cds",
                )
            )
    return rows


__all__ = [
    "AnnotateError",
    "Annotation",
    "parse_cds_tsv",
    "parse_snv_vcf",
    "annotate",
    "annotate_all",
    "reverse_complement",
    "CONSEQUENCE_LABELS",
    "MAX_SNVS",
    "MAX_TRANSCRIPTS",
    "MAX_SEGMENTS",
]
