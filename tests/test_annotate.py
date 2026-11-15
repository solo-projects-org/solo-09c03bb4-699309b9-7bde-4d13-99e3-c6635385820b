"""变异注释：CDS TSV / SNV VCF 解析与密码子推演的单元测试 + 网页集成测试。"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.annotate import (
    MAX_SEGMENTS,
    MAX_SNVS,
    MAX_TRANSCRIPTS,
    AnnotateError,
    Segment,
    annotate,
    annotate_all,
    parse_cds_tsv,
    parse_snv_vcf,
    reverse_complement,
)
from app.main import create_app

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# 与 examples/ann_ref.fa 相同的 60 bp 参考序列：
# 1-4 GGAT | 5-10 ATGGCA | 11-20 CTGCAGCTGC | 21-26 ACTTAA |
# 27-40 GATTACAGATTACA | 41-52 TTAGAATTTCAT | 53-60 GCGCGCGC
REF = "GGATATGGCACTGCAGCTGCACTTAAGATTACAGATTACATTAGAATTTCATGCGCGCGC"
NAME = "chrA"
VCF_HEADER = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"


def vcf_line(pos, ref, alt, chrom=NAME):
    return f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t."


def parse_vcf(lines, ref=REF, name=NAME):
    return parse_snv_vcf(VCF_HEADER + "\n".join(lines) + "\n", ref, name)


# ---------- CDS TSV ----------

def test_tsv_plus_single_segment():
    (tx,) = parse_cds_tsv("TX1\t+\t5\t10\n", REF)
    assert tx.tid == "TX1"
    assert tx.strand == "+"
    assert tx.segments == (Segment(5, 10),)
    assert tx.cds == "ATGGCA"
    assert tx.cds_index[5] == 0 and tx.cds_index[10] == 5


def test_tsv_plus_two_segments_spliced_in_order():
    (tx,) = parse_cds_tsv("TX1\t+\t21\t26\nTX1\t+\t5\t10\n", REF)
    # 文件中乱序给出也应按基因组坐标升序拼接
    assert tx.segments == (Segment(5, 10), Segment(21, 26))
    assert tx.cds == "ATGGCA" + "ACTTAA"


def test_tsv_minus_strand_reverse_complement():
    (tx,) = parse_cds_tsv("TX2\t-\t41\t52\n", REF)
    assert tx.cds == "ATGAAATTCTAA"  # revcomp(TTAGAATTTCAT)
    # 转录方向第一个碱基对应基因组最高坐标
    assert tx.cds_index[52] == 0
    assert tx.cds_index[41] == 11


def test_tsv_minus_strand_two_segments():
    # 负链多段：坐标大的段先转录
    (tx,) = parse_cds_tsv("TXM\t-\t5\t10\nTXM\t-\t21\t26\n", REF)
    assert tx.cds == reverse_complement("ACTTAA") + reverse_complement("ATGGCA")
    assert tx.cds_index[26] == 0
    assert tx.cds_index[5] == 11


def test_tsv_header_and_comments_skipped():
    text = "# 注释行\ntranscript_id\tstrand\tstart\tend\nTX1\t+\t5\t10\n\n"
    (tx,) = parse_cds_tsv(text, REF)
    assert tx.tid == "TX1"


def test_tsv_adjacent_segments_ok():
    (tx,) = parse_cds_tsv("TX1\t+\t5\t10\nTX1\t+\t11\t13\n", REF)
    assert tx.cds == "ATGGCA" + "CTG"


def test_tsv_empty_rejected():
    with pytest.raises(AnnotateError, match="没有任何区间记录"):
        parse_cds_tsv("# 只有注释\n\n", REF)


def test_tsv_wrong_columns_rejected():
    with pytest.raises(AnnotateError, match="4 列"):
        parse_cds_tsv("TX1\t+\t5\n", REF)


def test_tsv_bad_strand_rejected():
    with pytest.raises(AnnotateError, match="正负链"):
        parse_cds_tsv("TX1\t?\t5\t10\n", REF)


def test_tsv_non_integer_bounds_rejected():
    with pytest.raises(AnnotateError, match="正整数"):
        parse_cds_tsv("TX1\t+\ta\t10\n", REF)


def test_tsv_inverted_interval_rejected():
    with pytest.raises(AnnotateError, match="区间非法"):
        parse_cds_tsv("TX1\t+\t10\t5\n", REF)


def test_tsv_interval_out_of_range_rejected():
    with pytest.raises(AnnotateError, match="超出参考序列范围"):
        parse_cds_tsv("TX1\t+\t55\t61\n", REF)


def test_tsv_mixed_strands_rejected():
    with pytest.raises(AnnotateError, match="同向"):
        parse_cds_tsv("TX1\t+\t5\t10\nTX1\t-\t21\t26\n", REF)


def test_tsv_overlapping_segments_rejected():
    with pytest.raises(AnnotateError, match="重叠"):
        parse_cds_tsv("TX1\t+\t5\t10\nTX1\t+\t8\t16\n", REF)


def test_tsv_duplicate_segment_rejected():
    with pytest.raises(AnnotateError, match="重叠"):
        parse_cds_tsv("TX1\t+\t5\t10\nTX1\t+\t5\t10\n", REF)


def test_tsv_too_many_transcripts_rejected():
    lines = [f"TX{i}\t+\t5\t7\n" for i in range(MAX_TRANSCRIPTS + 1)]
    with pytest.raises(AnnotateError, match="超过上限"):
        parse_cds_tsv("".join(lines), REF)


def test_tsv_too_many_segments_rejected():
    # 用互不重叠的 1 bp 区间凑够段数（坐标都在参考序列内）
    lines = [f"TX1\t+\t{p}\t{p}\n" for p in range(1, (MAX_SEGMENTS + 1) * 3 + 1, 3)]
    with pytest.raises(AnnotateError, match="超过上限"):
        parse_cds_tsv("".join(lines), REF)


def test_tsv_cds_with_n_rejected():
    ref = REF[:4] + "N" + REF[5:]
    with pytest.raises(AnnotateError, match="非法字符"):
        parse_cds_tsv("TX1\t+\t3\t8\n", ref)


def test_tsv_cds_length_not_multiple_of_3_rejected():
    with pytest.raises(AnnotateError, match="3 的倍数"):
        parse_cds_tsv("TX1\t+\t5\t9\n", REF)  # 5 bp


# ---------- SNV VCF ----------

def test_vcf_ok_and_duplicates_merged():
    snvs = parse_vcf([vcf_line(5, "A", "T"), vcf_line(6, "T", "C"), vcf_line(5, "A", "T")])
    assert [(s.pos, s.ref, s.alt, s.lines) for s in snvs] == [
        (5, "A", "T", (3, 5)),
        (6, "T", "C", (4,)),
    ]


def test_vcf_lowercase_normalized():
    (s,) = parse_vcf([vcf_line(5, "a", "t")])
    assert (s.ref, s.alt) == ("A", "T")


def test_vcf_indel_rejected():
    with pytest.raises(AnnotateError, match="双等位 SNV"):
        parse_vcf([vcf_line(5, "A", "AT")])


def test_vcf_mnp_rejected():
    with pytest.raises(AnnotateError, match="双等位 SNV"):
        parse_vcf([vcf_line(5, "AT", "GC")])


def test_vcf_multiallelic_rejected():
    with pytest.raises(AnnotateError, match="多等位"):
        parse_vcf([vcf_line(5, "A", "T,C")])


def test_vcf_ref_mismatch_rejected():
    with pytest.raises(AnnotateError, match="与参考序列不符"):
        parse_vcf([vcf_line(5, "C", "T")])


def test_vcf_pos_out_of_range_rejected():
    with pytest.raises(AnnotateError, match="超出参考序列范围"):
        parse_vcf([vcf_line(61, "C", "T")])


def test_vcf_ref_eq_alt_rejected():
    with pytest.raises(AnnotateError, match="相同"):
        parse_vcf([vcf_line(5, "A", "A")])


def test_vcf_chrom_mismatch_rejected():
    with pytest.raises(AnnotateError, match="不一致"):
        parse_vcf([vcf_line(5, "A", "T", chrom="chrX")])


def test_vcf_too_many_records_rejected():
    lines = [vcf_line(5, "A", "T")] * (MAX_SNVS + 1)
    with pytest.raises(AnnotateError, match="超过上限"):
        parse_vcf(lines)


# ---------- 注释推演 ----------

def transcripts():
    return parse_cds_tsv(
        "TX1\t+\t5\t10\nTX1\t+\t21\t26\nTX2\t-\t41\t52\nTX3\t+\t5\t16\n", REF
    )


def snv(pos, ref, alt):
    (s,) = parse_vcf([vcf_line(pos, ref, alt)])
    return s


def test_annotate_plus_strand_missense():
    tx1 = transcripts()[0]
    a = annotate(snv(6, "T", "C"), tx1)
    assert (a.ref_codon, a.new_codon) == ("ATG", "ACG")
    assert (a.aa_ref, a.aa_new) == ("Met", "Thr")
    assert a.consequence == "missense"
    assert a.cds_pos == 2 and a.codon_number == 1


def test_annotate_plus_strand_synonymous():
    tx1 = transcripts()[0]
    a = annotate(snv(23, "T", "C"), tx1)
    assert (a.ref_codon, a.new_codon) == ("ACT", "ACC")
    assert a.consequence == "synonymous"


def test_annotate_plus_strand_spliced_codon():
    # 段1 = 5-8（ATGG，4 bp），段2 = 21-25（ACTTA，5 bp）：CDS = ATGGACTTA，
    # 第 2 个密码子 GAC 跨两段：G 在 pos 8（段1末尾），AC 在 pos 21-22（段2开头）
    (tx,) = parse_cds_tsv("TXS\t+\t5\t8\nTXS\t+\t21\t25\n", REF)
    assert tx.cds == "ATGGACTTA"
    a = annotate(snv(8, "G", "A"), tx)
    assert (a.ref_codon, a.new_codon) == ("GAC", "AAC")
    assert (a.aa_ref, a.aa_new) == ("Asp", "Asn")
    assert a.consequence == "missense"
    assert a.cds_pos == 4 and a.codon_number == 2
    # 同一跨段密码子的另一侧（段2开头）
    a2 = annotate(snv(21, "A", "C"), tx)
    assert (a2.ref_codon, a2.new_codon) == ("GAC", "GCC")
    assert (a2.aa_ref, a2.aa_new) == ("Asp", "Ala")
    assert a2.cds_pos == 5 and a2.codon_number == 2


def test_annotate_minus_strand_stop_lost():
    tx2 = transcripts()[1]
    a = annotate(snv(43, "A", "G"), tx2)  # 负链：A→G 即 CDS 上 T→C
    assert (a.ref_codon, a.new_codon) == ("TAA", "CAA")
    assert (a.aa_ref, a.aa_new) == ("Ter", "Gln")
    assert a.consequence == "stop_lost"
    assert a.cds_pos == 10 and a.codon_number == 4


def test_annotate_minus_strand_stop_gained():
    tx2 = transcripts()[1]
    a = annotate(snv(49, "T", "A"), tx2)  # 负链：T→A 即 CDS 上 A→T
    assert (a.ref_codon, a.new_codon) == ("AAA", "TAA")
    assert a.consequence == "stop_gained"


def test_annotate_minus_strand_synonymous():
    tx2 = transcripts()[1]
    a = annotate(snv(47, "T", "C"), tx2)  # CDS 上 A→G：AAA→AAG
    assert (a.ref_codon, a.new_codon) == ("AAA", "AAG")
    assert a.consequence == "synonymous"


def test_annotate_minus_strand_spliced_codon():
    # 负链两段：41-47 与 48-52。转录方向先取 48-52 的反向互补（ATGAA，idx 0-4），
    # 再取 41-47 的反向互补（ATTCTAA，idx 5-11）；第 2 个密码子 AAA 跨两段，
    # 其末位（idx 5）落在基因组 pos 47
    (tx,) = parse_cds_tsv("TXS\t-\t41\t47\nTXS\t-\t48\t52\n", REF)
    assert tx.cds == "ATGAAATTCTAA"
    a = annotate(snv(47, "T", "C"), tx)  # 负链：T→C 即 CDS 上 A→G
    assert (a.ref_codon, a.new_codon) == ("AAA", "AAG")
    assert a.consequence == "synonymous"
    assert a.cds_pos == 6 and a.codon_number == 2


def test_annotate_stop_to_stop_is_synonymous():
    tx2 = transcripts()[1]
    a = annotate(snv(41, "T", "C"), tx2)  # CDS 上 A→G：TAA→TAG
    assert (a.ref_codon, a.new_codon) == ("TAA", "TAG")
    assert (a.aa_ref, a.aa_new) == ("Ter", "Ter")
    assert a.consequence == "synonymous"


def test_annotate_miss_returns_none():
    tx1 = transcripts()[0]
    assert annotate(snv(30, "T", "A"), tx1) is None


def test_annotate_all_multi_transcript_and_outside():
    txs = transcripts()
    snvs = parse_vcf([vcf_line(5, "A", "T"), vcf_line(30, "T", "A")])
    rows = annotate_all(snvs, txs)
    hit5 = [r for r in rows if r.snv.pos == 5]
    assert {r.transcript for r in hit5} == {"TX1", "TX3"}  # 同一 SNV 命中两个转录本
    outside = [r for r in rows if r.snv.pos == 30]
    assert len(outside) == 1 and outside[0].consequence == "outside_cds"
    assert outside[0].transcript is None


# ---------- 网页集成 ----------

@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _annotation_files(vcf=None, cds=None, fasta=None):
    return {
        "fasta": ("ref.fa", fasta if fasta is not None else (EXAMPLES / "ann_ref.fa").read_bytes(), "text/plain"),
        "vcf": ("variants.vcf", vcf if vcf is not None else (EXAMPLES / "ann_variants.vcf").read_bytes(), "text/plain"),
        "cds": ("cds.tsv", cds if cds is not None else (EXAMPLES / "ann_cds.tsv").read_bytes(), "text/plain"),
    }


def test_annotate_index_loads(client):
    resp = client.get("/annotate")
    assert resp.status_code == 200
    assert "上传新注释" in resp.text


def test_successful_annotation(client):
    resp = client.post("/annotations", files=_annotation_files(), follow_redirects=False)
    assert resp.status_code == 303
    page = client.get(resp.headers["location"])
    assert page.status_code == 200
    # 五种结论都出现
    for label in ("同义", "错义", "终止获得", "终止丢失", "CDS 外"):
        assert label in page.text
    # 命中多个转录本的 SNV 每个转录本各占一行
    assert "TX1" in page.text and "TX2" in page.text and "TX3" in page.text
    # 原/新密码子与氨基酸
    assert "ATG" in page.text and "ACG" in page.text
    assert "Met" in page.text and "Thr" in page.text
    assert "TAA" in page.text and "CAA" in page.text
    # 列表页可见
    index = client.get("/annotate")
    assert "完成" in index.text


def test_invalid_cds_fails_and_old_reports_survive(client):
    ok = client.post("/annotations", files=_annotation_files(), follow_redirects=False)
    good_url = ok.headers["location"]
    ok2 = client.post(
        "/reports",
        files={
            "fasta": ("ref.fa", (EXAMPLES / "ref.fa").read_bytes(), "text/plain"),
            "vcf_a": ("a.vcf", (EXAMPLES / "lab_a.vcf").read_bytes(), "text/plain"),
            "vcf_b": ("b.vcf", (EXAMPLES / "lab_b.vcf").read_bytes(), "text/plain"),
        },
        follow_redirects=False,
    )
    compare_url = ok2.headers["location"]

    bad = _annotation_files(cds=b"TX1\t+\t5\t10\nTX1\t-\t21\t26\n")  # 同一转录本正负链不一致
    resp = client.post("/annotations", files=bad, follow_redirects=False)
    assert resp.status_code == 303
    page = client.get(resp.headers["location"])
    assert "失败" in page.text
    assert "同向" in page.text
    # 失败报告不显示任何注释结果表
    assert "注释结果" not in page.text

    # 旧注释报告与旧比对报告都不受影响
    assert "注释结果" in client.get(good_url).text
    assert "共有位点（4）" in client.get(compare_url).text


def test_invalid_vcf_fails_whole_annotation(client):
    bad = _annotation_files(vcf=(VCF_HEADER + vcf_line(5, "A", "AT") + "\n").encode())
    resp = client.post("/annotations", files=bad, follow_redirects=False)
    page = client.get(resp.headers["location"])
    assert "失败" in page.text
    assert "双等位 SNV" in page.text
    assert "注释结果" not in page.text


def test_annotation_report_not_found(client):
    assert client.get("/annotations/999").status_code == 404


def _ann_batch(i: int):
    """构造第 i 批注释数据：参考序列名、转录本、SNV 坐标都按批次唯一。"""
    name = f"chrA{i}"
    # 1-6 为 CDS（ATG+TAA 之间夹一个按批次变化的密码子），7-20 为非编码区
    codons = ["GCA", "GCT", "GCG", "GCC", "GGA", "GGT"]
    seq = "ATG" + codons[i] + "TAA" + "C" * 11  # 20 bp
    fasta = f">{name}\n{seq}\n".encode()
    cds = f"TX{i}\t+\t1\t9\n".encode()
    # pos 4-6 的碱基按批次不同，ALT 统一取与该 REF 不同的固定碱基
    ref_base = seq[3]
    alt = {"A": "C", "C": "A", "G": "A", "T": "A"}[ref_base]
    vcf = (VCF_HEADER + f"{name}\t4\t.\t{ref_base}\t{alt}\t.\t.\t.\n").encode()
    return fasta, vcf, cds, name


def test_concurrent_annotations_do_not_mix_batches(tmp_path):
    """6 批不同参考序列/转录本并发出传，逐份报告核对没有串批次。"""
    import asyncio

    import httpx

    app = create_app(str(tmp_path / "concurrent_ann.db"))
    n = 6

    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://t", follow_redirects=True
        ) as ac:
            async def post(i):
                fasta, vcf, cds, _ = _ann_batch(i)
                resp = await ac.post(
                    "/annotations",
                    files={
                        "fasta": ("ref.fa", fasta, "text/plain"),
                        "vcf": ("v.vcf", vcf, "text/plain"),
                        "cds": ("c.tsv", cds, "text/plain"),
                    },
                )
                assert resp.status_code == 200
                return resp

            return await asyncio.gather(*(post(i) for i in range(n)))

    pages = asyncio.run(run())
    seen_urls = set()
    for i, page in enumerate(pages):
        _, _, _, name = _ann_batch(i)
        assert name in page.text, f"注释报告参考序列名串批次: {page.url}"
        assert f"TX{i}" in page.text
        for j in range(n):
            if j != i:
                assert f"TX{j}" not in page.text
                assert f"chrA{j}" not in page.text
        seen_urls.add(str(page.url))
    assert len(seen_urls) == n  # 6 份独立报告
