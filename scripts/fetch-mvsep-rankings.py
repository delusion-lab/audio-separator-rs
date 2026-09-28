#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓取 MVSEP multisong leaderboard 快照，更新 rankings.json 的 mvsep 分节。

用法:
    python scripts/fetch-mvsep-rankings.py [--proxy http://127.0.0.1:12355]
                                           [--out <rankings.json 路径>]
                                           [--views instrum,vocals,bass,drums,other]
                                           [--top 20]

说明:
    - rankings.json 的 community 分节保持不变（由 deton24 社区指南维护，手动更新）；
      mvsep 分节整体替换为本脚本抓取的最新快照。
    - 抓取 5 个排序视图（instrum/vocals/bass/drums/other）后按算法名合并去重；
      同模型在不同视图出现的名次都记录到 views，各声部 SDR 跨视图补全。
    - leaderboard 为服务端渲染 HTML：每行 Rating / QC ID / 算法名 / Info / Ensemble /
      5 个 SDR 列（Bass/Drums/Other/Vocals/Instrumental）；部分列显示 '---' 表示该行无此数据。
    - 代理解析顺序: --proxy > ALL_PROXY > HTTPS_PROXY > HTTP_PROXY。
    - 仅标准库（urllib + 正则），无第三方依赖。
"""
import argparse
import json
import os
import re
import sys
import urllib.request

BASE_URL = "https://mvsep.com"
QUALITY_CHECKER = "/quality_checker/multisong_leaderboard"
VIEWS = ["instrum", "vocals", "bass", "drums", "other"]
# 固定表头 10 列：0 Rating 1 ID 2 name 3 Info 4 Ensemble 5-9 SDR×5
COL_MAP = {"bass": 5, "drums": 6, "other": 7, "vocals": 8, "instrumental": 9}


def resolve_proxy(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    for var in ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(var)
        if v and v.strip():
            return v.strip()
    return None


def fetch(url: str, proxy: str | None) -> str:
    handlers = []
    if proxy:
        handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        handlers.append(handler)
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers={"User-Agent": "audio-separator-rs/rankings-fetcher"})
    with opener.open(req, timeout=30) as resp:
        data = resp.read()
    return data.decode("utf-8", errors="replace")


def clean(tag: str) -> str:
    return re.sub(r"<[^>]+>", "", tag).strip()


def parse_view(html: str) -> list[dict]:
    """解析单个视图页面：返回 [{name, entry_id, url, sdr, rank}]，过滤数据集基准行。"""
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.S)
    out = []
    for r in rows:
        tds = re.findall(r"<td[^>]*>(.*?)</td>", r, re.S)
        if len(tds) < 10:
            continue
        name = clean(tds[2])
        if not name or "Multisong dataset" in name:
            continue
        try:
            rank = int(clean(tds[0]))
        except ValueError:
            continue
        entry_id = clean(tds[1])
        sdr = {}
        for col, idx in COL_MAP.items():
            txt = clean(tds[idx])
            if txt and txt != "---":
                try:
                    sdr[col] = float(txt)
                except ValueError:
                    pass
        url_m = re.search(r'href="([^"]+)"', tds[2])
        url = url_m.group(1) if url_m else (
            f"{BASE_URL}/quality_checker/entry/{entry_id}" if entry_id.isdigit() else None
        )
        out.append({"name": name, "rank": rank, "entry_id": entry_id, "url": url, "sdr": sdr})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch MVSEP multisong leaderboard snapshot into rankings.json")
    ap.add_argument("--proxy", default=None, help="HTTP proxy (e.g. http://127.0.0.1:12355); env vars fallback")
    ap.add_argument("--out", default=None, help="rankings.json path (default: repo-root/rankings.json next to this script)")
    ap.add_argument("--views", default=",".join(VIEWS), help=f"comma-separated sort views (default: {','.join(VIEWS)})")
    ap.add_argument("--top", type=int, default=0, help="rows per view to keep (0 = all parsed rows)")
    ap.add_argument("--base-url", default=BASE_URL, help=f"MVSEP base URL (default: {BASE_URL})")
    args = ap.parse_args()

    out_path = args.out or os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rankings.json"))
    views = [v.strip() for v in args.views.split(",") if v.strip()]

    proxy = resolve_proxy(args.proxy)
    if proxy:
        print(f"proxy: {proxy}")

    # 1) 抓取各视图
    by_name: dict[str, dict] = {}
    for view in views:
        url = f"{args.base_url}{QUALITY_CHECKER}?sort={view}"
        print(f"fetching {url}")
        html = fetch(url, proxy)
        for e in parse_view(html):
            merged = by_name.setdefault(e["name"], {
                "name": e["name"], "entry_id": e["entry_id"], "url": e["url"],
                "sdr": {}, "views": [],
            })
            for col, val in e["sdr"].items():
                if merged["sdr"].get(col) is None:
                    merged["sdr"][col] = val
            merged["views"].append({"view": view, "rank": e["rank"]})

    mvsep = []
    for name in sorted(by_name):
        e = by_name[name]
        mvsep.append({
            "name": e["name"],
            "entry_id": int(e["entry_id"]) if e["entry_id"].isdigit() else None,
            "url": e["url"],
            "sdr": e["sdr"],
            "views": sorted(e["views"], key=lambda v: v["rank"]),
        })
    if args.top > 0:
        mvsep = mvsep[:args.top]
    print(f"parsed {len(mvsep)} unique algorithms across {len(views)} views")

    # 2) 合并回 rankings.json（保留 community 分节）
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            doc = json.load(f)
    else:
        doc = {"version": 1, "community": [], "mvsep": []}
    doc["mvsep"] = mvsep

    # 3) 写回
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"written {out_path} (community {len(doc.get('community', []))} kept, mvsep {len(mvsep)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
