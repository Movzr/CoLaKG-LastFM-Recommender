# lastfm_utils.py
# 辅助读取 CoLaKG 项目里 data/lastfm 的各种文件
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "lastfm"


def load_lastfm_train(path: Optional[Path] = None) -> Dict[int, List[int]]:
    if path is None:
        path = DATA_DIR / "train.txt"
    user2items: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]
            user2items[u] = items
    return user2items


def load_lastfm_test(path: Optional[Path] = None) -> Dict[int, List[int]]:
    if path is None:
        path = DATA_DIR / "test.txt"
    user2items: Dict[int, List[int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            u = int(parts[0])
            items = [int(x) for x in parts[1:]]
            user2items[u] = items
    return user2items


def load_item_map(path: Optional[Path] = None) -> Dict[int, int]:
    if path is None:
        path = DATA_DIR / "item_map.txt"
    raw2mapped: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            raw_id = int(parts[0])
            mapped_id = int(parts[1])
            raw2mapped[raw_id] = mapped_id
    return raw2mapped


def load_user_map(path: Optional[Path] = None) -> Dict[int, int]:
    if path is None:
        path = DATA_DIR / "user_map.txt"
    raw2mapped: Dict[int, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            raw_id = int(parts[0])
            mapped_id = int(parts[1])
            raw2mapped[raw_id] = mapped_id
    return raw2mapped


def load_artists(path: Optional[Path] = None) -> Dict[int, Tuple[str, str, str]]:
    if path is None:
        path = DATA_DIR / "artists.dat"

    artists: Dict[int, Tuple[str, str, str]] = {}

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()

        # 判断是否是表头（有时候文件没有表头）
        def add_from_line(line: str):
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                aid = int(parts[0])
                name, url, pic = parts[1], parts[2], parts[3]
                artists[aid] = (name, url, pic)

        if not first.lower().startswith("id"):
            add_from_line(first)

        for line in f:
            line = line.strip()
            if not line:
                continue
            add_from_line(line)

    return artists


def build_mapped_item_metadata() -> Dict[int, dict]:
    raw2mapped = load_item_map()
    artists = load_artists()

    mapped_meta: Dict[int, dict] = {}

    for raw_id, mapped_id in raw2mapped.items():
        if raw_id in artists:
            name, url, pic = artists[raw_id]
        else:
            name, url, pic = f"Artist_{raw_id}", "", ""

        mapped_meta[mapped_id] = {
            "raw_id": raw_id,
            "name": name,
            "url": url,
            "pictureURL": pic,
        }

    return mapped_meta

def load_tags(path: Optional[Path] = None) -> Dict[int, str]:
    """
    读取 tags.dat
    支持 utf-8 / latin-1 / 忽略非法字符
    """

    if path is None:
        path = DATA_DIR / "tags.dat"

    tag_id2name: Dict[int, str] = {}

    # 尝试三种方式
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        # 实在不行就忽略非法字符
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

    # 判断是否有表头
    def add_line(line: str):
        parts = line.strip().split()
        if len(parts) >= 2:
            tid = int(parts[0])
            tag = " ".join(parts[1:])
            tag_id2name[tid] = tag

    # 去掉表头
    first = lines[0].strip()
    start = 0
    if first.lower().startswith("tagid"):
        start = 1

    # 处理每行
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        add_line(line)

    return tag_id2name



def load_user_taggedartists(path: Optional[Path] = None) -> List[Tuple[int, int, int]]:
    """
    读取 user_taggedartists.dat
    格式：userID artistID tagID timestamp
    返回列表 [(user_id, artist_id, tag_id), ...]
    """
    if path is None:
        path = DATA_DIR / "user_taggedartists.dat"

    records: List[Tuple[int, int, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline().strip()

        def add_line(line: str):
            parts = line.strip().split()
            if len(parts) >= 3:
                uid = int(parts[0])
                aid = int(parts[1])
                tid = int(parts[2])
                records.append((uid, aid, tid))

        if not first.lower().startswith("userid"):
            add_line(first)

        for line in f:
            line = line.strip()
            if not line:
                continue
            add_line(line)

    return records


def build_tag_to_mapped_items() -> Dict[str, List[int]]:
    """
    构建：tag_name -> [mapped_item_id, ...]
    利用：
      - tags.dat:        tagID -> tagValue
      - user_taggedartists.dat: artistID + tagID
      - item_map.txt:   original_artist_id -> mapped_item_id
    """
    tag_id2name = load_tags()
    records = load_user_taggedartists()
    raw2mapped = load_item_map()

    # 先 tag_id 维度聚合 raw artist
    tagid_to_raw_items: Dict[int, Set[int]] = {}
    for _, aid, tid in records:
        if tid not in tag_id2name:
            continue
        if aid not in raw2mapped:
            continue
        if tid not in tagid_to_raw_items:
            tagid_to_raw_items[tid] = set()
        tagid_to_raw_items[tid].add(aid)

    # 转成 tag_name -> mapped_item_id 列表
    tagname_to_items: Dict[str, List[int]] = {}
    for tid, raw_set in tagid_to_raw_items.items():
        tag_name = tag_id2name.get(tid, None)
        if tag_name is None:
            continue
        mapped_list: List[int] = []
        for raw_id in raw_set:
            mapped_id = raw2mapped.get(raw_id, None)
            if mapped_id is not None:
                mapped_list.append(mapped_id)
        if mapped_list:
            tagname_to_items[tag_name] = mapped_list

    return tagname_to_items



if __name__ == "__main__":
    user2items = load_lastfm_train()
    print(f"Train users: {len(user2items)}")

    meta = build_mapped_item_metadata()
    print(f"Mapped items: {len(meta)}")

    example_key = next(iter(meta.keys()))
    print("Example item:", example_key, meta[example_key])

    tag2items = build_tag_to_mapped_items()
    print("Tags:", len(tag2items))
    ex_tag = next(iter(tag2items.keys()))
    print("Example tag:", ex_tag, "items:", len(tag2items[ex_tag]))
