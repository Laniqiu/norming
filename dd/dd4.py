"""
数据处理等
检查
"""


if __name__ == "__main__":
    from pathlib import Path
    import pandas as pd
    root = Path("/Users/laniqiu/My Drive")
    files = root.joinpath("parsed_res/").glob("*")

    # 统计每个upos和xpos的内容
    for f in files:
        if f.name.startswith("."):  # filter .* files
            continue
        data = pd.read_table(f, sep="\t", head=None)
        # 一些分词、词性标注结果需要纠正










