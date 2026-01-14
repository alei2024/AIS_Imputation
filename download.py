import tarfile  # 处理.tar.gz压缩包的解压缩
import zipfile  # 处理.zip压缩包的解压缩
import sys      # 获取命令行参数
import os       # 操作系统交互（创建目录等）
import wget     # 简单的文件下载工具,专门用于从网络 URL 下载文件到本地
import requests # 发送HTTP请求，用于文件下载(如果wget满足不了需求，可使用更强大的requests)
import pandas as pd  # 数据处理（读取CSV、计算统计量）
import pickle   # 序列化保存Python对象（均值/标准差）;高效保存与加载Python对象

os.makedirs("data/", exist_ok=True)
if sys.argv[1] == "physio":
    url = "https://physionet.org/files/challenge-2012/1.0.0/set-a.tar.gz?download"
    wget.download(url, out="data")
    with tarfile.open("data/set-a.tar.gz", "r:gz") as t:
        t.extractall(path="data/physio")

elif sys.argv[1] == "pm25":
    #已手动下载好
    url = "https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/STMVL-Release.zip"
    urlData = requests.get(url).content
    filename = "data/STMVL-Release.zip"
    with open(filename, mode="wb") as f:
        f.write(urlData)
    with zipfile.ZipFile(filename) as z:
        z.extractall("data/pm25")
        
    def create_normalizer_pm25():
        df = pd.read_csv(
            "./data/pm25/Code/STMVL/SampleData/pm25_ground.txt",
            index_col="datetime",
            parse_dates=True,
        )
        #parse_dates=True自动将datetime列的字符串时间戳解析为 Pandas 的 Datetime 类型。
        #效果：可以直接使用df.index.month、df.index.year等时间属性，这是后续按月份过滤的基础。
        test_month = [3, 6, 9, 12]
        for i in test_month:
            df = df[df.index.month != i]
        mean = df.describe().loc["mean"].values
        std = df.describe().loc["std"].values
        path = "./data/pm25/pm25_meanstd.pk"
        with open(path, "wb") as f:
            pickle.dump([mean, std], f)
    create_normalizer_pm25()
