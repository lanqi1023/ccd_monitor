# CCD Web Monitor v2.0

安装 [Galaxy SDK](https://www.daheng-imaging.com/list-58-1.html)

修改 ccd.py:2 为 Galaxy SDK 安装路径
```
sys.path.append(r'_PATH_TO_GALAXY_SDK_')
```

安装 Python 依赖
```
pip install -r requirements.txt
```

运行服务
```
python src/server.py
```

CCD Monitor: <http://127.0.0.1:8000>
Input:       <http://127.0.0.1:8000/input>
Weight:      <http://127.0.0.1:8000/weight>
