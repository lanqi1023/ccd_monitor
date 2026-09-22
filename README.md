# CCD Web Monitor v2.1

1. 安装 [Galaxy SDK](https://www.daheng-imaging.com/list-58-1.html)

2. 确认环境变量包含 `GALAXY_SDK_DEVELOPMENT`
    ``` powershell
    echo $env:GALAXY_SDK_DEVELOPMENT
    ```

3. 安装 Python 依赖
    ``` powershell
    pip install -r requirements.txt
    ```

4. 运行服务
    ``` powershell
    python src/server.py
    ```

    CCD Monitor: <http://127.0.0.1:8000>
    Input:       <http://127.0.0.1:8000/input>
    Weight:      <http://127.0.0.1:8000/weight>
