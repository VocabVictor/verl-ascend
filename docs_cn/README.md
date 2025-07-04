# verl 文档

## 构建文档

```bash
# If you want to view auto-generated API docstring, please make sure verl is available in python path. For instance, install verl via:
# pip install .. -e[test]

# 安装构建文档所需的依赖
pip install -r requirements-docs.txt

# 构建文档
make clean
make html
```

## 使用浏览器打开文档

```bash
python -m http.server -d _build/html/
```
Launch your browser and navigate to http://localhost:8000 to view the documentation. Alternatively you could drag the file `_build/html/index.html` to your local browser and view directly.
