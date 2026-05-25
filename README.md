# 环境
使用uv管理环境
```bash
# 如果没有uv, 先安装uv
pip install uv

# 创建虚拟环境并安装依赖
uv sync

# 添加与删除依赖
uv add <package_name>
uv remove <package_name>

# 输出requirements.txt
uv export -r > requirements.txt

# 运行项目
uv run <filename>
```

# 数据集
数据集命名为`WMRC_general`文件夹放在根目录后可以使用`load_global.py`检查数据集是否正确加载
```bash
uv run load_global.py --pt_dir ./WMRC_general --check
```

