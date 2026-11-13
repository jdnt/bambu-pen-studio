# 贡献指南

感谢你对手写字体与画笔排版的兴趣。

## 提交前

1. 不要提交任何个人字形、照片、模型权重、字体文件或 Token。
2. 新功能请附最小复现步骤或截图。
3. 涉及训练数据的改动，请在 PR 描述中说明数据来源与授权。
4. 字体产物请遵守 `NOTICE.md` 中的 zi2zi-JiT 署名要求。

## 开发

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python webapp/server.py
```

## 提交信息

建议使用：

```text
feat: ...
fix: ...
docs: ...
```

请保持改动聚焦，避免把训练数据、生成字体和截图混在代码提交里。
