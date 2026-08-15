NVC Realtime - Studio One 安装与使用说明
========================================

适用系统
--------
- Windows 10/11 64 位
- Studio One 64 位
- 用户已经拥有可正常运行的 NVC 源码和 Python/CUDA 环境整合包

压缩包内容
----------
VST2\
  NVC Realtime.dll
  NVCRealtime.resources\worker\nvc_worker.py

VST3\
  NVCRealtime.vst3\

VST2 安装方法
-------------
1. 把 VST2 文件夹中的以下两个项目一起复制到 Studio One 已扫描的 VST2 插件目录：
   - NVC Realtime.dll
   - NVCRealtime.resources 文件夹
2. DLL 和 NVCRealtime.resources 必须保持同级，不能只复制 DLL。
3. 常见的自定义 VST2 目录示例：C:\VSTPlugins
4. 在 Studio One 的“选项/位置/VST 插件”中确认该目录已加入扫描列表，然后重新扫描插件。

正确示例：
C:\VSTPlugins\NVC Realtime.dll
C:\VSTPlugins\NVCRealtime.resources\worker\nvc_worker.py

VST3 安装方法
-------------
1. 把 VST3 文件夹中的整个 NVCRealtime.vst3 文件夹复制到：
   C:\Program Files\Common Files\VST3\
2. 不要只复制 Contents 里面的单个文件。
3. 复制后在 Studio One 中重新扫描插件。

正确示例：
C:\Program Files\Common Files\VST3\NVCRealtime.vst3\Contents\x86_64-win\NVCRealtime.vst3

首次使用
--------
1. 在 Studio One 的音轨上加载“NVC Realtime”。
2. 点击 NVC ROOT，选择已有 NVC 源码与环境整合包的根目录。
3. 插件会自动检测该目录下的 runtime\python.exe。
4. PYTHON 保持空白时，手动选择整合包中可用的 64 位 python.exe。
5. 选择 .pth 模型；需要索引检索时再选择 .index 文件。
6. 点击右下角 ENGINE。状态变成 READY 后开始输出变声结果。

NVC 整合包要求
-------------
所选根目录至少需要包含：
- infer\rtnvc.py
- configs\config.py
- Python 环境及其依赖
- HuBERT、RMVPE 等原 NVC 实时推理所需文件
- 用户选择的 .pth 模型

配置与日志
----------
- 最后一次成功启动的路径配置保存在：
  %LOCALAPPDATA%\NVCRealtime\settings.ini
- 运行日志保存在：
  %TEMP%\NVCRealtime\logs\
- Worker 的临时 JSON 配置也保存在上述临时目录，系统清理临时文件时可自动删除。
- VST2 与 VST3 共用最后一次成功配置。
- 删除 settings.ini 可以清除已保存的路径。

常见问题
--------
- Studio One 找不到 VST2：确认已添加 DLL 所在目录并重新扫描。
- Studio One 找不到 VST3：确认复制的是整个 .vst3 文件夹。
- VST2 提示 worker resource is missing：NVCRealtime.resources 没有与 DLL 放在同一目录。
- 启动提示 Python 错误：重新选择 NVC ROOT，并检查 runtime\python.exe 或手动选择 Python。
- 启动提示模型或源码缺失：检查界面选择路径以及 NVC 整合包内容。
- 系统提示缺少 MSVC DLL：安装 Microsoft Visual C++ 2015-2022 Redistributable x64。

操作说明
--------
- 单击并拖动滑块：调整参数。
- 双击滑块数值：直接输入数字。
- BLOCK 可选范围为 20~1000 ms。
- CROSSFADE 可选范围为 10~100 ms。
- 与原版实时 GUI 一致，实际 SOLA Crossfade 为 CROSSFADE 和 40 ms 中的较小值，与 BLOCK 独立。
- 例如 BLOCK 20 ms、CROSSFADE 100 ms 时，实际 SOLA Crossfade 为 40 ms。
- 状态栏的 actual CF 会显示当前实际使用的 Crossfade。
