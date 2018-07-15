rem pyinstaller  -i "quill.ico" "laser_marker_finder.py"
pyinstaller -F -i "quill.ico" "laser_blob_withoutGUI.py"
rem -F 生成单个执行文件 -w取消命令行窗口 -i指明可执行文件图标h