调用命令行:
    python3 main.py --use_gpu 1 --channels 3 --checkpoint checkpointdir --out_dir outdir --image imagepath

参数说明:
    --use_gpu 是否使用gpu
    --channels 图片的通道数，彩色图片为3
    --checkpoint 模型文件存放位置
    --out_dir 图片输出目录
    --image 输入图片位置

将python文件编译成.so的动态加载库
	python3 setup.py build_ext --inplace