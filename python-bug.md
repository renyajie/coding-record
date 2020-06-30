# pip速度过慢:
    https://www.cnblogs.com/sunnydou/p/5801760.html

# vscode Importing the numpy c-extensions failed.
    pip uninstall -y numpy
    pip uninstall -y setuptools
    pip install setuptools
    pip install numpy

# vscode ImportError: DLL load failde: XXXX
    在环境变量中加入Anaconda的三个路径：
    D:\xxx\Anaconda3;
    D:\xxx\Anaconda3\Scripts;
    D:\xxx\Anaconda3\Library\bin;
    https://blog.csdn.net/m0_37586991/article/details/88709139