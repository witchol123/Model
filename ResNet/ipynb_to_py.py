import os


def ipynb_to_py(ipynb_file, py_ext='.py'):
    py_file = os.path.splitext(ipynb_file)[0] + py_ext
    # 如果文件中有相同文件存在
    if os.path.exists(py_file):
        os.rename(py_file, py_file+'(1)')
        return None
    command = 'jupyter nbconvert --to script %s' % (ipynb_file, )
    os.system(command)


fld_path = 'D:\\Code\\item_structer\\ResNet'
extensionName = 'ipynb'

i = 0
for filename in os.listdir(fld_path):
    if filename.endswith(extensionName):
        i += 0
        print('正在转换第%s个文件:%s' % (i, filename))
        ipynb_to_py("train.ipynb")
