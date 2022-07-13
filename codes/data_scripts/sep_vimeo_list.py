import os, shutil

if __name__ ==  "__main__":

    #### TODO: change your file path herec
    inPath = '/media/data3/wh/data/dataset/SR/vimeo_septuplet/sequences/'
    outPath = '/media/data3/wh/data/dataset/SR/vimeo_septuplet/sequences/test/'
    guide = '/media/data3/wh/data/dataset/SR/vimeo_septuplet/sep_testlist.txt'
    
    f = open(guide, "r")
    lines = f.readlines()
    
    if not os.path.isdir(outPath):
        os.mkdir(outPath)

    for l in lines:
        line = l.replace('\n','')
        this_folder = os.path.join(inPath, line)
        dest_folder = os.path.join(outPath, line)
        print(this_folder)
        shutil.copytree(this_folder, dest_folder)
    print('Done')
