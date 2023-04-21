


import os


def to_server_path(data_file):
    dst_img_list = []
    with open(data_file, 'r') as f:
        for line in f:
            words = line.strip().split(' ')
            img_path = words[0]
            if not os.path.exists(img_path):
                print('not exist ', img_path)
            pws = img_path.split('/')
            #pre = '/home/jovyan/datasets/suitcase_reid/train_dataset/'
            pre = '/home/jovyan/data-vol-1/car_attributes/'
            dst_path = pre + '/'.join(pws[5:])
            dst_item = ' '.join([dst_path]+words[1:])
            dst_img_list.append(dst_item)
    print('get img list')

    data_file_path = data_file.split('/')
    data_file_name = data_file_path[-1]
    dst_file_name = data_file_name[:-4]+'_server.txt'
    dst_file_path = '/'.join(data_file_path[:-1]) + '/' + dst_file_name

    with open(dst_file_path, 'w') as f:
        f.write('\n'.join(dst_img_list))
        f.write('\n')


data_dir = '../dataset/front_dataset'
for f in os.listdir(data_dir):
    if 'org.txt' not in f:
        continue
    f_path = os.path.join(data_dir, f)
    to_server_path(f_path)
    print('done', f)



