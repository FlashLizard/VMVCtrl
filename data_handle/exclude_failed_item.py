import json
import os

failed_list =  ['104/532229', '99/507529', '51/266599', '93/475684', '109/559284', '109/557401', '112/573555', '92/474413', '148/750904', '24/133842', '20/114234', '84/434753', '54/281397', '21/118423', '27/146190', '118/603720', '55/289066', '32/170692', '49/258810']
input_dir = 'food'
output_dir = 'food'
for dir in failed_list:
    os.system(f'rm -rf {os.path.join(output_dir, dir)}')


dirs1 = os.listdir(input_dir)
dirs2 = []
cnt = 5
for dir in dirs1:
    if os.path.isdir(os.path.join(input_dir, dir)):
        path = os.path.join(input_dir, dir)
        sub_dirs = os.listdir(path)
        for sub_dir in sub_dirs:
            if cnt == 0:
                break
            if os.path.isdir(os.path.join(path, sub_dir)):
                dirs2.append(os.path.join(dir, sub_dir))
                cnt -= 1

with open(f'{output_dir}/small_data.json', 'w') as json_file:
    json.dump(dirs2, json_file)