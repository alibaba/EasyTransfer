# with open('./data/sample_data.tsv', 'r') as file:
#     data = file.readlines()
#
# all_data = []
# for item in data:
#     text, label = item.strip().split('\t')
#     all_data.append(
#         {
#             'text': text,
#             'label': label
#         }
#     )
# task_split_result = []
# count = 0
# while count <= len(all_data)-1:
#     task_split_result.append(all_data[count: count+300])
#     count += 300
# all_data_store = []
# all_data_meta = []
# for index, dataset in enumerate(task_split_result):
#     label_set = []
#     text_set = []
#     for item in dataset:
#         text = item['text']
#         label = item['label']
#         label_set.append(label)
#         text_set.append(item)
#     label_set = set(label_set)
#     all_data_store.append(
#         {
#             'taskKey': 'text_classify_' + str(index),
#             'dataset': text_set
#         }
#     )
#     all_data_meta.append(
#         {
#             "taskKey": 'text_classify_' + str(index),
#             "labelMap": label_set,
#             "taskIndex": index
#         }
#     )
# # store base task
# import json
# with open('./data/base_tasks.json', 'w') as file:
#     json.dump(str({"data":all_data_meta[:40]}), file)
# # store meta info
# with open('./data/meta_info.json', 'w') as file:
#     json.dump(str({"data":all_data_meta}), file)
# # store all data
# with open('./data/all_data.json', 'w') as file:
#     json.dump(str({"data":all_data_store}), file)

import json

with open('./data/meta_info.json', 'r') as file:
    meta_data = eval(json.load(file))['data']
import os
for index, task in enumerate(meta_data):
    base_dir = './MeLL_pytorch/data/'
    if index >=40:
        taskKey = task['taskKey']
        base_dir += taskKey
        os.makedirs(base_dir)
        with open(base_dir+'/'+'lifelong_task.json', 'w') as file:
            json.dump(str({"data":task}), file)