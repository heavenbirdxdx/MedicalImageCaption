# -*- coding: utf-8 -*-
import json
from transformers import BertTokenizer, BertModel
import tqdm
'''
通过皮肤病结构化信息生成知识图谱，生成graph.content和graph.cites两个文件
graph.content格式：
<keyword_id>    <keyword_string>    <keyword_embedding>     <keyword_label>
解释：
<keyword_id>：关键词的id
<keyword_string>：关键词内容
<keyword_embedding>：关键词的bert768维embedding
<keyword_label>：关键词的类别（0：根结点，1：疾病类型，2：性状，3：颜色，4：位置，5：形状，6：伴生症状，7：治疗方式）

graph.sites格式：
<keyword_id1> <keyword_id2>
解释：
关键词1（疾病类型） 具有    关键词2（其他关键词）   的性质
'''

tokenizer = BertTokenizer.from_pretrained('../bert-base-chinese')
model = BertModel.from_pretrained('../bert-base-chinese')

# 存储graph.content
keyword_dict = dict()

with open('/home/xdx/project-MedicalImageCpation/graph/keyword.json', 'r') as fr:
    keyword_infos = json.load(fr)

id = 0
with open('../graph/graph.content', 'w') as fw:
    for keyword_info in tqdm.tqdm(keyword_infos):
        for keyword in keyword_info["keyword"]:
            if keyword not in keyword_dict.keys():
                keyword_dict[keyword] = {}
                keyword_dict[keyword]['id'] = id
                keyword_dict[keyword]['str'] = keyword
                inputs = tokenizer(keyword, return_tensors='pt')
                outputs = model(**inputs)
                keyword_dict[keyword]['embedding'] = outputs.pooler_output
                keyword_dict[keyword]['label'] = keyword_info["id"]
                id += 1
                print(keyword_dict[keyword]['embedding'].shape)
                rs = " ".join(list(map(str, keyword_dict[keyword]['embedding'].detach().numpy().tolist()[0])))
                fw.write("{}\t{}\t{}\t{}\n".format(keyword_dict[keyword]['id'],keyword_dict[keyword]['str'],rs,keyword_dict[keyword]['label']))
# print(keyword_dict)
    
with open('/home/xdx/MedicalImageCaption/data2/data2.json', 'r') as fr:
    disease_infos = json.load(fr)

pairset = set()

with open('../graph/graph.cites', 'w') as fw:
    fw.write("{}\t{}\n".format(0, 1))
    fw.write("{}\t{}\n".format(0, 2))
    fw.write("{}\t{}\n".format(0, 3))
    fw.write("{}\t{}\n".format(0, 4))
    # 遍历每一个病人信息
    for disease_info in disease_infos:
        # keyword_id1是当前疾病的id
        keyword_id1 = keyword_dict[disease_info["病理诊断"]]["id"]
        # caption是皮肤病描述
        caption = disease_info["皮肤镜描述"]
        # 遍历全部的keyword
        for keyword_info in keyword_infos:
            # 跳过root和疾病类型
            if keyword_info["id"] == 0 or keyword_info["id"] == 1:
                continue
            for keyword in keyword_info["keyword"]:
                if keyword in caption:
                    keyword_id2 = keyword_dict[keyword]["id"]
                    pairset.add((keyword_id1, keyword_id2))
    for pair in pairset:
        fw.write("{}\t{}\n".format(pair[0], pair[1]))
