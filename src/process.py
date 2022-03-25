import jieba
import json
import jieba.posseg
ans = set()

with open("/home/xdx/MedicalImageCaption/data2/data2.json", "r") as fr:
    datas = json.load(fr)
    for data in datas:
        res = data["皮肤镜描述"].split("。")[-2]
        
        ans.add(res)
print(ans)
