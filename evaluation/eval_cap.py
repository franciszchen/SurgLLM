import json

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

with open("caption_result.json","r") as f:
    dataset = json.load(f)

annotations = {}
annotations["info"] = {
        "description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
        "url": "http://mscoco.org",
        "version": "1.0",
        "year": 2014,
        "contributor": "Microsoft COCO group",
        "date_created": "2015-01-27 09:11:52.357475"
    }

annotations["annotations"] = []
annotations["images"] = []

for i in dataset:
    image_dict = {
            "license": 3,
            "url": "http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg",
            "file_name": "COCO_val2014_000000391895.jpg",
            "id": 391895,
            "width": 640,
            "date_captured": "2013-11-14 11:18:45",
            "height": 360
        }
    image_dict["id"] = i["id"]
    temp_dict = {}
    temp_dict["image_id"] =  i["id"]
    temp_dict["id"] =  i["id"]
    temp_dict["caption"] = i["conversations"][1]["value"]
    annotations["annotations"].append(temp_dict)
    annotations["images"].append(image_dict)

with open("caption_annotation_cap.json","w") as f:
    f.write(json.dumps(annotations,indent=2))

predictions = []
for i in dataset:
    temp_dict = {}
    temp_dict["image_id"] =  i["id"]
    temp_dict["caption"] = i["predict"]
    predictions.append(temp_dict)


with open("caption_annotation_cap.json","w") as f:
    f.write(json.dumps(predictions,indent=2))


# create coco object and coco_result object
coco = COCO("caption_annotation_cap.json")
coco_result = coco.loadRes("caption_prediction_cap.json")

# create coco_eval object by taking coco and coco_result
coco_eval = COCOEvalCap(coco, coco_result)

# evaluate on a subset of images by setting
# coco_eval.params['image_id'] = coco_result.getImgIds()
# please remove this line when evaluating the full validation set
coco_eval.params['image_id'] = coco_result.getImgIds()

# evaluate results
# SPICE will take a few minutes the first time, but speeds up due to caching
coco_eval.evaluate()

# print output evaluation scores
for metric, score in coco_eval.eval.items():
    print(f'{metric}: {score:.3f}')