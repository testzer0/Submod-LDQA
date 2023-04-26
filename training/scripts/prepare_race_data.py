import os
import json
import sys

from unidecode import unidecode

in_dir = "/raid/infolab/adithyabhaskar/submodopt/submission/data/RACE/"

def save_race():   
    for split in ['train', 'dev']:
        in_path = os.path.join(in_dir, "{}.json".format(split))
        out_path = "data/race/{}.jsonl".format(split)

        data = json.load(open(in_path))
        with open(out_path, 'w+') as outfile:
            for entry in data:
                for key in entry:
                    if key.startswith("option") or key == "query":
                        entry[key] = " " + entry[key]
                json.dump(entry, outfile)
                outfile.write('\n')

        config = {
            "num_choices": 4
        }        

        with open('data/race/config.json', 'w+') as f:
            json.dump(config, f)
        
save_race()