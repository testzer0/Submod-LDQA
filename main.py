from config import *
from utils.globals import *

from models.subset_selection import *
from utils.submodular import *

def main():
    for split in ['train', 'dev']:
        save_selected_sentences_quality("data/subsets/quality/dpr/facLoc/", \
            split=split, n=10, diversity=1, min_length=5, \
            selection_fn=select_sentences_MI, embedder_fn=get_embedding_dpr, \
            call_fn=select_subset_groups_for_question, mode="facility_location", \
            filter_based_on_similarity=False)

if __name__ == '__main__':
    main()