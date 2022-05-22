import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

import utils

if __name__ == "__main__":
    outdir = Path("processed-dev")
    outdir.mkdir(exist_ok=True)

    # read in train
    full_jsonlist = utils.load_jsonlist("wiki.jsonlist")
    full_counts = utils.load_sparse("processed/train.npz")
    full_ids = utils.load_json("processed/train.ids.json")

    # split into a dev set
    train_jsonlist, dev_jsonlist, train_counts, dev_counts, train_ids, dev_ids = (
        train_test_split(
            full_jsonlist,
            full_counts,
            full_ids,
            test_size=0.25,
            random_state=11225
        )
    )

    # save
    utils.save_jsonlist(train_jsonlist, Path(outdir, "train.jsonlist"))
    utils.save_sparse(train_counts, Path(outdir, "train.npz"))
    utils.save_json(train_ids, Path(outdir, "train.ids.json"))

    utils.save_jsonlist(dev_jsonlist, Path(outdir, "dev.jsonlist"))
    utils.save_sparse(dev_counts, Path(outdir, "dev.npz"))
    utils.save_json(dev_ids, Path(outdir, "dev.ids.json"))

    