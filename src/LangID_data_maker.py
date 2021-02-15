import logging
from pathlib import Path
import glob
from dataset_creator import ParserWithUsage, split_data_set
from language_datasets import LangIDDataSet


def make_data(inp_path, out_path):
    dataset = LangIDDataSet(inp_path)
    current_label = dataset[0]['label']
    f = open(out_path + "/" + current_label + "0.txt", "w")
    ldict = {current_label: 0}
    for item in dataset:
        label = item['label']
        text = item['text']
        if label != current_label:
            current_label = label
            f.close()
            if label not in ldict:
                ldict = {label: 0}
            else:
                ldict[label] = ldict[label] + 1
            cnum = str(ldict[label])
            f = open(out_path + "/" + current_label + cnum + ".txt", "w")
        f.write(text + "\n")
    f.close()

def make_folds(inp_path, out_path, folds, seed):
    dataset = LangIDDataSet(inp_path)
    split_data_set(dataset, out_path, seed, folds)
    for fold_id in range(folds):
        f_path = out_path / f"{fold_id}.jsonl"

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    parser = ParserWithUsage()
    parser.description = "Creates files for langid training from JSONL file"
    parser.add_argument("--folds", help="Number of folds for cross-validation", default=10,
                        type=int)
    parser.add_argument("--seed", help="Seed for the cross-fold splitting", default=42,
                        type=int)
    parser.add_argument("--dataset_path", help="Path to JSONL-files directory", required=True, type=str)
    parser.add_argument("--output", help="Path to directory where to save the processed output",
                        type=Path, required=True)
    parser.add_argument("--force", "-f", help="Whether to overwrite the output directory",
                        action="store_true")
    parser.add_argument("--suffix", help="Files must end in suffix.jsonl. Default suffix is empty string", default="",
                        type=str)
    args = parser.parse_args()
    k: int = args.folds
    suffix: str = args.suffix
    seed: int = args.seed
    dataset_path_str: str = args.dataset_path
    if not dataset_path_str.endswith('/'):
        dataset_path_str += '/'
    dataset_path: Path = Path(dataset_path_str)
    out_path: Path = args.output
    force: bool = args.force

    logging.info("STARTED")
    if out_path.exists():
        msg = f"Output path already exists: {out_path}."
        if force:
            logging.warning(
                f"{msg} Will overwrite.")
            import shutil
            shutil.rmtree(out_path)
            out_path.mkdir(exist_ok=False)
        else:
            raise ValueError(f"{msg} Use --force to overwrite")
        if out_path.is_file():
            raise ValueError(f"Output path is a file. Please provide a directory: {out_path}")
    else:
        out_path.mkdir(exist_ok=False)
    if not dataset_path.exists():
        raise ValueError(f"Path for does not exist: {dataset_path}")

    files = []
    for filename in glob.iglob(dataset_path_str + '**/*' + suffix + '.jsonl', recursive=True):
        files.append(filename)
    logging.info("Reading files: " + files.__str__())
    dataset = LangIDDataSet.make_from_files(files)
    meta = {"folds": k, "num_examples": len(dataset)}
    logging.info("Writing meta information")
    out_meta = out_path / "meta.json"
    with out_meta.open(mode="w") as o:
        import json
        json.dump(meta, o, indent=4)
    output_file = out_path / "all.jsonl"
    logging.info("Creating new files")
    dataset.save_data_as_jsonl(output_file)
    split_data_set(dataset, out_path=out_path, k=k, seed=seed)
    logging.info("DONE")


if __name__ == "__main__":
    main()
