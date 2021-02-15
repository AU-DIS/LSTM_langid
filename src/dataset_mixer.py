import logging
from pathlib import Path
import glob
from dataset_creator import ParserWithUsage, split_data_set, make_matching_split
from language_datasets import LangIDDataSet

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    parser = ParserWithUsage()
    parser.description = "Combines data given in a directory of JSONL-files " \
                         "to a new JSONL files with new k-folds"
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
    parser.add_argument("--eval_length", help="how long the evaluation set should be", required=True, type=int, default=10)
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
    evaluation_length: int = args.eval_length

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
    make_matching_split(out_path, k, evaluation_length)
    logging.info("DONE")


if __name__ == "__main__":
    main()
