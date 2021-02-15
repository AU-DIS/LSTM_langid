from jsonlines import jsonlines
import logging
from pathlib import Path
import glob
from dataset_creator import ParserWithUsage

def fasttext_from_jsonl(file_path_input, writer):
    with jsonlines.open(file_path_input) as reader:
        for line in reader:
            example = line['text'] + " __label__" + line['label'] + "\n"
            writer.write(example)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    parser = ParserWithUsage()
    parser.description = "Takes a directory of JSONL files and outputs valid files for fasttext training"
    parser.add_argument("--dataset_path", help="Path to JSONL-file", required=True, type=str)
    parser.add_argument("--output", help="Path to directory where to save the processed output",
                        type=Path, required=True)
    parser.add_argument("--force", "-f", help="Whether to overwrite the output directory",
                        action="store_true")
    args = parser.parse_args()
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
        raise ValueError(f"Path does not exist: {dataset_path}")

    files = []
    for filename in glob.iglob(dataset_path_str + '**/*' + '.jsonl', recursive=True):
        if "all" not in filename:
            files.append(filename)

    logging.info("Reading files: " + files.__str__())
    for i in range(len(files)):
        filename = files[i]
        logging.info("Reading file: " + filename.__str__())
        out_file = out_path / (str(i) + ".txt")
        writer = open(out_file, mode='w', encoding='utf-8')
        fasttext_from_jsonl(filename, writer)
        writer.close()
    logging.info("DONE")


if __name__ == "__main__":
    main()
