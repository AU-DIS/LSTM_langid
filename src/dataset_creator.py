import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence, Set
import jsonlines
import pyconll
from sklearn.model_selection import KFold
import re
from language_datasets import LangIDDataSet
from string import digits


def make_jsonl_from_UD20(dataset_path: Path, langs: Set[str], writer, exact: bool, max_window: int, min_window: int, num_datapoints=1000000):
    """
    Converts CoNLL-U files to JSONL files.

    :param dataset_path: Path where UD is located
    :param langs: Set of languages to read from UD
    :param writer: jsonl writer
    :param max_window: Maximum length of non-overlapping windows.
    """

    lang_limit = {lang: 0 for lang in langs}
    for f in dataset_path.glob("*/*.conllu"):
        lang_code = f.name[:2]
        if lang_code in langs:
            counter = lang_limit[lang_code]
            if counter > num_datapoints:
                continue
            logging.info(f"Processing file: {f}")
            conll_obj = pyconll.load_from_file(f)
            for sentence in conll_obj:
                full_string = sentence.text
                windows = get_windows_from_text(full_string, max_window, exact=exact)
                counter += len(windows)
                if counter > num_datapoints:
                    break
                for w in windows:
                    writer.write({'text': w, 'label': lang_code})
            lang_limit[lang_code] = counter


def clean_sentence(line):
    # We remove some special characters and fix small errors in the data, to improve the quality of the data
    line = line.replace("\n", '') #{"text": "- Mor.\n", "label": "da"}
    line = line.replace("- ", '') #{"text": "- Mor.", "label": "da"}
    line = line.replace("_", '') #{"text": "- Mor.", "label": "da"}
    line = line.replace("\\", '')
    line = line.replace("\"", '')
    line = line.replace("  ", " ")
    remove_digits = str.maketrans('', '', digits)
    line = line.translate(remove_digits)
    words = line.split()
    new_words = []
    # Below fixes large I instead of l. Does not catch everything, but should also not really make any mistakes either
    for word in words:
        clean_word = word
        s = clean_word
        if clean_word[1:].__contains__("I"):
            indices = find(clean_word, "I")
            for indx in indices:
                if clean_word[indx-1].islower():
                    if len(clean_word) > indx + 1:
                        if clean_word[indx+1].islower():
                            s = s[:indx] + "l" + s[indx + 1:]
                    else:
                        s = s[:indx] + "l" + s[indx + 1:]
        new_words.append(s)
    new_line = " ".join(new_words)
    return new_line


def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]


def make_jsonl_from_opensub(dataset_path: Path, langs: Set[str], writer, exact:bool, max_window: int, min_window: int = 10,
                            num_datapoints=100000):
    lang_limit = {lang: 0 for lang in langs}
    for f in dataset_path.glob("*.txt"):
        langcode = f.name[:-4]
        if langcode in langs:
            logging.info(f"Processing file: {f}")
            counter = lang_limit[langcode]
            if counter > num_datapoints:
                continue
            file = f.open(mode="r", encoding="utf8")
            for line in file.readlines():
                line = clean_sentence(line)
                if counter < num_datapoints:
                    windows = get_windows_from_text(line, max_window, min_window=min_window, exact=exact)
                    counter += len(windows)
                    for sentence in windows:
                        writer.write({'text':sentence, 'label':langcode})
                else:
                    break
            lang_limit[langcode] = counter
            file.close()


def validate_sentence(current_window, min_window):
    if len(current_window) < min_window:
        return False
    if not re.search('[a-zA-Z]', current_window):
        return False
    return True


def get_windows_from_text(text: str, max_window: int, min_window: int = 10, exact: bool = False) -> Sequence[str]:
    """
    Returns a list of non-overlapping widows of up to max_window characters that always start at
    the beginning of a word.
    :param text: text to split
    :param max_window: max length of a window in characters
    :return: list of windows
    """
    from queue import SimpleQueue
    text = clean_sentence(text)
    words = SimpleQueue()
    for w in text.split(" "):
        words.put(w)
    windows_to_return = []
    current_window = ""
    while not words.empty():
        next_word = words.get()
        sep_size = 1
        if len(current_window) == 0:
            sep_size = 0
        if len(current_window) + sep_size + len(next_word) <= max_window:
            if sep_size > 0:
                current_window += " "
            current_window += next_word
        else:
            current_window = clean_sentence(current_window)
            if validate_sentence(current_window, min_window):
                if not exact:
                    windows_to_return.append(current_window)
                else:
                    windows_to_return.append(current_window[0:min_window])
            current_window = next_word
    # Remember to add the content of the last window
    current_window = clean_sentence(current_window)
    if validate_sentence(current_window, min_window) and not exact:
        windows_to_return.append(current_window)
    return windows_to_return


def split_data_set(dataset: LangIDDataSet, out_path: Path, k: int, seed: int):
    """
    Splits a dataset into the given number of folds using a random choice with provided seed.
    :param dataset: dataset to split
    :param out_path: where to save each split (directory). Each split is saved in a file x.jsonl
        where x is the split ID.
    :param k: number of splits to create
    :param seed: seed for splitter
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for fold_id, (_, test_index) in enumerate(kf.split(dataset)):
        logging.info(f"Writing fold: {fold_id}")
        with jsonlines.open(out_path / f"{fold_id}.jsonl", mode='w') as writer:
            for i in test_index:
                writer.write(dataset[i])


class ParserWithUsage(argparse.ArgumentParser):
    """ A custom parser that writes error messages followed by command line usage documentation."""

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def get_substring(line: str, substring_length: int):
    line['text'] = line['text'][0:substring_length]
    return line


def make_matching_split(dataset_path: Path, num_splits: int, substring_length: int):
    test_dataset_path = dataset_path/"sub_dataset"
    test_dataset_path.mkdir(exist_ok=False)
    for i in range(0, num_splits):
        with jsonlines.open(dataset_path/f"{i}.jsonl") as reader:
            with jsonlines.open(dataset_path/"sub_dataset"/f"{i}.jsonl", mode="w") as writer:
                logging.info(f"Writing eval fold: {i}")
                for line in reader:
                    str = get_substring(line, substring_length)
                    writer.write(str)

def main():
    languages_to_read = "da, en, sv, no, de, cs, es, fr, pt, it, tr, nl, fi, pl, ro, hu, lt, ca, hr, et"
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO,
                        datefmt='%m/%d/%Y %H:%M:%S')
    parser = ParserWithUsage()
    parser.description = "Converts UD or opensubtitles data to k-splits ready to use in experiments."
    parser.add_argument("--folds", help="Number of folds for cross-validation", default=10,
                        type=int)
    parser.add_argument("--seed", help="Seed for the cross-fold splitting", default=42,
                        type=int)
    parser.add_argument("--max-window", help="Max window size in characters", default=50,
                        type=int)
    parser.add_argument("--min-window", help="Min window size in characters", default=10, type=int)
    parser.add_argument("--data_source", help="Is the provided path for 'UD20' or 'opensubtitles'?", required=True, type=str, )
    parser.add_argument("--dataset_path", help="Path to UD directory", required=True, type=Path)
    parser.add_argument("--output_path", help="Path to directory where to save the processed output",
                        type=Path, required=True)
    parser.add_argument("--force", "-f", help="Whether to overwrite the output directory",
                        action="store_true")
    parser.add_argument("--languages", help="The languages to be included by language code, as commaseperated string", default=languages_to_read, type=str)
    parser.add_argument("--exact_length", help="whether the string needs to be an exact length or contain full words, this will be = min_length", default=False)
    parser.add_argument("--eval_length", help="how long the evaluation set should be", required=True, type=int)
    args = parser.parse_args()

    k: int = args.folds
    seed: int = args.seed
    dataset_path: Path = args.dataset_path
    out_path: Path = args.output_path
    force: bool = args.force
    max_window: int = args.max_window
    min_window: int = args.min_window
    languages_to_read: set = {str(item).strip() for item in args.languages.split(',')}
    data_source: str = args.data_source
    exact_length: bool = args.exact_length
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
        raise ValueError(f"Path does not exist: {dataset_path}")

    output_file = out_path / "all.jsonl"
    # languages_to_read = {'da', 'en'}
    meta = {}
    if data_source == "UD20":
        with jsonlines.open(output_file, mode='w') as writer:
            make_jsonl_from_UD20(dataset_path, languages_to_read, writer, exact_length, max_window, min_window)
        logging.info(f"Finished writing data, splitting into {k} sections.")
    elif data_source == "opensubtitles":
        with jsonlines.open(output_file, mode='w') as writer:
            make_jsonl_from_opensub(dataset_path, languages_to_read, writer, exact_length, max_window, min_window)
    dataset = LangIDDataSet(output_file)
    #region Metainformation
    for key, value in vars(args).items():
        meta[key] = str(value)
    meta["num_examples"] = len(dataset)
    logging.info("Writing meta information")
    out_meta = out_path / "meta.json"
    with out_meta.open(mode="w") as o:
        import json
        json.dump(meta, o, indent=4)
    #endregion

    split_data_set(dataset, out_path=out_path, k=k, seed=seed)
    make_matching_split(out_path, k, evaluation_length)
    logging.info("DONE")


if __name__ == "__main__":
    main()
