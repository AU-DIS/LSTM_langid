#!/bin/sh
mkdir datasets
cd datasets || exit
mkdir "UD20_raw_files"
cd "UD20_raw_files" || exit
curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105{/ud-treebanks-v2.5.tgz,/ud-documentation-v2.5.tgz,/ud-tools-v2.5.tgz}
tar -xvzf ud-treebanks-v2.5.tgz
rm ud-documentation-v2.5.tgz
rm ud-tools-v2.5.tgz
rm ud-treebanks-v2.5.tgz