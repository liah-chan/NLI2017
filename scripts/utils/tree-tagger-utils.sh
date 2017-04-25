# /bin/bash

###get lemma
# if [ -f "$2"] ; then
# rm "$2"
# fi
rm "$2"
while IFS='' read -r line || [[ -n "$line" ]]; do
	label=$(cut -d$'\t' -f2  <<< "$line")
    var=$(cut -d$'\t' -f1  <<< "$line" | tree-tagger-english | cut -d$'\t' -f3 | tr '\n' ' ') 
    echo -e "$var\t$label">> "$2"
    # echo -e "\n" >> "$2"
done < "$1"
###example:
###bash tree-tagger-utils.sh ../../data/dev_data.word.txt ../../data/dev_data.lemma.txt


#get pos
# rm "$2"
# while IFS='' read -r line || [[ -n "$line" ]]; do
# 	label=$(cut -d$'\t' -f2  <<< "$line")
#     var=$(cut -d$'\t' -f1  <<< "$line" | tree-tagger-english | cut -d$'\t' -f2 | tr '\n' ' ') 
#     echo -e "$var\t$label">> "$2"
#     # echo -e "\n" >> "$2"
# done < "$1"

###example:
###bash tree-tagger-utils.sh ./NLI/test_data.txt ./NLI/test_data.pos.txt
