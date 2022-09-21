#! /bin/bash

modified_folders=$(git diff --name-only HEAD^ HEAD functions/ | xargs -L1 dirname | uniq)

for function in $modified_folders; do
    FILE="${function}/cmd.sh"
    if test -f $FILE; then
        echo "Changes have been made to '$function' function";
        chmod +x $FILE && $FILE
    fi
done
