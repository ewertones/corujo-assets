#! /bin/bash

# latest Commit
LATEST_COMMIT=$(git rev-parse HEAD)

for folder in functions/* scheduler/*; do
    COMMIT=$(git log -1 --format=format:%H --full-diff $folder/*)

    if [[ $COMMIT = $LATEST_COMMIT ]]; then

        echo "Changes have been made to the '$folder' model";

        /bin/bash "${folder}"/cmd.sh

    else

        echo "No change detected in the '${folder}' folder";

    fi;
done
