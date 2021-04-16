#!/bin/bash

set -e

CMD="bash"

(cd serial && eval $CMD make > /dev/null)
(cd framework && eval $CMD make > /dev/null)

echo "run,data,k,iters,type,time"
run=$(( 0 ))
for data in $@; do
	for iters in 10 50 100; do
		for k in 8 64 128 256; do
                        for exe in ./serial/k-means_serial ./framework/k-means; do
                                echo -n "$run,$data,$k,$iters,$exe,"
                                eval $CMD $exe $data $k $iters centroids_file assignments_file
                                diff centroids_file centroids_file
                        done
			run=$(( $run + 1 ))
                done
        done
done
