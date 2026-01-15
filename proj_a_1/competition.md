256419
timeout 300 minizinc --solver gecode -a enroll.mzn data/competition.dzn --fzn-flags "-restart luby -restart-scale 250" 2>&1 | tee /mnt/adata-disk/projects/agh/cp/project-a-group-enroll/temp_sol.txt


