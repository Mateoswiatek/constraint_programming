# Constraint Programming — Realistic Use Case

1. Check the ``handout.pdf`` for the instructions.
2. Solve the problem! Do not forget about commiting your changes.
3. You profile your model with the ``profile.sh`` script. It requires decent versions of ``bash`` and ``MiniZinc`` binaries in ``$PATH``). 

## Visualization

If you want to visualize your solution, you have to:

1. Install ``gnuplot`` and make sure it's available in ``$PATH``
2. Compile ``portschedule_draw.cpp``, i.e. ``g++ portschedule_draw.cpp -o visualizer``.
3. Store solution you want to visualize in a file, i.e. ``solution.sol``. Example of an solution to visualize is displayed below:

    ```westend  = [8, 1, 0, 4, 0, 2, 3, 4];
    eastend  = [12, 4, 2, 9, 4, 4, 4, 8];
    stack    = [0, 11, 0, 20, 31, 4, 0, 2];
    endstack = [8, 17, 4, 30, 39, 8, 2, 10];
    reclaim  = [8, 17, 4, 30, 39, 8, 2, 10];
    finish   = [12, 20, 6, 35, 43, 10, 3, 14];
    which    = [3, 1, 2, 1, 1, 2, 1, 1];
    obj = 45;
    ```

4. Run visualier, i.e. ``./visualizer solution.sol``
5. Now there should be file ``solution.sol.ps`` you can view in your favorite PDF reader.

## Setup 

* [ ] Make sure, you have a **private** group (you don't have to create a new one, if you have a nice one already)
  * [how to create a group](https://docs.gitlab.com/ee/user/group/#create-a-group)
* [ ] Add @bobot-is-a-bot as the new group member (role: **maintainer**)
  * [how to add an user](https://docs.gitlab.com/ee/user/group/#add-users-to-a-group)
* [ ] Fork this project into your private group
  * [how to create a fork](https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#creating-a-fork)


## How To Submit Solutions

* [ ] Clone repository: git clone:
    ```bash 
    git clone <repository url>
    ```
* [ ] Solve the exercises 
    * use MiniZincIDE, whatever
* [ ] Commit your changes
    ```bash
    git add <path to the changed files>
    git commit -m <commit message>
    ```
* [ ] Push changes to the gitlab master branch
    ```bash
    git push -u origin master
    ```

The rest will be taken care of automatically. You can check the `GRADE.md` file for your grade / test results. Be aware that it may take some time (up to one hour) till this file appears / gets updated.  
