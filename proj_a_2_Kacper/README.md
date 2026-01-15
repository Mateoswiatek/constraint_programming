# Group Enroll

This project concerns assigning students to groups given various constraints and preferences.
It has been directly inspired by the system working at the AGH University (Faculty of Computer Science, Electronics and Telecommunications).

## :boxing_glove: Competition :boxing_glove:

Soon you will be able to check the :boxing_glove: [the competition scoreboard on wiki](https://gitlab.com/agh-courses/25/cp/project-a-group-enroll/-/wikis/Scoreboard). :boxing_glove:

## Instructions

1. Fork this project into a **private** group:
2. Add @bobot-is-a-bot as the new project's member (role: `maintainer`) 
4. Read this Readme till the end for the instructions.
5. Solve the problem!
6. Automated tests will be run periodically to check quality of your model. The results will be available in the `GRADE.md`
7. If done before the deadline, contact teacher via Teams, so he can check it earlier.
8. To take part in the competition create file `competition.sol` containing solution for the `competition.dzn` instance.

## Problem Details

Imagine working in the faculty administration at the AGH UST in Krakow. Every year the same story: first you have to prepare the schedule for the whole year, later you have to assign students to the available groups. Recently your faculty started to notice the obnoxious difficulty of those tasks and tried to introduce some automation. Unfortunately it doesn't work as well as you hoped. Therefore you are going to the matter into your own hands. To be realistic, you will focus on the group assignment as it's less disruptive than the whole schedule. After spending some time under the shower and having several eureka moments, you came with the following definition of the problem.

So, first, the input has to contain the current schedule of all the involved groups[^group] over the set of days[^day]. Only then we will focus on the students[^student] and their preferences. Finally we will combine them to obtain an objective function and print the output.  

### Schedule 

Each group belongs to a class[^class][^group_class] (e.g. `Constraint Programming: Lab`) with a specified duration[^class_duration] in a well defined time unit[^time]. We also have to know which day[^group_day] and at what hour[^group_start] the groups start their activities. This way we can guarantee that student doesn't attend two different groups at the same time. We shouldn't forget about the space restrictions, each class has an upper limit for the number of students in the single group[^class_size].

Other important factor is that we have to know in what building the group occupies the classroom, so later we can calculate the time required to move between the groups. We have to guarantee that student is able to transport between the buildings in time. Finally the schedule can change during the semester, there can be alternative weeks, some courses may to end in the middle of the semester; it's almost impossible to handle all the irregularities. Therefore we will also define what groups do not conflict each other even if they are happening at the same time according to our simplified schedule. One would be worried that would requires lot of work, but luckily somebody has already preprocessed the data and calculated which groups conflict with each other[^groups_conflicts]!

### Preferences

Given the schedule, we can focus on the students[^student] and their preferences. Each student should belong to exactly one group[^group] per class[^class], but there are exceptions — more about them later...

First, some students live at the campus and they do not care, whether they have breaks between the classes. On the other hand, other would prefer to not sit at the university without any purpose. There is an array[^student_break_importance] telling how much important is lack of breaks for the student.

> Wasting time on long boring breaks is not ok :-1:.

secondly, students assign preferences[^preference][^student_prefers] to the available groups; higher the preference is, more they would like to attend the group. The preference = `-1` means that this group is excluded for the given student. If all the groups of a given class are excluded for the student, it means that they do not attend this class — the exception foretold in the first paragraph of this section. 

> Being assigned to groups one does not prefer is not ok :-1:.

### Objective

Given the schedule and student preferences we have to define what assignments are better than others. Obviously, every student (with already mentioned exceptions) has to be assigned to a single group of each class. Also the assignment has to be feasible (no bilocation allowed), but what makes the assignment better than others?

1. Students should not waste their time on long boring breaks. For each student and each day we calculate the time they have to spend at the university ("end of the last group" - "start of the of the first group"). We sum these values for the whole week and call it **time spent at the university**. Then we calculate what is the minimal time, the student has to spend at the university — total duration of al the student's classes.  and how much they spend on learning (sum all the classes' durations this day). Difference between these values is the time wasted between classes. Due to the fact that during the semester classes may change, this value in theory could go below zero, but we should assume the `0` is the minimal value. The wasted time is summed for each student and normalized to full hours[^normalized]. We will call this value **break disappointment**.
2. Students should not attend groups they hate. For each student and each class we calculate difference between their favorite group and the one they have been assigned. We sum those values for each student and will call this value **preference disappointment**.

Next for each student we calculate the following **total disappointment**, being the weighted average of **break** and **preference** disappointments, where weight of the first one is given in the [^student_break_importance]`student_break_importance` array and the total sum of weights ie equal to `10`. This way every student has the same impact on the objective. It's presented in the formula below, `ceil_div` is the division "rounded up" (e.g. `ceil_div(5, 3) = 2`):

**total disappointment[student]** =  
                ceil_div(**student break importance[student]** × **break disappointment[student]** +  
                (`10` - **student break importance[student]**) × **preference disappointment[student]**), `10`)

Finally, our objective is to :exclamation:**minimize sum of squared total disappointments**:exclamation:

### Output

An example output for `trivial.dzn` is presented below:

```
assignment = [{3,4},{1,6},{3,5},{2,4},{1}];
total_break_disappointment = 0;
total_preference_disappointment = 2;
objective = 2;
```

where:
- `assignment = ` array of sets containing groups assigned to the students, e.g. `assignment[0] = {3,4}` means that the student with index `0` has been assigned to groups `3` and `4` — notice that the last student in the example attends only a single class.
- `total_break_disappointment = ` sum of **break disappointment[student]** (before multiplying by the corresponding weight);
- `total_preference_disappointment = ` sum of **preference disappointment[student]**;
- `objective = ` objective value.

The last three lines (objective components) will be useful for the grader script to check if your model calculates them correctly.

## Parameters Explained

[^group]: `Group` is a set containing unique identifiers of groups.
[^group_class]: `group_class` array maps groups to the corresponding classes[^class].
[^group_day]: `group_day` array defines what day[^day] the groups start.
[^group_start]: `group_start` array defines at what hour (in the instance specific time unit[^time]) the groups start.
[^groups_conflicts]: `groups_conflicts` matrix defines which groups conflict with each other.
[^class]: `Class` is a set containing unique identifiers of classes.
[^class_duration]: `class_duration` array contains classes' durations in the instance specific time unit[^time].
[^class_size]: `class_size` array contains how many students can attend a single group of this class.
[^day]: `Day` is a set containing unique identifiers of days.
[^time]: `Time` set contains available time indexes, e.g. `0` is the beginning of the day and the biggest value is the end of the day.
[^normalized]: normalized time is number of hours (rounded up) in the give time period, you can calculate it with `ceil_div(period, n_time_units_in_hour)`.
[^preference]: `Preference` is a set of possible preferences over the groups, the `-1` means that the student is not allowed to be assigned to the given group.
[^student]: `Student` is a set containing unique identifiers of students.
[^student_break_importance]: `student_break_importance` array tells how much important is *not* having long breaks for the student. The values belong to range `0..5`: `0` — student doesn't care, `5` – student hates long breaks.
[^student_prefers]: `student_prefers` matrix contains prefences[^preference] student[^student] assigns to various groups[^group]. If all the groups of the given class have assigned the lowest preference it means that student doesn't attend this class.
