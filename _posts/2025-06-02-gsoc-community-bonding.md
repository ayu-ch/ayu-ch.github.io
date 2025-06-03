# GSoC : Community Bonding Period
  
So the GSoC's Community Bonding period just got over.
Initially, me, my mentors and other mentees along with their mentors had an introductory meet.  Afterwards,  I had a more focused dicussion with my mentors about the project. In that meeting, we discussed the overall approach: I should aim to remove the definition `#define USE_THE_REPOSITORY_VARIABLE` file by file by removing `the_repository` and other global variables. 
 After the meet, I began exploring different files, see previous attempts of refactoring done on it and trying to figure out why it was rejected or see other attempts which were successful and understand their approach, from the mailing list. 

One of the things which I learnt is that for some commands, repository passed can be `NULL` as they can be called from outside the repository and we should keep that in mind while trying to refactor those commands.

I then chose few files and global variables to work with:
 
- `add-patch.c`  : Tackling this file to remove the definition was difficult as it had a global variable called `comment_line_str`. My mentor, Christian suggested me that I can house this inside a struct `config_set` which lies in the struct `repository`. So, my next goal is to try to store the global variable in that struct.
- `blame.c` : I was able to remove the definition from the file but it would be better if I try to include in a patch series where builtin/blame.c is also refactored.
- `repository_format_precious_objects` : I came across this global variable and I moved it into the struct `repository`. Along with this, I also tackled two files which included this variable: 
  - `builtin/prune.c` : I refactored this file thus removing the defintion.
  - `builtin/repack.c` : I tried to refactor this but it contained functions which were of type `each_ref_fn` which do not support repository and also the global function `is_bare_repository()`.  So I decided to drop this file. 
  I plan to submit the patch series (excluding `repack.c`) to the mailing list this week.
- `core_apply_sparse_checkout` : This variable houses the `core.sparsecheckout` setting. I moved this setting to the struct `repo_settings`. I will also send a patch for this soon this week. 

---
At first, I kept on going down the rabbit hole as I traced all the code paths and hence, got confused. But after talking to my mentors, I gained some clarity and felt pretty confident with time. I thank my mentors Christian Couder (GitLab) and Ghanshyam Thakkar and also Patrick Steinhardt (GitLab) and Jialuo She for solving some of my doubts.
I'm excited to continue working on this project and sharing my experience in the weeks to come.
See you soon!

-Ayush:)