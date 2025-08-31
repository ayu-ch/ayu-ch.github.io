# GSoC: Final report
  
## Final Report

Hello everyone! 
It's hard to believe how quickly summer has passed. It feels like I was writing my first blog yesterday. Since then, it has been a great journey! I feel very honoured to have been a part of Google Summer of Code with Git. I am extremely thankful to my mentor, Christian Couder from GitLab, who guided me through this project with patience and also encouragement.

### Quick Summary
So, I took the project of refactoring the global state in Git, i.e, trying to remove the global variables and placing them in their respective contexts. This is a long and ongoing project in Git which will still be continued in the organization after my project ends.

### My experience

It was fun and challenging contributing to Git. The contribution workflow is quite different as compared to other organizations and the reviewing process takes quite some time. However, I got used to it quickly. This process, although slow, is very thorough. And having a good amount of active contributors, there are a lot of reviews and we get to see everyone's perspective. Along with communicating regularly with my mentor, I really didn't felt stressed at any time. There were some moments where it was a bit challenging but we got through it! Will talk about it more later in this blog

### Patches:

Here are the patch series I posted throught my GSoC and their status:

- preload-index: remove dependency on global variables and 'the_repository' [(thread)](https://lore.kernel.org/git/cover.1749557133.git.ayu.chandekar@gmail.com/#r) 
    - environment: remove the global variable 'core_preload_index' [(master)](https://git.kernel.org/pub/scm/git/git.git/commit/?id=b1d47b464e553331c555f122c5e341dfbfb618bd)
    - preload-index: stop depending on 'the_repository' [(master)](https://git.kernel.org/pub/scm/git/git.git/commit/?id=1fde1c5daf399bb2d645261e38a7f4b8b1de04c6)
- builtin/prune: remove dependency on global variables and 'the_repository' [(thread)](https://lore.kernel.org/git/cover.1749343601.git.ayu.chandekar@gmail.com/#r)
    - repository: move 'repository_format_precious_objects' to repo scope [(master)](https://git.kernel.org/pub/scm/git/git.git/commit/?id=44e300a97480ef272a596e02b912b72528043193)
    - builtin/prune: stop depending on 'the_repository' [(master)](https://git.kernel.org/pub/scm/git/git.git/commit/?id=7cd03a555a0d951759a5bce201ad0686c0fc8b12)
-  commit: avoid scanning trailing comments when 'core.commentChar' is "auto" [(thread)](https://lore.kernel.org/git/20250626132233.414789-1-ayu.chandekar@gmail.com/#r)
    - commit: avoid scanning trailing comments when 'core.commentChar' is "auto" [(master)](https://git.kernel.org/pub/scm/git/git.git/commit/?id=e69bbfa2947ee733a2b76f1dc28ceaf415368f2a)
    - config: set comment_line_str to "#" when core.commentChar=auto [(master)](https://git.kernel.org/pub/scm/git/git.git/commit/?id=92b7c7c9f5fde4972a653ddb90eca52c592de07e)
- builtin/fmt-merge-msg: remove dependency on global variables and 'the_repository' [(thread)](https://lore.kernel.org/git/cover.1753804956.git.ayu.chandekar@gmail.com/T/#m5e73b5fae2fc0abea0f0818caf0ed736504b4038)
    - environment: remove the global variable 'merge_log_config' [(master)](https://git.kernel.org/pub/scm/git/git.git/commit/?id=9a49aef8dcdf899e94cddab14eacc7118c611524)
    - builtin/fmt-merge-msg: stop depending on 'the_repository' [(master)](https://git.kernel.org/pub/scm/git/git.git/commit/?id=22d421fed9cd5757a9da5a97e5b53ded54e93fe9)
- environment: remove sparse-checkout related global variables [(thread)](https://lore.kernel.org/git/cover.1750157825.git.ayu.chandekar@gmail.com/)
    - environment: move access to "core.sparsecheckout" into repo_settings [(seen)](https://git.kernel.org/pub/scm/git/git.git/commit/?h=seen&id=2facab3bc779be3dfdaa453e424e775cadf243c8)
    - environment: move access to "core.sparsecheckoutcone" into repo_settings [(seen)](https://git.kernel.org/pub/scm/git/git.git/commit/?h=seen&id=1fd8bf149677cf9ba7e388668a648d4281dbdd7f)
    - environment: remove the global variable 'sparse_expect_files_outside_of_patterns' [(seen)](https://git.kernel.org/pub/scm/git/git.git/commit/?h=seen&id=f35bd7460807c5422f1116e72c7a5d86cd8868f4)
(URL for the commits in 'seen' branch may change when it gets merged to master, but thread will be there)

I also have a patch series currently in development which aims to remove the `#define USE_THE_REPOSITORY_VARIABLE` definition from some files so I will be sending it soon.

### Challenges

Although I enjoyed this journey, there were a few challenges which I faced. Here are some of them:

- Trying to determine which global variables to remove. This was confusing because before selecting something, you don't actually know how complex it would be to remove a specific variable. For eg, It was fairly easy for me to remove the variable `core_preload_index` but it was difficult to choose it because I went through a lot other global variables before this. Variables like `is_bare_repository_cfg` are significantly difficult to remove because there is a lot of code around it, and it's just not easy as you have to make changes to a lot of existing logic.

- Trying to determine where to place a specific global variable. There were 3 options for the most of the time: 1. Placing it in `struct repository`. 2. Placing it in `struct repo_settings`. 3. Localizing the variable by reading it there itself (mostly when it's used very sparsely). The thing with 3rd option is that it's easy to integrate but it has a problem regarding user experience, as if the config for that variable is entered invalid, the user might experience the error mid operation and not while they're setting the config. The other problem which happened was I placed variables around sparse checkout in the `struct repo_settings` initially. A lot later I received comments that this may look better if it was placed in the `struct repository`. So, right now, I'm working on this.

- Sometimes when you're doing some change, there's a certain catch to that change which you know might be questioned by someone and it's hard to address because you don't know any other way to do that change, and when you post it to the mailing list, you get that question. And the thing with me is that I always felt very unsure to reply immediately because I thought maybe I haven't checked thoroughly or maybe I need to give more time and think about the reply and the problem. I really love the reviews because they challenge me to think better and I also appreciate everyone in the community taking out their time to review it, not just the maintainer.

- Another place where I was struggled was when there were multiple patch series in review, I felt it was difficult for me to do the context switch everytime there was a review on a particular patch and then make changes again.

- One area I could have done better was posting my blogs on time. While I started off well, there were phases where I didn’t have significant updates to share, which made me procrastinate on writing.

### What's next?

 I absolutely loved working on Git this summer, and I’ve come to really appreciate the community around it. I definitely want to continue contributing, as I’m really interested in the internals of Git and excited about bringing new ideas and features to the project. One of my most memorable moments was spotting a bug, fixing it myself, and seeing the change go through. It was incredibly motivating and made me want to keep doing more of that work.

Going forward, I plan to:

- Work on reducing global state wherever possible.

- I want to learn more about the internals, see what changes can I do to improve the performance or maybe add new features.

- Dive deeper into the sequencer and commit-related code. As I went through that part while fixing the bug I found, I felt I could add more features there maybe?

### Acknowledgements

I’m really grateful to my mentor from GitLab, Christian Couder, for guiding me throughout this journey. We used to have weekly meetings where he answered my questions and shared his insights. He also taught me how to write good commits, which is something I’ll carry forward in all my future work. His feedback and encouragement made a huge difference. I made a lot of silly mistakes in my code and he always pointed them out before I sent it to the mailing list ;)

A big thanks as well to the Git community. Every time someone took the time to review my patches or share their thoughts, it pushed me to think harder and improve my work. I especially want to thank Junio, Phillip, and Patrick for their valuable feedback and guidance. It has been amazing to be part of such a supportive and knowledgeable group of people.

And of course, thanks to Google Summer of Code for giving me this opportunity. It has been an incredible summer, and I’m really excited to keep contributing!

Thanks for reading the blog!
-Ayush:)