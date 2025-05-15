#  New Beginnings: GSoC

**Welcome to my first-ever blog post!**  
This year, I set out with a clear goal: to dive into the world of open source. I had heard about Google Summer of Code (GSoC) before coming to college. In my freshman year, i got to know how it opens the doors to open-source and how people who have been contributing to these organizations mentor you. And today as a sophomore, I'm very happy to share that I've been selected as a GSoC'2025 Contributer for Git Organization,
for the project "Refactoring in order to reduce Git's global state" under the mentorship of by [Christian Couder](https://github.com/chriscool) from [GitLab](https://about.gitlab.com/)  and [Ghanshyam Thakkar](https://github.com/spectre10).  My fork of git where I'll be pushing before sending it to the mailing list can be found [here](https://github.com/ayu-ch/git). This will be a series of blogs throughout my GSoC journey. 

Before getting selected I've read blogs of people who were once a mentee, and it inspired me a lot.  So whether you're a student hoping to apply next year, or just curious about open source, I hope this gives you some insight and inspiration. 

### About the Project

Git currently uses a global object called `the_repository`, which refers to a single instance of `struct repository`. Many internal functions rely on this global object rather than accepting a `struct repository` as an explicit parameter. This design inherently assumes a single active repository, making it difficult to support multi-repository use cases and obstructing the long-term goal of libification of Git.

A key architectural limitation is that while `struct repository` encapsulates some repository-specific information, many important environment variables and configuration settings that logically belong to a repository are still stored as global variables, primarily in `environment.c`, not within the `repository` struct. As a result, even if multiple repositories were to exist concurrently, they would still share this global state, leading to incorrect behavior, race conditions, or subtle bugs. 

This project aims to refactor Git's environment handling by relocating global variables into more appropriate local contexts, primarily within struct repository and struct repo_settings. However, some global variables may only apply to specific subsystems. In such cases, rather than placing them in struct repository or struct repo_settings, they should be moved into a context that better reflects their scope.

### Goals

The time period from 8th May to 1st June is Community Bonding period. This time is to read related documentation, get more familiar with the codebase, start engaging with the community.

These are some things which I would do in this period:
- Read contributing, mentoriship program guidelines and go through the Pro Git book.
- Connect and discuss the project's scope and plan with my mentors.
- Try documenting different global variables, how they are used and how it can be moved to their corresponding scope.

---

I'm incredibly excited to begin this journey and contribute to a tool which is so widely used and respected. Over the next few months, I'll be sharing updates, learnings, challenges, and milestones as I work through my project.

-Ayush:)