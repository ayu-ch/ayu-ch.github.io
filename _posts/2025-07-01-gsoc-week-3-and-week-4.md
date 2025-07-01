# GSoC: Week 3 and 4

This blog covers both Week 3 and Week 4, since there wasn’t enough content in Week 3 to create a separate post. I decided to combine them into a single update.

## Week 3
As I talked about in week 2, I posted a v4 of **[environment: remove sparse-checkout related global variables](https://lore.kernel.org/git/cover.1750157825.git.ayu.chandekar@gmail.com/)**. In this week, I got comments from Junio that calls to `prepare_repo_settings()` are missing in some code paths, so I had to prepare a new patch to make sure that `prepare_repo_settings()` is called before accessing our `settings.sparse_checkout` and `settings.sparse_checkout_cone` settings. 

While I was working on comment line variables as I mentioned in the prevoius blog, I stumbled upon a bug where it showed incorrect comment character when `core.commentString=auto` and we try to rebase. So, I worked on fixing this issue and sent a patch: [commit: avoid scanning trailing comments when 'core.commentChar' is "auto"](https://lore.kernel.org/git/20250626132233.414789-1-ayu.chandekar@gmail.com/T/#md490b421ee816b43325f5671676360538c551363). 

The patch [preload-index: remove dependency on global variables and 'the_repository'](https://lore.kernel.org/git/cover.1749557133.git.ayu.chandekar@gmail.com/) also got merged to master. 

 A lot of my time this week also went in doing some of my resume verification work for college. 

That was pretty much what I did in week-3. 

## Week 4

Week 4 was a packed week because of lots of patches which were there on my table:

-  [builtin/prune: remove dependency on global variables and 'the_repository'](https://lore.kernel.org/git/cover.1749343601.git.ayu.chandekar@gmail.com/#r): This patch was sent to the mailing list long ago, but didn't get any reviews, so I sent a ping to the list about this patch. I soon got reviews on this regarding some small fixes. I sent a v2 of this patch to fix the issues from the comments on v1. The v2 is currently under a discussion.
- [environment: remove sparse-checkout related global variables](https://lore.kernel.org/git/cover.1750157825.git.ayu.chandekar@gmail.com/): After working on this patch to make sure that `prepare_repo_settings()` has been called before accessing the settings, I sent a v5 of the patch. I then got reviews from Junio that the calls to `prepare_repo_settings()` might be too deep in the code unlike the ones in `cmd_foo()` functions. This might cause performance issues if it's called every now and then. So, currently, I am working on this.
- [commit: avoid scanning trailing comments when 'core.commentChar' is "auto"](https://lore.kernel.org/git/20250626132233.414789-1-ayu.chandekar@gmail.com/#r): This is the bug-fix patch which I worked on last week. I got reviews by Junio and Kristoffer, suggesting some changes in commit message and test which I added. So, after making the changes, I sent a v2. Got some comments from Phillip for the test, so I did those changes and sent a v3. Phillip says that the patch looks good, but we can add a new commit so that the `comment_line_str` is set to `#` when `core.commentString` is set to auto.

Overall, I think it's going a bit slow due to multiple patches under review at the same time, which makes it kind of difficult to context-switch and also try to come up with a new patch. But will do better this time!

Thanks a lot to Christian from GitLab for mentoring me with patches, especially helping me write good commit messages:)

See you soon!
-Ayush:)
